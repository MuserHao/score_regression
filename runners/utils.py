import numpy as np
import matplotlib
import matplotlib.cm as cm
from PIL import Image

import torch
from .sde_lib import *


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_model_fn(model, cond, train=False):
    """Create a function to give the output of the score-based model.
    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.
    Returns:
      A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.
        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(torch.cat((cond, x), 1), labels)
        else:
            model.train()
            return model(torch.cat((cond, x), 1), labels)

    return model_fn


def get_score_fn(sde, model, cond, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.
    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, cond, train=train)

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 99
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, VESDE):

        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model_fn(x, labels)
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn

def DepthNorm(depth, max_depth=1000.0):
    return max_depth / depth


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(value, vmin=10, vmax=1000, cmap="binary"):

    value = value.cpu().numpy()[0, :, :]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0

    cmapper = cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def load_from_checkpoint(ckpt, model, optimizer, epochs, loss_meter=None):

    checkpoint = torch.load(ckpt)
    ckpt_epoch = epochs - (checkpoint["epoch"] + 1)
    if ckpt_epoch <= 0:
        raise ValueError(
            "Epochs provided: {}, epochs completed in ckpt: {}".format(
                epochs, checkpoint["epoch"] + 1
            )
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])

    return model, optimizer, ckpt_epoch


def init_or_load_model(
    depthmodel,
    enc_pretrain,
    epochs,
    lr,
    ckpt=None,
    device=torch.device("cuda:0"),
    loss_meter=None,
):

    if ckpt is not None:
        checkpoint = torch.load(ckpt)

    model = depthmodel(encoder_pretrained=enc_pretrain)

    if ckpt is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if ckpt is not None:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

    start_epoch = 0
    if ckpt is not None:
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch <= 0:
            raise ValueError(
                "Epochs provided: {}, epochs completed in ckpt: {}".format(
                    epochs, checkpoint["epoch"] + 1
                )
            )

    return model, optimizer, start_epoch


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(
            np.asarray(Image.open(file).resize((640, 480)), dtype=float) / 255, 0, 1
        ).transpose(2, 0, 1)

        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)