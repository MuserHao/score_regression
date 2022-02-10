import scipy.integrate as integrate
import tqdm
from .utils import *
from torchvision.utils import save_image
import torch

def reverse_diffusion_sampler(x_mod, rgb_image, scorenet, sigmas, step_lr=0.000008, range=[0., 1.], coupled=True):
    """
    The function make prediction of target map for a given rbg image with reverse diffusion sampler

    :param x_mod: an initial random sample with the shape [repeat, batch, channel, width, height]
    :param rgb_image: the rgb image containing with shape [batch, channel, width, height]
    :param scorenet: the score network takes input of (rgb, target) image and return the score function
    :param sigmas: the noise levels of revese diffusion process
    :param n_steps_each: number of steps each noise level
    :param step_lr: learning rate for the smallest noise level
    :param range: the range of output data
    :return: list of sampled target maps with shapes [batch * repeat , out_channels, width, height]
             and converged sample average as prediction target map [batch , out_channels, width, height]
    """

    targets = []
    batch = x_mod.shape[1]
    repeat = x_mod.shape[0]
    width = x_mod.shape[3]
    channels = x_mod.shape[2]
    rgb_image = rgb_image.unsqueeze(1).expand(-1, repeat, -1, -1, -1)
    rgb_image = rgb_image.contiguous().view(-1, 3, width, width)
    x_mod = x_mod.view(-1, channels, width, width)
    step_lr = torch.from_numpy(np.array(step_lr)).float()

    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="reverse diffusion sampling"):
            timesteps = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            timesteps = timesteps.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            if coupled:
                error = torch.cat(rgb_image.shape[1] * [x_mod[:, 3:channels, :, :]], 1)
                corrupted_rgb_image = rgb_image + error * sigma * 0.1
            else:
                corrupted_rgb_image = rgb_image  # + torch.randn_like(rgb_image) * sigma

            x_mod[:, :3, :, :] = corrupted_rgb_image
            grad = scorenet(x_mod, timesteps)
            noise = torch.randn_like(grad) * torch.sqrt(step_size * 0.02) * sigma
            target = x_mod[:, 3:channels, :, :] + step_size * grad + noise
            x_mod[:, 3:channels, :, :] = torch.clamp(target, range[0], range[1])
            targets.append(x_mod[:, 3:channels, :, :].to('cpu'))

        target_pred = x_mod[:, 3:channels, :, :]
        target_pred = target_pred.contiguous().view(batch, repeat, channels - 3, width, width)
        target_pred = target_pred.mean(1)  # taking mean with repeated dimension

        return targets, target_pred


def reverse_diff_sampler(x_mod, rgb_image, scorenet, sigmas, step_lr=0.0001, range=[0., 1.]):
    """
    The function make prediction of target map for a given rbg image with reverse diffusion sampler

    :param x_mod: an initial random sample with the shape [repeat, batch, channel, width, height]
    :param rgb_image: the rgb image containing with shape [batch, channel, width, height]
    :param scorenet: the score network takes input of (rgb, target) image and return the score function
    :param sigmas: the noise levels of revese diffusion process
    :param n_steps_each: number of steps each noise level
    :param step_lr: learning rate for the smallest noise level
    :param range: the range of output data
    :return: list of sampled target maps with shapes [batch * repeat , out_channels, width, height]
             and converged sample average as prediction target map [batch , out_channels, width, height]
    """

    targets = []
    batch = x_mod.shape[1]
    repeat = x_mod.shape[0]
    width = x_mod.shape[3]
    channels = x_mod.shape[2]
    rgb_image = rgb_image.unsqueeze(1).expand(-1, repeat, -1, -1, -1)
    rgb_image = rgb_image.contiguous().view(-1, 3, width, width)
    x_mod = x_mod.view(-1, channels, width, width)

    step_lr = torch.from_numpy(np.array(step_lr)).float()
    rgb_images = []

    with torch.no_grad():
        diff_image = rgb_image
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="construct forward diffusion"):
            if c % 100 == 0:
                diff_image = diff_image.clone() * torch.sqrt(1 - sigma ** 2) + torch.randn_like(diff_image) * sigma
                rgb_images.append(diff_image)


        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="reverse diffusion sampling"):
            timesteps = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            timesteps = timesteps.long()

            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            corrupted_rgb_image = rgb_images[-(c//100)-1]
            # if c % 498 == 0:
            #     save_image(corrupted_rgb_image, 'diff_{}.png'.format(c))
            x_mod[:, :3, :, :] = corrupted_rgb_image
            grad = scorenet(x_mod, timesteps)
            noise = torch.randn_like(grad) * torch.sqrt(step_size * 0.02) * sigma
            # target = (1/torch.sqrt(1-sigma**2))*(x_mod[:, 3:channels, :, :] + step_size * grad) + noise
            # target = (1 - torch.sqrt(1-sigma**2)) *(x_mod[:, 3:channels, :, :]) + torch.sqrt(1-sigma**2) * grad
            target = grad
            x_mod[:, 3:channels, :, :] = torch.clamp(target, range[0], range[1])
            targets.append(x_mod[:, 3:channels, :, :].to('cpu'))

        target_pred = x_mod[:, 3:channels, :, :]
        target_pred = target_pred.contiguous().view(batch, repeat, channels - 3, width, width)
        target_pred = target_pred.mean(1)  # taking mean with repeated dimension

        return targets, target_pred



def diffusion_data(data, sigmas, timesteps=1000):
    """
    Create diffusion process from data, return the diffusion data
    -------------
    data: original data, expected tensor with shape [t, b, c, w, h]
    sigmas:
    timesteps: number of diffusion steps
    ------------
    return: diffused data from x_0 to x_T
    """
    current = data.clone().detach()
    for i in range(timesteps):
        current = current * torch.sqrt(1-sigmas[i]**2) + torch.randn_like(current) * sigmas[i]
        # current = torch.clamp(current, 0., 1.)
        data = torch.cat((current, data), 0)
    return data

def diffusion_timeseries_data(data, lookback=5, noise_level=1):
    """
    Create diffusion process from data, return the reverse chain data
    -------------
    data: original data, expected tensor with shape [t, b, c, w, h]
    lookback: number of lookback steps
    noise_leve: diffusion noise level
    ------------
    return: reversed chain data with lookback steps
    X: features with lookback steps
    Y: targets
    """

    diffuse = data + torch.randn_like(data) * noise_level
    origin = data
    for i in range(6):
        diffuse = origin + torch.randn_like(diffuse) * (np.exp(i - 3))
        diffuse = torch.clamp(diffuse, 0., 1.)
        data = torch.cat((diffuse, data), 0)
    # sample = make_grid(data.squeeze(1), nrow=10)
    # save_image(sample, os.path.join(self.args.image_folder, 'diff.png'))
    x = []
    y = []
    for index in range(data.size()[0] - lookback - 1):
        x.append(data[index: index + lookback, ...])
        y.append(data[index + lookback: index + lookback + 1, ...])
    x = torch.cat(x, 1)
    y = torch.cat(y, 1)
    return x.permute(1, 0, 2, 3, 4), y.permute(1, 0, 2, 3, 4)


def get_ode_sampler(sde, shape, inverse_scaler, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver.
    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      cond: Conditional variable
      inverse_scaler: The inverse data normalizer.
      rtol: A `float` number. The relative tolerance level of the ODE solver.
      atol: A `float` number. The absolute tolerance level of the ODE solver.
      method: A `str`. The algorithm used for the black-box ODE solver.
        See the documentation of `scipy.integrate.solve_ivp`.
      eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def drift_fn(model, cond, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, cond, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, cond, z=None):
        """The probability flow ODE sampler with black-box ODE solver.
        Args:
          model: A score model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, cond, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
