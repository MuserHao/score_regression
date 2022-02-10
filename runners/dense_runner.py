import numpy as np
import pandas as pd
import tqdm
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from losses.depthloss import ssim as ssim_criterion
from losses.depthloss import depth_loss as gradient_criterion
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image, make_grid
from models.densedepth import DenseDepth
from datasets.nyuv2 import getTrainingTestingData

from .utils import (
    AverageMeter,
    DepthNorm,
    colorize,
    load_from_checkpoint,
    init_or_load_model,
)

import matplotlib.pyplot as plt

__all__ = ['DenseRunner']


class DenseRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.diffusion = True

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def train(self):
        depot_root = '/depot/yuzhu/data'

        nyu_train_transform = transforms.Compose([
            transforms.CenterCrop((400, 400)),
            transforms.Resize(self.config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        nyu_test_transform = transforms.Compose([
            transforms.CenterCrop((400, 400)),
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        nyu_depth_train = transforms.Compose([
            transforms.CenterCrop((400, 400)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])
        nyu_depth_test = transforms.Compose([
            transforms.CenterCrop((400, 400)),
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        train_loader, test_loader = getTrainingTestingData(os.path.join(depot_root, 'NYUv2', 'nyu_data.zip'),
                                                         batch_size=self.config.training.batch_size)

        num_trainloader = len(train_loader)
        num_testloader = len(test_loader)

        l1_criterion = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        model = DenseDepth(self.config.model.pre_train)

        model = torch.nn.DataParallel(model)

        optimizer = self.get_optimizer(model.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            model.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0
        training_mses = []
        testing_mses = []

        for epoch in range(self.config.training.n_epochs):
            for idx, batch in enumerate(train_loader):
                step += 1
                model.train()

                X = torch.Tensor(batch["image"])
                y = torch.Tensor(batch["depth"])

                if self.diffusion:
                    sigma = torch.rand(1)
                    y = y + torch.randn_like(y) * sigma * 31.62
                    X = X + torch.randn_like(X) * sigma
                X = X.to(self.config.device)
                y = y.to(self.config.device)

                y_pred = model(X)
                # calculating the losses
                l1_loss = l1_criterion(y_pred, y)

                ssim_loss = torch.clamp(
                    (1 - ssim_criterion(y_pred, y, 1000.0 / 10.0)) * 0.5,
                    min=0,
                    max=1,
                )

                gradient_loss = gradient_criterion(y, y_pred, device=self.config.device)

                loss = (
                        (1.0 * ssim_loss)
                        + (1.0 * torch.mean(gradient_loss))
                        + (0.1 * torch.mean(l1_loss))
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 10000 == 0:
                    model.eval()
                    try:
                        test_batch = next(test_iter)
                        test_X = torch.Tensor(test_batch["image"])
                        test_y = torch.Tensor(test_batch["depth"])
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                    target_pred = model(X)
                    # print(torch.min(y), torch.max(y), torch.min(test_y), torch.max(test_y))
                    # print(num_trainloader)
                    train_mse_loss = mse(target_pred, y).item()
                    del X
                    del y
                    del target_pred
                    target_pred = model(test_X)
                    test_mse_loss = mse(target_pred, test_y).item()
                    print("Training MSE loss is: %5.4f" % (train_mse_loss))
                    print("Testing MSE loss is: %5.4f" % (test_mse_loss))
                    training_mses.append(train_mse_loss)
                    testing_mses.append(test_mse_loss)
                    del test_X
                    del test_y
                    del target_pred


                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                    ]
                    # torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))


    def test(self):
        batch_size = 10
        sigma = 1.0
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        model = DenseDepth(self.config.model.pre_train)
    
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0])
    
        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)
    
        model.eval()
        depot_root = '/depot/yuzhu/data'
    
        _, dataloader = getTrainingTestingData(os.path.join(depot_root, 'NYUv2', 'nyu_data.zip'), batch_size=batch_size)
    
        data_iter = iter(dataloader)
        batch = next(data_iter)
        batch = next(data_iter)

        rgb_image, target = torch.Tensor(batch['image']), torch.Tensor(batch['depth'])


        rgb_image = rgb_image.to(self.config.device)
        target = target.to(self.config.device)

    
        # MSE loss evaluation
        mse = torch.nn.MSELoss()

        losses_g = []
        losses_u = []
        losses_e = []
        sigmas = np.linspace(0, sigma, 50)
        with torch.no_grad():
            for sig in tqdm.tqdm(sigmas):
                nimg = rgb_image + torch.randn_like(rgb_image) * sig
                target_pred = model(nimg)
                mse_loss = mse(target_pred, target).item()
                losses_g.append(mse_loss/10000)

                nimg = rgb_image + torch.rand_like(rgb_image) * sig * np.sqrt(12)
                target_pred = model(nimg)
                mse_loss = mse(target_pred, target).item()
                losses_u.append(mse_loss/10000)

                param = torch.ones_like(rgb_image) * (1 / (sig + 0.01))
                dist = torch.distributions.Exponential(param)
                nimg = rgb_image + dist.sample()
                target_pred = model(nimg)
                mse_loss = mse(target_pred, target).item()
                losses_e.append(mse_loss/10000)


        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(sigmas, losses_g, 'blue', label='noise type = Gaussian')
        ax.plot(sigmas, losses_u, 'green', label='noise type = Uniform')
        ax.plot(sigmas, losses_e, 'purple', label='noise type = Exponential')
        plt.xlabel('Diffusion sigmas')
        plt.ylabel('Mean squared error')
        plt.title('Test loss curve of diffusion regression')
        ax.legend()
        fig.savefig(os.path.join(self.args.image_folder, 'loss_curve.png'))

        df = pd.DataFrame({'sigmas': sigmas, 'Gaussian': losses_g, 'Uniform': losses_u, 'Exponential': losses_e})
        df.to_csv(os.path.join(self.args.image_folder, 'loss_curve.csv'))


        
        image_grid = make_grid(rgb_image, nrow=1)
        save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion.png'))
        
        with torch.no_grad():
            target_pred = model(rgb_image)

            mse_loss = mse(target_pred, target).item() / 10000
            rel_loss = ((target_pred - target) / target).abs().mean()
            print("MSE loss is %5.4f" % (mse_loss))
            print("REL loss is %5.4f" % (rel_loss))
            target = make_grid(target, nrow=1)
            torch.save(target, os.path.join(self.args.image_folder, 'target_{}.pth').format(sigma))
            target_pred = make_grid(target_pred, nrow=1)
            torch.save(target_pred, os.path.join(self.args.image_folder, 'target_end_pred_{}.pth').format(sigma))
            true_pred_target = torch.cat((target, target_pred), 2)
            true_pred_target = true_pred_target.mean(0).data.squeeze().cpu().numpy().astype(np.float32)
            plt.imsave(os.path.join(self.args.image_folder, 'target_true_pred_with_loss_{}.png'.format(mse_loss)),
                       true_pred_target, cmap="jet")
