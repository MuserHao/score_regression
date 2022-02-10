import numpy as np
import cv2
import tqdm
import torchvision.transforms.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from losses.depthloss import ssim as ssim_criterion
from losses.depthloss import depth_loss as gradient_criterion
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from models.densedepth import CondDenseDepth
from datasets.nyuv2 import getTrainingTestingData
from losses.dsm import *
import random

import matplotlib.pyplot as plt

__all__ = ['DiffusionRunner']


class DiffusionRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.diff_steps = self.config.model.num_scales

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

    def reverse_sampler(self, initial, rgb_image, scorenet, sigmas, step_lr=1.5, steps=50, nrange=[10., 1000.], corrupted=True):

        targets = []
        batch = initial.shape[1]
        repeat = initial.shape[0]
        width = initial.shape[3]
        height = initial.shape[4]
        channels = initial.shape[2]
        rgb_image = rgb_image.unsqueeze(1).expand(-1, repeat, -1, -1, -1)
        rgb_image = rgb_image.contiguous().view(-1, 3, width * 2, height * 2)
        initial = initial.view(-1, channels, width, height)

        step_lr = torch.from_numpy(np.array(step_lr)).float()

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="reverse diffusion sampling"):
                for _ in range(steps):
                    timesteps = torch.ones(initial.shape[0], device=initial.device) * c
                    timesteps = timesteps.long()
                    step_size = step_lr * (sigma / sigmas[-1])
                    if corrupted:
                        corrupted_rgb_image = rgb_image + torch.randn_like(rgb_image) * sigma * 0.5
                    else:
                        corrupted_rgb_image = rgb_image

                    grad = scorenet(corrupted_rgb_image, initial, timesteps)
                    noise = torch.randn_like(grad) * torch.sqrt(step_size * 0.1) * sigma
                    noise = noise * grad
                    target = initial + step_size * grad #+ noise
                    initial = torch.clamp(target, nrange[0], nrange[1])
                    del grad
                    del target
                    targets.append(initial.to('cpu'))

        target_pred = initial
        target_pred = target_pred.contiguous().view(batch, repeat, channels, width, height)
        target_pred = target_pred.mean(1)  # taking mean with repeated dimension

        return targets, target_pred


    def reverse_EM(self, initial, rgb_image, scorenet, sigmas, nsample=5,
                    step_lr1=40.0, step_lr2=1.0, steps=100, nrange=[10., 1000.]):
        targets = []
        batch = initial.shape[1]
        repeat = initial.shape[0]
        width = initial.shape[3]
        height = initial.shape[4]
        channels = initial.shape[2]
        rgb_image = rgb_image.unsqueeze(1).expand(-1, repeat, -1, -1, -1)
        rgb_image = rgb_image.contiguous().view(-1, 3, width * 2, height * 2)
        initial = initial.view(-1, channels, width, height)

        step_lr1 = torch.from_numpy(np.array(step_lr1)).float()
        step_lr2 = torch.from_numpy(np.array(step_lr2)).float()

        with torch.no_grad():
            offset = sigmas[0]
            timesteps = torch.ones(initial.shape[0], device=initial.device) * (len(sigmas) - 1)
            timesteps = timesteps.long()
            sigmas = sigmas[1:]
            sigmas = sigmas.flip(0)
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="reverse EM prediction"):
                for _ in range(steps):
                    prev_timesteps = timesteps
                    timesteps = torch.ones(initial.shape[0], device=initial.device) * len(sigmas-c-1)
                    timesteps = timesteps.long()
                    step_size1 = step_lr1 * sigma**2 / nsample
                    step_size2 = step_lr2 * sigma**2 / nsample

                    corrupted_rgb_image = torch.ones_like(rgb_image) * 0.0
                    noninformative = torch.ones_like(rgb_image) * 0.0
                    for _ in range(nsample):                    
                        corrupted_rgb_image += rgb_image + torch.randn_like(rgb_image) * sigma 
                        noninformative  +=  rgb_image + torch.randn_like(rgb_image) * (sigma+0.1) 
                        # noninformative  +=  torch.randn_like(rgb_image) * (sigma) + rgb_image.mean()


                    corrupted_rgb_image = corrupted_rgb_image / nsample
                    noninformative = noninformative / nsample

                    grad1 = scorenet(corrupted_rgb_image, initial, timesteps)
                    grad2 = scorenet(noninformative, initial, prev_timesteps)
                    target = initial + step_size1 * grad1 - step_size2 * grad2
                    initial = torch.clamp(target, nrange[0], nrange[1])
                    del grad1
                    del grad2
                    del target
                    targets.append(initial.to('cpu'))
                steps = int(steps*0.66)

        target_pred = initial
        target_pred = target_pred.contiguous().view(batch, repeat, channels, width, height)
        target_pred = target_pred.mean(1)  # taking mean with repeated dimension

        return targets, target_pred

    def train(self):
        depot_root = '/depot/yuzhu/data'
        target_range = 6.5144

        train_loader, test_loader = getTrainingTestingData(os.path.join(depot_root, 'NYUv2', 'nyu_data.zip'),
                                                           batch_size=self.config.training.batch_size)
        num_trainloader = len(train_loader)
        num_testloader = len(test_loader)

        mse = torch.nn.MSELoss()

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        model = CondDenseDepth(self.diff_steps, self.config.model.pre_train).to(self.config.device)
        model = torch.nn.DataParallel(model)

        optimizer = self.get_optimizer(model.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            model.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0
        sigmas = torch.exp(torch.from_numpy(
            np.linspace(np.log(self.config.model.sigma_min), np.log(self.config.model.sigma_max),
                        self.diff_steps))).to(self.config.device)

        for epoch in range(self.config.training.n_epochs):
            for idx, batch in enumerate(train_loader):
                step += 1
                model.train()

                X = torch.Tensor(batch["image"]).to(self.config.device)
                y = torch.Tensor(batch["depth"])
                # y = DepthNorm(y)  # + torch.rand_like(y) / target_range
                y = y.to(self.config.device)

                timesteps = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)

                loss = anneal_joint_dsm_score(model, X, y, timesteps, sigmas)    # for joint diffusion
                # loss = anneal_marginal_dsm_score(model, X, y, timesteps, sigmas)  # for marginal diffusion
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                    ]
                    # torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    def test(self):
        batch_size = 2
        repeat = 1
        imgs = []
        random.seed(9)
        # mask = [0, 0, 0, 0, 0]
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        model = CondDenseDepth(self.diff_steps, self.config.model.pre_train).to(self.config.device)

        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        model.eval()
        depot_root = '/depot/yuzhu/data'
        target_range = 6.5144

        dataloader, _ = getTrainingTestingData(os.path.join(depot_root, 'NYUv2', 'nyu_data.zip'), batch_size=batch_size)

        data_iter = iter(dataloader)
        batch = next(data_iter)
        rgb_image, target = torch.Tensor(batch['image']), torch.Tensor(batch['depth'])
        rgb_image = rgb_image.to(self.config.device)
        target = target.to(self.config.device)

        rgb_image = rgb_image + torch.randn_like(rgb_image) * np.sqrt(0.80)

        # MSE loss evaluation
        mse = torch.nn.MSELoss()

        samples = torch.rand(repeat, batch_size, target.shape[1], target.shape[2],
                             target.shape[3]).to(self.config.device) * 30. + 500.

        sigmas = torch.exp(torch.from_numpy(
            np.linspace(np.log(self.config.model.sigma_min), np.log(self.config.model.sigma_max),
                        self.diff_steps))).to(self.config.device)

        # all_samples, target_pred = self.reverse_sampler(samples, rgb_image, model, sigmas, corrupted=False)
        all_samples, target_pred = self.reverse_EM(samples, rgb_image, model, sigmas)

        sample_loss = []
        sample_num = []
        for i, sample in enumerate(tqdm.tqdm(all_samples)):
            sample = sample.view(batch_size, repeat, target.shape[1], target.shape[2],
                                 target.shape[3])
            if i >= 0:
                sample_loss.append(
                    mse(sample.mean(1).squeeze(1).to(self.config.device), target).item())
                sample_num.append(i)
            sample = sample.view(batch_size * repeat, target.shape[1], target.shape[2],
                                 target.shape[3])
            target_grid = make_grid(sample, nrow=repeat)
            dep = target_grid[0, :, :].unsqueeze(0)
            dep = F.to_pil_image(dep)

            if i % 20 == 0:
                imgs.append(dep)
                target_grid = target_grid.mean(0).data.squeeze().numpy().astype(np.float32)
                plt.imsave(os.path.join(self.args.image_folder, 'target_prediction_{}.png'.format(i)),
                   target_grid, cmap="jet")
                # save_image(target_grid, os.path.join(self.args.image_folder, 'target_prediction_{}.png'.format(i)))

        rel_loss = ((target_pred - target) / target).abs().mean()
        target = make_grid(target, nrow=1)
        target_pred = make_grid(target_pred, nrow=1)
        target = target[:1,:,:].cpu()
        target_pred = target_pred[:1,:,:].cpu()

        # one final deblurring step
        # target_pred = target_pred.permute(1, 2, 0).cpu().numpy()
        # target_pred = cv2.GaussianBlur(target_pred, (7,7), 1)
        # # target_pred = cv2.bilateralFilter(target_pred, 9, 75, 75)
        # target_pred = torch.from_numpy(target_pred).unsqueeze(0)

        mse_loss = mse(target_pred, target) /  10000
        print("MSE loss is %5.4f" % (mse_loss))
        print("REL loss is %5.4f" % (rel_loss))


        torch.save(target_pred, os.path.join(self.args.image_folder, 'target_pred.pth'))

        true_pred_target = torch.cat((target, target_pred), 2)

        image_grid = make_grid(rgb_image, nrow=1)
        save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))

        true_pred_target = true_pred_target.mean(0).data.squeeze().numpy().astype(np.float32)
        plt.imsave(os.path.join(self.args.image_folder, 'target_true_pred_with_loss_{}.png'.format(mse_loss)),
                   true_pred_target, cmap="jet")

        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:],
                     duration=1, loop=0)
