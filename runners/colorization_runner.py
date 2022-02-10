import numpy as np
import tqdm
from losses.dsm import *
import torchvision.transforms.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.celeba import CelebAColorization
from models.densedepth import CondDenseColor, CondDenseColorPlus
from torchvision.utils import save_image, make_grid
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import warnings
warnings.filterwarnings("ignore")


__all__ = ['ColorizationRunner']


class ColorizationRunner():
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

    def convert_to_rgb(self, gray_image, ab_image):
        if gray_image.shape[1] != 1:
            gray_image = gray_image[:, :1, :, :]
        gray_image = (gray_image + 1.) * 50.
        ab_image = ab_image * 110.

        img_lab = torch.cat((gray_image, ab_image), 1).permute(0, 2, 3, 1).cpu().numpy()
        rgb_imgs = []
        for img in img_lab:
            img_rgb = lab2rgb(img)
            rgb_imgs.append(img_rgb)
        rgb_imgs = np.stack(rgb_imgs, axis=0)
        return torch.from_numpy(rgb_imgs).permute(0, 3, 1, 2)

    def reverse_sampler(self, initial, gray_image, scorenet, sigmas, step_lr=.05, steps=2000, corrupted=True,
                        nrange=[-.8, .8]):

        targets = []
        batch = initial.shape[1]
        repeat = initial.shape[0]
        width = initial.shape[3]
        height = initial.shape[4]
        channels = initial.shape[2]
        gray_image = gray_image.unsqueeze(1).expand(-1, repeat, -1, -1, -1)
        gray_image = gray_image.contiguous().view(-1, 3, width, height)
        initial = initial.view(-1, channels, width, height)

        step_lr = torch.from_numpy(np.array(step_lr)).float()

        with torch.no_grad():
            offset = sigmas[0]
            timesteps = torch.ones(initial.shape[0], device=initial.device) * (len(sigmas) - 1)
            timesteps = timesteps.long()
            sigmas = sigmas[1:]
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="reverse diffusion sampling"):
                if corrupted:
                    corrupted_image = gray_image + torch.randn_like(gray_image) * sigma * 0.1
                else:
                    corrupted_image = gray_image
                
                grad = scorenet(corrupted_image, initial, timesteps)
                target = initial + (offset-sigma) * grad 
                initial = torch.clamp(target, nrange[0], nrange[1])
                offset = sigma

                for _ in range(steps):
                    timesteps = torch.ones(initial.shape[0], device=initial.device) * (len(sigmas) - c - 1)
                    timesteps = timesteps.long()
                    step_size = step_lr * sigma**2 #* (torch.rand_like(step_lr) + 0.5) 

                    grad = scorenet(corrupted_image, initial, timesteps)
                    noise = torch.randn_like(grad) * torch.sqrt(step_size * 0.01) * sigma
                    target = initial + step_size * grad #+ noise
                    initial = torch.clamp(target, nrange[0], nrange[1])
                    del grad
                    del target
                    targets.append(initial.to('cpu'))
                
                steps = int(steps*0.9)

        target_pred = initial
        target_pred = target_pred.contiguous().view(batch, repeat, channels, width, height)
        target_pred = target_pred.mean(1)  # taking mean with repeated dimension

        return targets, target_pred


    def reverse_EM(self, initial, gray_image, scorenet, sigmas, nsample=5,
                    step_lr1=20.0, step_lr2=1.00, steps=200, nrange=[-1., 1.]):
        targets = []
        batch = initial.shape[1]
        repeat = initial.shape[0]
        width = initial.shape[3]
        height = initial.shape[4]
        channels = initial.shape[2]
        gray_image = gray_image.unsqueeze(1).expand(-1, repeat, -1, -1, -1)
        gray_image = gray_image.contiguous().view(-1, 3, width, height)
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

                    corrupted_image = torch.ones_like(gray_image) * 0.0
                    noninformative = torch.ones_like(gray_image) * 0.0
                    for _ in range(nsample):                    
                        corrupted_image +=  gray_image + torch.randn_like(gray_image) * sigma 
                        noninformative  +=  gray_image + torch.randn_like(gray_image) * (sigma+0.1) 


                    corrupted_rgb_image = corrupted_image / nsample
                    noninformative = noninformative / nsample

                    grad1 = scorenet(corrupted_image, initial, timesteps)
                    grad2 = scorenet(noninformative, initial, prev_timesteps)
                    target = initial + step_size1 * grad1 - step_size2 * grad2
                    initial = torch.clamp(target, nrange[0], nrange[1])
                    del grad1
                    del grad2
                    del target
                    targets.append(initial.to('cpu'))
                steps = int(steps*0.75)

        target_pred = initial
        target_pred = target_pred.contiguous().view(batch, repeat, channels, width, height)
        target_pred = target_pred.mean(1)  # taking mean with repeated dimension

        return targets, target_pred

    def train(self):
        target_range = 255.
        depot_root = '/data'        ## Change to your data path
        if self.config.data.random_flip:
            dataset = CelebAColorization(root=depot_root, split='train',
                                         transforms=transforms.Compose([
                                             transforms.CenterCrop(140),
                                             transforms.Resize(
                                                 (self.config.data.image_size, self.config.data.image_size),
                                                 Image.BICUBIC),
                                             transforms.RandomHorizontalFlip(),
                                         ]))
        else:
            dataset = CelebAColorization(root=depot_root, split='train',
                                         transforms=transforms.Compose([
                                             transforms.CenterCrop(140),
                                             transforms.Resize(
                                                 (self.config.data.image_size, self.config.data.image_size),
                                                 Image.BICUBIC),
                                         ]))

        test_dataset = CelebAColorization(root=depot_root, split='test',
                                          transforms=transforms.Compose([
                                              transforms.CenterCrop(140),
                                              transforms.Resize(
                                                  (self.config.data.image_size, self.config.data.image_size),
                                                  Image.BICUBIC),
                                          ]))

        train_loader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                  num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)
        mse = torch.nn.MSELoss()
        test_iter = iter(test_loader)

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        model = CondDenseColorPlus(self.diff_steps, self.config.model.pre_train).to(self.config.device)
        model = torch.nn.DataParallel(model)

        optimizer = self.get_optimizer(model.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            model.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0
        sigmas = torch.tensor(np.linspace(self.config.model.sigma_min, self.config.model.sigma_max,
                               self.diff_steps)).float().to(self.config.device)

        for epoch in range(self.config.training.n_epochs):
            for idx, batch in enumerate(train_loader):
                step += 1
                model.train()

                X = torch.Tensor(batch["L"]).to(self.config.device)
                X = X.repeat(1, 3, 1, 1)
                y = torch.Tensor(batch["ab"])
                y = y.to(self.config.device)
                # print(torch.min(X), torch.max(X))
                # print(torch.min(y), torch.max(y))

                timesteps = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)

                loss = colorization_dsm_score(model, X, y, timesteps, sigmas)  # for joint diffusion
                # loss = anneal_marginal_dsm_score(model, X, y, timesteps, sigmas)
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
        batch_size = 10
        repeat = 1
        imgs = []
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        model = CondDenseColorPlus(self.diff_steps, self.config.model.pre_train).to(self.config.device)

        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        model.eval()
        depot_root = '/data'        ## Change to your data path

        test_dataset = CelebAColorization(root=depot_root, split='train',
                                          transforms=transforms.Compose([
                                              transforms.CenterCrop(140),
                                              transforms.Resize(
                                                  (self.config.data.image_size, self.config.data.image_size),
                                                  Image.BICUBIC),
                                          ]))
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, drop_last=True)

        data_iter = iter(dataloader)
        batch = next(data_iter)
        gray_image, target = torch.Tensor(batch['L']), torch.Tensor(batch['ab'])
        gray_image = gray_image.repeat(1, 3, 1, 1)
        
        gray_image = gray_image.to(self.config.device)
        target = target.to(self.config.device)

        # MSE loss evaluation
        mse = torch.nn.MSELoss()

        samples = torch.rand(repeat, batch_size, target.shape[1], target.shape[2],
                             target.shape[3]).to(self.config.device) * 2 - 0.5

        sigmas = torch.tensor(np.linspace(self.config.model.sigma_max, self.config.model.sigma_min,
                               self.diff_steps)).float().to(self.config.device)

        all_samples, target_pred = self.reverse_sampler(samples, gray_image, model, sigmas,
                                    corrupted=True, nrange=[-0.8, 0.8]) 
        # all_samples, target_pred = self.reverse_EM(samples, gray_image, model, sigmas, nrange=[-0.8, 0.8])


        sample_loss = []
        sample_num = []
        for i, sample in enumerate(tqdm.tqdm(all_samples)):
            sample = sample.view(batch_size, repeat, target.shape[1], target.shape[2],
                                 target.shape[3])
            sample = self.convert_to_rgb(gray_image.cpu(), sample.mean(1).squeeze(1))
            target_grid = make_grid(sample, nrow=repeat)

            if i % 50 == 0:
                imgs.append(F.to_pil_image(target_grid))
                save_image(target_grid, os.path.join(self.args.image_folder, 'target_prediction_{}.png'.format(i)))


        rgb_image_gen = torch.cat((target, target_pred), 0)
        rgb_image_gen = self.convert_to_rgb(torch.cat((gray_image, gray_image), 0), rgb_image_gen)
        org_image = self.convert_to_rgb(gray_image, target)

        mse_loss = mse(target_pred, target)
        print("MSE loss is %5.4f" % (mse_loss))
        
        gray_image = make_grid(gray_image, nrow=1)
        org_image = make_grid(org_image, nrow=1)
        rgb_image_gen = make_grid(rgb_image_gen, nrow=batch_size)
        save_image(org_image, os.path.join(self.args.image_folder, 'org_image.png'))
        save_image(gray_image, os.path.join(self.args.image_folder, 'grayscale_image.png'))
        save_image(rgb_image_gen, os.path.join(self.args.image_folder, 'color_image.png'))

        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:],
                     duration=1, loop=0)
