import torch
import torch.autograd as autograd
from .depthloss import ssim


mse = torch.nn.MSELoss()

def dsm(energy_net, samples, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


def dsm_score_estimation(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss


def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels.type(torch.LongTensor)].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)


def anneal_cond_dsm_score(scorenet, x_samples, y_samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels.type(torch.LongTensor)].view(y_samples.shape[0], *([1] * len(y_samples.shape[1:])))
    perturbed_samples = y_samples + torch.randn_like(y_samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - y_samples)
    joint = torch.cat((x_samples, perturbed_samples), 1)
    scores = scorenet(joint, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)

def anneal_marginal_dsm_score(scorenet, x_samples, y_samples, labels, sigmas, yscale=31.62, anneal_power=2.):
    used_sigmas = sigmas[labels.type(torch.LongTensor)].view(y_samples.shape[0], *([1] * len(y_samples.shape[1:])))
    perturbed_samples = y_samples + torch.randn_like(y_samples) * used_sigmas * yscale
    target = y_samples - perturbed_samples
    x_samples = x_samples.float()
    perturbed_samples = perturbed_samples.float()
    scores = scorenet(x_samples, perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    # loss += 1 / 2. * torch.abs(scores - target).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0)

def anneal_joint_dsm_score(scorenet, x_samples, y_samples, labels, sigmas, xscale=1.0, yscale=31.62, anneal_power=2.):
    used_sigmas = sigmas[labels.type(torch.LongTensor)].view(y_samples.shape[0], *([1] * len(y_samples.shape[1:])))
    perturbed_samples = y_samples + torch.randn_like(y_samples) * used_sigmas * yscale
    x_samples = x_samples + torch.randn_like(x_samples) * used_sigmas * xscale
    # target = - 1 / (used_sigmas ** 2) * (perturbed_samples - y_samples)
    target = y_samples - perturbed_samples
    x_samples = x_samples.float()
    perturbed_samples = perturbed_samples.float()
    scores = scorenet(x_samples, perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    # loss += 1 / 2. * torch.abs(scores - target).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0)

def colorization_dsm_score(scorenet, x_samples, y_samples, labels, sigmas, xscale=1.0, yscale=1.0, anneal_power=2.):
    used_sigmas = sigmas[labels.type(torch.LongTensor)].view(y_samples.shape[0], *([1] * len(y_samples.shape[1:])))
    perturbed_samples = y_samples + torch.randn_like(y_samples) * used_sigmas * yscale
    x_samples = x_samples + torch.randn_like(x_samples) * used_sigmas * xscale
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - y_samples)
    # target = y_samples - perturbed_samples
    x_samples = x_samples.float()
    perturbed_samples = perturbed_samples.float()
    scores = scorenet(x_samples, perturbed_samples, labels)
    # loss = 1000. * (1 - ssim(target, scores, 1.0)) * used_sigmas.squeeze() ** anneal_power
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    # loss += 1 / 2. * torch.abs(scores - target).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0)

def anneal_cond_mse(scorenet, x_samples, y_samples, labels):
    scores = scorenet(x_samples, labels)
    loss = mse(scores, y_samples)
    return loss.mean(dim=0)

def anneal_mse(scorenet, diff_x, diff_y, y_samples, timesteps, sigma):
    est = scorenet(torch.cat((diff_x, diff_y), 1), timesteps)
    loss = 1 / (2. * sigma**2) * mse(est, y_samples)
    return loss.mean(dim=0)
