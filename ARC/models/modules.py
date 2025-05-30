import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from scipy.special import iv

class Scheduler:
    def __call__(self, **kwargs):
        raise NotImplemented()


class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (iteration - self.start_iteration) * self.m + self.start_value


class ExponentialScheduler(LinearScheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0, base=10):
        self.base = base

        super(ExponentialScheduler, self).__init__(start_value=math.log(start_value, base),
                                                   end_value=math.log(end_value, base),
                                                   n_iterations=n_iterations,
                                                   start_iteration=start_iteration)

    def __call__(self, iteration):
        linear_value = super(ExponentialScheduler, self).__call__(iteration)
        return self.base ** linear_value


class Encoder(nn.Module):
    def __init__(self, z_dim, in_channels,hidden_size):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        params = self.net(x)
        return params


class Decoder(nn.Module):
    def __init__(self, z_dim,encoder_dim,hidden_size):
        super(Decoder, self).__init__()
        self._decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU()
        )

    def decoder(self, latent):
        x_hat = self._decoder(latent)
        return x_hat



eps = 1e-7


def entropy(
    embeddings: torch.Tensor,
    kappa: float = 10,
    support: str = "sphere",
    reduction: str = "expectation",
) -> torch.Tensor:

    k = embeddings.shape[0]
    d = embeddings.shape[1]

    if support == "sphere":
        csim = kappa * torch.matmul(embeddings, embeddings.T)
        const = (
            -math.log(kappa) * (d * 0.5 - 1)
            + 0.5 * d * math.log(2 * math.pi)
            + math.log(iv(0.5 * d - 1, kappa) + 1e-7)
            + math.log(k)
        )
        if reduction == "average":
            entropy = -torch.logsumexp(csim, dim=-1) + const  # -> log(p).sum
            entropy = entropy.mean()
        elif reduction == "expectation":
            logp = -torch.logsumexp(csim, dim=-1) + const
            entropy = F.softmax(-logp, dim=-1) * logp
            entropy = entropy.sum()
        else:
            raise NotImplementedError(f"Reduction type {reduction} not implemented")
    elif support == "discrete":
        embeddings_mean = embeddings.mean(0)
        if reduction == "expectation":
            entropy = -(embeddings_mean * torch.log(embeddings_mean + eps)).sum()
        elif reduction == "average":
            entropy = -torch.log(embeddings_mean + eps).mean()
        else:
            raise NotImplementedError(f"Reduction type {reduction} not implemented")
    else:
        raise NotImplementedError(f"Support type {support} not implemented")

    return entropy


def reconstruction(
    projection1: torch.Tensor,
    projection2: torch.Tensor,
    kappa: float = 10,
    support: str = "sphere",
) -> torch.Tensor:

    d = projection1.shape[1]

    if support == "sphere":
        const = (
            -(0.5 * d - 1) * math.log(kappa)
            + 0.5 * d * math.log(2 * math.pi)
            + math.log(iv(0.5 * d - 1, kappa) + 1e-7)
        )
        csim = kappa * torch.sum(projection1 * projection2, dim=-1)
        rec = -csim + const

    elif support == "discrete":
        rec1 = -torch.sum(projection1 * torch.log(projection2 + eps), dim=-1)
        rec2 = -torch.sum(projection2 * torch.log(projection1 + eps), dim=-1)
        rec = 0.5 * (rec1 + rec2)

    rec_mean = rec.mean()
    return rec_mean



def calculate_pairwise_similarity(z_1, z_2, z_3, method='mmd'):
    if method == 'js':
        sim_1_2 = js_divergence(z_1, z_2)
        sim_1_3 = js_divergence(z_1, z_3)
        sim_2_3 = js_divergence(z_2, z_3)
    else:
        sim_1_2 = mmd_loss(z_1, z_2)
        sim_1_3 = mmd_loss(z_1, z_3)
        sim_2_3 = mmd_loss(z_2, z_3)

    return sim_1_2, sim_1_3, sim_2_3


def js_divergence(p_output, q_output):
    p_output = F.softmax(p_output, dim=1)
    q_output = F.softmax(q_output, dim=1)

    m = 0.5 * (p_output + q_output)

    kl_p_m = F.kl_div(m.log(), p_output, reduction='batchmean')
    kl_q_m = F.kl_div(m.log(), q_output, reduction='batchmean')

    js = 0.5 * (kl_p_m + kl_q_m)

    return np.exp(1 - js.item()) - 1


def mmd_loss(x, y, kernel_mul=2.0, kernel_num=5):
    n_samples = int(x.size()[0])
    total = torch.cat([x, y], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0 - total1) ** 2).sum(2)

    bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    kernels = sum(kernel_val)

    XX = kernels[:n_samples, :n_samples]
    YY = kernels[n_samples:, n_samples:]
    XY = kernels[:n_samples, n_samples:]
    YX = kernels[n_samples:, :n_samples]

    loss = torch.mean(XX + YY - XY - YX)

    return np.exp(-loss.item())



class Loss(nn.Module):
    def __init__(self, batch_size, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature_InfoNCE(self, h_i, h_j, batch_size,weights=None):
        self.batch_size = batch_size

        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)


        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)

        loss /= N
        return loss

    def forward_feature_PSCL(self, z1, z2, r=3.0):
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1 - mask1) * F.normalize(z1, dim=1) * np.sqrt(r)
        z2 = mask2 * z2 + (1 - mask2) * F.normalize(z2, dim=1) * np.sqrt(r)
        loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
        square_term = torch.matmul(z1, z2.T) ** 2
        loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                     z1.shape[0] / (z1.shape[0] - 1)

        return loss_part1 + loss_part2
