import torch
from tqdm import tqdm
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from gumbel_sinkhorn_topk import gumbel_sinkhorn_topk, soft_topk


def sample_gumbel(size, eps=1e-20):
    """
    randomly sample standard gumbel variables
    """
    u = torch.rand(size, device=torch.device('cuda:0'))
    return -torch.log(-torch.log(u + eps) + eps)

if False:
    result_dict = torch.load('gumbel_gap_to_discrete.pt')
    fig = plt.figure(figsize=(6, 5), dpi=120)
    sigma_list = [0.2 / 2 ** n for n in range(0, 11)]
    for (tau, gap), style in zip(result_dict.items(), ['r.-', 'g.-', 'b.-', 'm.-', 'y.-']):
        plt.plot(sigma_list, [_.cpu()*2 for _ in gap], style, label=r'GS-TopK ($\tau=' + f'{tau:.3f}' + r'$)')
    plt.plot(sigma_list, [2 for _ in gap], 'kx-', label=r'SOFT-TopK (for all $\tau$)')
    plt.legend()
    plt.xlabel(r'$\sigma$ (log scale)')
    plt.ylabel(r'Gap to Discrete Solution')
    plt.xscale('log')
    fig.savefig('gumbel_discrete_gap_plot.pdf', bbox_inches='tight')
    sys.exit(0)

ori_list = torch.tensor([5, 4, 3., 3, 2, 1], dtype=torch.float, device=torch.device('cuda:0'))
ori_list /= ori_list.max()

ori_indices = torch.sort(ori_list, descending=True).indices[:3]
#ori_results = torch.stack((torch.zeros_like(ori_list), torch.ones_like(ori_list)), dim=0)
ori_results = torch.zeros_like(ori_list)
ori_results[ori_indices] = 1
#ori_results[1, ori_indices] = 0
sigma_list = [0.2 / 2 ** n for n in range(0, 11)]
result_dict = {}
for tau, noise in tqdm(product([0.05, 0.01, 0.005, 0.001], sigma_list)):
    #_, gs_results = gumbel_sinkhorn_topk(ori_list, 3, tau=tau, max_iter=1000, noise_fact=noise, sample_num=10000, return_prob=True)
    #diff_gs = (gs_results - ori_results.unsqueeze(0)).abs().sum() / gs_results.shape[0]
    print(f'tau={tau}, noise={noise}')
    diff_gs = []
    for sample in range(500):
        distort_list = ori_list + sample_gumbel(ori_list.shape[0]) * noise
        distort_indices = torch.sort(distort_list, descending=True).indices[:3]
        distort_results = torch.zeros_like(distort_list)
        distort_results[distort_indices] = 1
        _, gs_results = soft_topk(distort_list, 3, tau=tau, max_iter=10000, return_prob=True)
        diff_gs.append((gs_results - distort_results).abs().sum())
    diff_gs = sum(diff_gs) / len(diff_gs)
    print(f'gs-topk: diff={diff_gs:.4f}')

    _, soft_results = soft_topk(ori_list, 3, tau=tau, max_iter=1000, return_prob=True)
    diff_soft = (soft_results - ori_results).abs().sum()

    print(f'soft-topk: diff={diff_soft:.4f}')
    if tau not in result_dict:
        result_dict[tau] = [diff_gs]
    else:
        result_dict[tau].append(diff_gs)

torch.save(result_dict, 'gumbel_gap_to_discrete.pt')

sys.exit(0)

sigma = 0.1
num_samples = 10000
for sigma in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0]:
    diff = 0
    for sample_id in range(num_samples):
        ori_list = torch.tensor([5,4,3.0001,3,2,1], dtype=torch.float, device=torch.device('cuda:0'))
        distort_list = ori_list + sample_gumbel(6) * sigma
        ori_indices = torch.sort(ori_list, descending=True).indices[:3]
        distort_indices = torch.sort(distort_list, descending=True).indices[:3]
        diff += (ori_indices.sort().values - distort_indices.sort().values).abs().sum().to(dtype=torch.float)
    print(f'sigma={sigma}, gap={diff / num_samples:.6f}')
sys.exit(0)

delta = np.arange(-1, 1.0, 0.001, dtype=np.longdouble)
delta_k_kp1 = 0.1
sigma = np.arange(0.00, 1.0, 0.001, dtype=np.longdouble) + 5e-4
epsilon = 1e-3
delta, sigma = np.meshgrid(delta, sigma)
z = np.log((1+epsilon * np.exp(delta / sigma)) / (1 - epsilon)) #-np.log(1 / (1+epsilon))
fig = plt.figure(figsize=(20, 5), dpi=120)
for plot_idx, delta_k_kp1 in enumerate([0.1, 0.05, 0.01, 0.005]):
    #f_ub_1 = np.exp(z) * (sigma * np.log(delta ** 2 + 4 * sigma ** 2) - 2 * sigma * np.log(z) + np.pi * delta) / (2 * delta**2 + 8* sigma**2)
    h_ub = (sigma * np.log(sigma**2 - (2 * np.abs(delta) * sigma) / np.log(1-epsilon) + (delta**2 + 4 * sigma**2) / np.log(1-epsilon)**2) + np.pi * np.abs(delta)) \
           / ((1 - epsilon) * (2 * delta**2 + 8 * sigma**2))
    f_ub = (2 * sigma * np.log(z**2 * sigma**2 - 2*delta*sigma*z + delta**2 + 4 * sigma**2) - 2*delta*np.arctan((z - delta/sigma)/2) - 4 * sigma * np.log(z) + np.pi * delta) / ((1-epsilon) * (4 * delta**2 + 16 * sigma**2))
    #f_ub1 = (2 * sigma * np.log(z**2 * sigma**2 + 2*np.abs(delta)*sigma*z + delta**2 + 4 * sigma**2) - 4 * sigma * np.log(z) + 2 * np.pi * delta) / ((1-epsilon) * (4 * delta**2 + 16 * sigma**2))
    print(np.all(f_ub <= h_ub))
    prob_term = (1 + np.exp(delta_k_kp1 /sigma)) #** 2
    #f_ub_2 = (np.log(np.abs(delta/(sigma*epsilon)-1))) / (delta ** 2 / sigma) + 1 / (delta * epsilon - delta **2 / sigma)
    f_delta_sigma = h_ub / prob_term #np.fmin(f_ub_1, f_ub_2)
    ax = fig.add_subplot(1, 4, plot_idx+1, projection='3d')
    ax.plot_surface(delta, sigma, f_delta_sigma, cmap=plt.get_cmap('rainbow'))
    #ax.contour(delta, sigma, f_delta_sigma, zdir = 'z', offset = -15, cmap = plt.get_cmap('rainbow'))
    ax.set_zlim(bottom=0, top=30)
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'$\sigma$')
    ax.set_zlabel(r'$\frac{f^\prime(\delta, \sigma, \epsilon)}{1+\exp - \frac{x_{k+1}-x_{k}}{\sigma}}$')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_title(r'$x_k-x_{k+1}$'+ f'={delta_k_kp1}')
    ax.view_init(30, 30)
#ax.set_ylim(top=0)
fig.savefig('gumbel_fx_plot.pdf', bbox_inches='tight')
sys.exit(0)

delta = []
sigma = []
f_delta_sigma = []
with open('gumbel_result2.txt', 'r') as file:
    for line in file.readlines():
        split_line = [_.strip() for _ in line.split(',')]
        if len(split_line) == 3:
            delta.append(float(split_line[0]))
            sigma.append(float(split_line[1]))
            if split_line[2] == 'nan':
                split_line[2] = 0
            f_delta_sigma.append(float(split_line[2]))
with open('gumbel_result.txt', 'r') as file:
    for line in file.readlines():
        split_line = [_.strip() for _ in line.split(',')]
        if len(split_line) == 3:
            if float(split_line[0]) == 0 or round(float(split_line[1]) / 0.01) % 2 == 1:
                continue
            delta.append(float(split_line[0]))
            sigma.append(float(split_line[1]))
            if split_line[2] == 'nan':
                split_line[2] = 0
            f_delta_sigma.append(float(split_line[2]))

delta = np.array(delta).reshape(20, 51)[:, 1:]
sigma = np.array(sigma).reshape(20, 51)[:, 1:]
f_delta_sigma = np.array(f_delta_sigma).reshape(20, 51)[:, 1:]
fig = plt.figure(figsize=(5, 5), dpi=120)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(delta, sigma, f_delta_sigma, cmap=plt.get_cmap('rainbow'))
#ax.invert_yaxis()
ax.set_zlim(bottom=0)
ax.set_xlabel(r'$\delta$')
ax.set_ylabel(r'$\sigma$')
ax.set_zlabel(r'$f(\delta, \sigma)$')
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_ylim(top=0)
fig.savefig('gumbel_fx_plot.pdf', bbox_inches='tight')
sys.exit(0)

samples = int(1e11)
sigma = 1
delta = 1


for delta in torch.arange(0, 1, 0.1, device=torch.device('cuda:0')):
    for sigma in torch.arange(0, 1.01, 0.02, device=torch.device('cuda:0')):
        delta += 1e-3
        sigma += 1e-3
        gpu_max = 500000000
        estimate = 0
        nums = 0
        for id in tqdm(range(samples // gpu_max)):
            a = delta + sample_gumbel(gpu_max) * sigma - sample_gumbel(gpu_max) * sigma
            a = a[a > 1e-3]
            estimate += torch.sum(1 / a) * 1 / (1 + torch.exp(-delta / sigma))
            nums += a.shape[0]
        print(f'{delta:.4f}, {sigma:.4f}, {estimate / nums:.4f}', flush=True)