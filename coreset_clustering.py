from gumbel_sinkhorn_topk import gumbel_sinkhorn_topk
import torch
from facility_location_methods import kmeans
import time
import xlwt

def compute_objective(points, cluster_centers, choice_cluster=None):
    dist = ((points.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2).sum(dim=-1)
    if choice_cluster is None:
        choice_cluster = torch.argmin(dist, dim=1)
    return torch.sum(torch.gather(dist, 1, choice_cluster.unsqueeze(1)))


wb = xlwt.Workbook()
ws = wb.add_sheet('coreset')
ws.write(0, 0, 'name')

dataset = [torch.rand(200, 2, device=torch.device('cuda:0')) for _ in range(10)]
num_clusters = 5
coreset_size = 50

for index, points in enumerate(dataset):
    method_idx = 0
    print('-' * 20)
    print(f'{index} points={len(points)}')
    ws.write(index+1, 0, index)

    # kmeans++
    method_idx += 1
    perv_time = time.time()
    choice_cluster, cluster_centers = kmeans(points, num_clusters, init_x='plus', distance='euclidean', device=points.device)
    objective = compute_objective(points, cluster_centers, choice_cluster).item()
    print(f'{index} kmeans++ objective={objective:.4f} choice={choice_cluster} time={time.time()-perv_time}')
    if index == 0:
        ws.write(0, method_idx*2-1, 'greedy-objective')
        ws.write(0, method_idx*2, 'greedy-time')
    ws.write(index+1, method_idx*2-1, objective)
    ws.write(index+1, method_idx*2, time.time()-perv_time)

    # GS-TopK coreset
    method_idx += 1
    perv_time = time.time()
    latent_vars = torch.rand(points.shape[0], device=points.device, requires_grad=True)
    optimizer = torch.optim.Adam([latent_vars], lr=.1)
    best_obj = float('inf')
    best_top_k_indices = []
    best_found_at_idx = -1
    for train_idx in range(200):
        gumbel_weights_float = torch.sigmoid(latent_vars)
        #noise = 1 - 0.75 * train_idx / 1000
        top_k_indices, probs = gumbel_sinkhorn_topk(gumbel_weights_float, coreset_size,
                                                    tau=.05, sample_num=1000, noise_fact=0.25, return_prob=True)
        # compute objective
        dist = ((points.unsqueeze(0) - points.unsqueeze(1)) ** 2).sum(dim=-1)
        exp_dist = torch.exp(-15 / dist.mean() * dist)
        exp_dist_probs = exp_dist.unsqueeze(0) * probs.unsqueeze(-1)
        probs_per_dist = exp_dist_probs / exp_dist_probs.sum(1, keepdim=True)
        obj = (probs_per_dist * dist).sum(dim=(1, 2))
        '''
        dist = ((points.unsqueeze(0) - points.unsqueeze(1)) ** 2).sum(dim=-1)
        dist_sorted, dist_argsort = torch.sort(dist, dim=1)
        sorted_probs = probs[:, dist_argsort]
        cumsum_sorted = torch.cumsum(sorted_probs, dim=2)
        mask = (cumsum_sorted <= 1).to(dtype=torch.float)
        def index_3dtensor_by_2dmask(t, m):
            return torch.gather(t, 2, m.sum(dim=-1).to(dtype=torch.long).unsqueeze(-1)).squeeze(-1)
        t = - (index_3dtensor_by_2dmask(cumsum_sorted, mask) - 1) #/ index_3dtensor_by_2dmask(sorted_probs, mask)
        new_probs = sorted_probs.scatter_add(2, mask.sum(dim=-1).to(dtype=torch.long).unsqueeze(-1), t.unsqueeze(-1))
        new_mask = mask.scatter_add(2, mask.sum(dim=-1).to(dtype=torch.long).unsqueeze(-1), torch.ones_like(t.unsqueeze(-1))).detach_()
        probs_with_dist = new_mask * new_probs * dist_sorted.unsqueeze(0)
        obj = probs_with_dist.sum(dim=(1, 2))
        '''
        '''
        obj = []
        for sample_idx in range(probs.shape[0]):
            obj.append(0)
            for row_idx in range(points.shape[0]):
                cum_prob = 0
                for prob, col_idx in zip(probs[sample_idx][dist_argsort[row_idx]], dist_argsort[row_idx]):
                    cum_prob += prob
                    if cum_prob < 1:
                        obj[-1] += prob * dist[row_idx, col_idx]
                    else:
                        obj[-1] += (1 - cum_prob + prob) * dist[row_idx, col_idx]
                        break
        #points_w_probs = points.unsqueeze(0) * probs.unsqueeze(2)
        #top_k_points = torch.stack([torch.gather(points_w_probs[:, :, _], 1, top_k_indices) for _ in range(points_w_probs.shape[2])], dim=-1)
        #obj = [compute_objective(points, one_sample_top_k_points) for one_sample_top_k_points in top_k_points]
        obj = torch.stack(obj)
        '''
        obj.mean().backward()
        best_idx = torch.argmin(obj)
        min_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
        if min_obj < best_obj:
            best_obj = min_obj
            best_top_k_indices = top_k_indices
            best_found_at_idx = train_idx
        if train_idx % 10 == 0:
            print(f'idx:{train_idx} {obj.min():.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}')
        optimizer.step()
        optimizer.zero_grad()

    coreset_points = torch.stack([torch.gather(points[:, _], 0, best_top_k_indices) for _ in range(points.shape[1])], dim=-1)
    min_indices = torch.argmin(((points.unsqueeze(1) - coreset_points.unsqueeze(0)) ** 2).sum(dim=-1), dim=1)
    weights = torch.zeros(points.shape[0], coreset_points.shape[0], device=points.device)
    weights[torch.arange(points.shape[0]), min_indices] = 1
    weights = weights.sum(dim=0)

    _, cluster_centers = kmeans(coreset_points, num_clusters, weights, init_x='plus', distance='euclidean', device=points.device)
    objective = compute_objective(points, cluster_centers).item()

    print(f'{index} gumbel objective={objective:.4f} time={time.time()-perv_time}')
    if index == 0:
        ws.write(0, method_idx * 2 - 1, 'reparam-gumbel-objective')
        ws.write(0, method_idx * 2, 'reparam-gumbel-time')
    ws.write(index+1, method_idx*2-1, best_obj.item())
    ws.write(index+1, method_idx*2, time.time()-perv_time)

    wb.save('result.xls')
