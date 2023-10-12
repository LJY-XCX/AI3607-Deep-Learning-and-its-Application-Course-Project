from gumbel_sinkhorn_topk import gumbel_sinkhorn_topk
import torch
from legacy.knapsack_traditional_methods import greedy_knapsack, ortools_knapsack
import time
import xlwt
from datetime import datetime

####################################
#             config               #
####################################
methods = [
    #'greedy',
    'gurobi',
    #'scip',
    #'egn',
    #'soft-topk',
    #'soft-topk-sampling',
    #'gs-topk'
]
num_items = 5000 #2000/1000
max_weight = 10
gumbel_sample_num = 1000
solver_timeout = 600 #120 #30
egn_beta = 1

device = torch.device('cuda:0')

def compute_obj_differentiable(values, latent_probs, device=torch.device('cpu')):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float, device=device)
    if not isinstance(latent_probs, torch.Tensor):
        latent_probs = torch.tensor(latent_probs, device=device)
    return torch.sum(values * latent_probs)

#dataset = SCP_ORLIB()
import random
random.seed(1)
train_dataset = []
for i in range(100):
    weights = [random.random() * 1. for _ in range(num_items)]
    values = [random.random() * 1. for _ in range(num_items)]
    train_dataset.append((f'rand{i}', weights, values))

for method_name in methods:
    model_path = f'knapsack_{num_items}-{max_weight}_{method_name}.pt'

random.seed(0)
dataset = []
for i in range(100):
    weights = [random.random() * 1. for _ in range(num_items)]
    values = [random.random() * 1. for _ in range(num_items)]
    dataset.append((f'rand{i}', weights, values))
dataset = [dataset[0] for _ in range(100)]

wb = xlwt.Workbook()
ws = wb.add_sheet(f'knapsack_{num_items}-{max_weight}')
ws.write(0, 0, 'name')
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

torch.random.manual_seed(1)
for index, (name, weights, values) in enumerate(dataset):
    method_idx = 0
    print('-' * 20)
    print(f'{name} items={len(weights)} max weight={max_weight}')
    ws.write(index+1, 0, name)

    if 'greedy' in methods:
        # greedy
        method_idx += 1
        perv_time = time.time()
        objective, selected = greedy_knapsack(weights, values, max_weight)
        print(f'{name} greedy objective={objective:.4f} selected={sorted(selected)} time={time.time()-perv_time}')
        if index == 0:
            ws.write(0, method_idx*2-1, 'greedy-objective')
            ws.write(0, method_idx*2, 'greedy-time')
        ws.write(index+1, method_idx*2-1, objective)
        ws.write(index+1, method_idx*2, time.time()-perv_time)

    if 'scip' in methods:
        # SCIP - integer programming
        method_idx += 1
        perv_time = time.time()
        ip_obj, ip_scores = ortools_knapsack(weights, values, max_weight, solver_name='SCIP', linear_relaxation=False, timeout_sec=solver_timeout)
        ip_scores = torch.tensor(ip_scores)
        selected = torch.nonzero(ip_scores, as_tuple=False).view(-1)
        selected = sorted(selected.cpu().numpy().tolist())
        print(f'{name} SCIP objective={ip_obj:.4f} selected={selected} time={time.time()-perv_time}')
        if index == 0:
            ws.write(0, method_idx*2-1, 'SCIP-objective')
            ws.write(0, method_idx*2, 'SCIP-time')
        ws.write(index+1, method_idx*2-1, ip_obj)
        ws.write(index+1, method_idx*2, time.time()-perv_time)

    if 'gurobi' in methods:
        # Gurobi - integer programming
        method_idx += 1
        perv_time = time.time()
        ip_obj, ip_scores = ortools_knapsack(weights, values, max_weight, solver_name='Gurobi', linear_relaxation=False, timeout_sec=solver_timeout)
        ip_scores = torch.tensor(ip_scores)
        selected = torch.nonzero(ip_scores, as_tuple=False).view(-1)
        selected = sorted(selected.cpu().numpy().tolist())
        print(f'{name} Gurobi objective={ip_obj:.4f} time={time.time()-perv_time} selected={selected}')
        if index == 0:
            ws.write(0, method_idx*2-1, 'Gurobi-objective')
            ws.write(0, method_idx*2, 'Gurobi-time')
        ws.write(index+1, method_idx*2-1, ip_obj)
        ws.write(index+1, method_idx*2, time.time()-perv_time)

    weights = torch.tensor(weights, dtype=torch.float, device=device)

    def sinkhorn_knapsack(sample_num, noise, tau, sk_iters, opt_iters, sample_num2=None, noise2=None):
        latent_vars = torch.rand_like(weights)
        latent_vars.requires_grad_(True)
        optimizer = torch.optim.Adam([latent_vars], lr=.1)
        best_obj = 0
        best_top_k_indices = []
        best_found_at_idx = -1
        if type(noise) == list and type(tau) == list and type(sk_iters) == list and type(opt_iters) == list:
            iterable = zip(noise, tau, sk_iters, opt_iters)
        else:
            iterable = zip([noise], [tau], [sk_iters], [opt_iters])
        for noise, tau, sk_iters, opt_iters in iterable:
            for train_idx in range(opt_iters):
                gumbel_weights_float = torch.sigmoid(latent_vars)
                # noise = 1 - 0.75 * train_idx / 1000
                top_k_indices, probs = gumbel_sinkhorn_topk(gumbel_weights_float, max_covering_items,
                    max_iter=sk_iters, tau=tau, sample_num=sample_num, noise_fact=noise, return_prob=True)
                obj, bipartite_adj = compute_obj_differentiable(weights, sets, probs, bipartite_adj, probs.device)
                (-obj).mean().backward()
                if train_idx % 10 == 0:
                    print(f'idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}')
                if sample_num2 is not None and noise2 is not None:
                    top_k_indices, probs = gumbel_sinkhorn_topk(gumbel_weights_float, max_covering_items,
                    max_iter=sk_iters, tau=tau, sample_num=sample_num2, noise_fact=noise2, return_prob=True)
                obj = compute_objective(weights, sets, top_k_indices, bipartite_adj, device=device)
                best_idx = torch.argmax(obj)
                max_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
                if max_obj > best_obj:
                    best_obj = max_obj
                    best_top_k_indices = top_k_indices
                    best_found_at_idx = train_idx
                if train_idx % 10 == 0:
                    print(
                        f'idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}')
                optimizer.step()
                optimizer.zero_grad()
        return best_obj, best_top_k_indices

    # SOFT-TopK
    if 'soft-topk' in methods:
        method_idx += 1
        perv_time = time.time()
        model_path = f'max_covering_{max_covering_items}-{num_sets}-{num_items}_soft-topk.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = sinkhorn_max_covering(1, 0, .05, 100, 8000)
        #best_obj, best_top_k_indices = sinkhorn_max_covering(1, 0, args.tau, 100, 500)
        print(f'{name} SOFT-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time()-perv_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'SOFT-TopK-objective')
            ws.write(0, method_idx * 2, 'SOFT-TopK-time')
        ws.write(index+1, method_idx*2-1, best_obj.item())
        ws.write(index+1, method_idx*2, time.time()-perv_time)

    # SOFT-TopK w/ sampling
    if 'soft-topk-sampling' in methods:
        method_idx += 1
        perv_time = time.time()
        model_path = f'max_covering_{max_covering_items}-{num_sets}-{num_items}_soft-topk.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = sinkhorn_max_covering(1, 0, .05, 100, 1500, gumbel_sample_num, 0.25)
        # best_obj, best_top_k_indices = sinkhorn_max_covering(1, 0, args.tau, 100, 1500, gumbel_sample_num, 0.25)
        print(f'{name} SOFT-TopK-Sampling objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time()-perv_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'SOFT-TopK-Sampling-objective')
            ws.write(0, method_idx * 2, 'SOFT-TopK-Sampling-time')
        ws.write(index+1, method_idx*2-1, best_obj.item())
        ws.write(index+1, method_idx*2, time.time()-perv_time)

    # GS-TopK
    if 'gs-topk' in methods:
        method_idx += 1
        perv_time = time.time()
        model_path = f'max_covering_{max_covering_items}-{num_sets}-{num_items}_gs-topk.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = sinkhorn_max_covering(gumbel_sample_num, 0.15, .05, 100, 1000)
        # best_obj, best_top_k_indices = sinkhorn_max_covering(gumbel_sample_num, args.noise, args.tau, 100, 1000)
        print(f'{name} GS-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time()-perv_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'GS-TopK-objective')
            ws.write(0, method_idx * 2, 'GS-TopK-time')
        ws.write(index+1, method_idx*2-1, best_obj.item())
        ws.write(index+1, method_idx*2, time.time()-perv_time)

        #method_idx += 1
        #perv_time = time.time()
        #model.load_state_dict(torch.load(model_path))
        #best_obj, best_top_k_indices = sinkhorn_max_covering(gumbel_sample_num, [0.15, 0.1, 0.05], [.05, 0.04, 0.03], [100, 200, 300], [1000, 100, 100])
        #print(
        #    f'{name} Homotopy-GS-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - perv_time}')
        #if index == 0:
        #    ws.write(0, method_idx * 2 - 1, 'Homotopy-GS-TopK-objective')
        #    ws.write(0, method_idx * 2, 'Homotopy-GS-TopK-time')
        #ws.write(index + 1, method_idx * 2 - 1, best_obj.item())
        #ws.write(index + 1, method_idx * 2, time.time() - perv_time)

    #wb.save(f'max_covering_result__{max_covering_items}-{num_sets}-{num_items}_{timestamp}.xls')
    #wb.save(f'max_covering_result__{max_covering_items}-{num_sets}-{num_items}_{gumbel_sample_num}-{args.noise}-{args.tau}_{timestamp}.xls')
