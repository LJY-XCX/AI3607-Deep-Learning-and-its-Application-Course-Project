from copy import deepcopy
from ortools.linear_solver import pywraplp
import numpy as np
import torch


def gumbel_sinkhorn_L1(scores, row_prob, k, max_iter=100, tau=1., noise_fact=1., sample_num=1000, return_prob=False):
    anchors = torch.tensor([scores.min(), scores.max()], device=scores.device)
    dist_mat = torch.abs(scores.unsqueeze(-1) - anchors.view(1, 2))

    assert row_prob.shape[0] == scores.shape[0]
    row_prob = row_prob.to(device=scores.device)
    col_prob = torch.stack(
        (torch.full((1,), row_prob.sum() - k, dtype=torch.float, device=scores.device),
         torch.full((1,), k, dtype=torch.float, device=scores.device),),
        dim=1
    )

    sk = Sinkhorn(max_iter=max_iter, tau=tau, batched_operation=True)

    def sample_gumbel(t_like, eps=1e-20):
        """
        randomly sample standard gumbel variables
        """
        u = torch.empty_like(t_like).uniform_()
        return -torch.log(-torch.log(u + eps) + eps)

    s_rep = torch.repeat_interleave(-dist_mat.unsqueeze(0), sample_num, dim=0)
    gumbel_noise = sample_gumbel(s_rep[:, :, 0]) * noise_fact
    gumbel_noise = torch.stack((gumbel_noise, -gumbel_noise), dim=-1)
    s_rep = s_rep + gumbel_noise
    rows_rep = torch.repeat_interleave(row_prob, sample_num, dim=0)
    cols_rep = torch.repeat_interleave(col_prob, sample_num, dim=0)

    output = sk(s_rep, rows_rep, cols_rep)

    top_k_indices = torch.argsort(output[:, :, 1], dim=-1, descending=True)
    sum_prob = 0

    if return_prob:
        return top_k_indices, output[:, :, 1]
    else:
        return top_k_indices


def greedy_knapsack(weights, values, max_weight):
    selected_items = []
    selected_weight = 0
    selected_values = 0
    assert len(weights) == len(values)
    ratio = [v / w for w, v in zip(weights, values)]
    sorted_indices = np.argsort(ratio)[::-1]

    for idx in sorted_indices:
        if selected_weight + weights[idx] <= max_weight:
            selected_items.append(idx)
            selected_values += values[idx]
            selected_weight += weights[idx]

    return selected_values, selected_items


def ortools_knapsack(weights, values, max_weight, solver_name=None, linear_relaxation=True, timeout_sec=60):
    # define solver instance
    if solver_name is None:
        if linear_relaxation:
            solver = pywraplp.Solver('Knapsack',
                                    pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        else:
            solver = pywraplp.Solver('Knapsack',
                                    pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    else:
        solver = pywraplp.Solver.CreateSolver(solver_name)

    # Initialize variables
    VarX = {}
    total_weight = 0

    for item_id, item_weight in enumerate(weights):
        if linear_relaxation:
            VarX[item_id] = solver.NumVar(0.0, 1.0, f'x_{item_id}')
        else:
            VarX[item_id] = solver.BoolVar(f'x_{item_id}')
        total_weight += VarX[item_id] * item_weight

    solver.Add(total_weight <= max_weight)

    # the objective
    total_value = 0
    for item_id in range(len(values)):
        total_value += VarX[item_id] * values[item_id]

    solver.Maximize(total_value)

    if timeout_sec > 0:
        solver.set_time_limit(int(timeout_sec * 1000))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        return solver.Objective().Value(), [VarX[_].solution_value() for _ in range(len(weights))]
    else:
        print('The problem does not have an optimal solution. status={}.'.format(status))
        return solver.Objective().Value(), [VarX[_].solution_value() for _ in range(len(weights))]
