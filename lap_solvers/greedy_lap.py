import torch


def greedy_lap(s, nrows=None, ncols=None):
    r"""
    Computes LAP by always picking the largest element in matrix. The time cost is :math:`O(n^3)`, but it can be
    accelerated by GPU.

    :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
    :param nrows: :math:`(b)` number of objects in dim1
    :param ncols: :math:`(b)` number of objects in dim2
    :return: :math:`(b\times n_1 \times n_2)` permutation matrix
    """
    batch_size = s.shape[0]

    assert s.shape[2] >= s.shape[1]

    tmp_s = torch.zeros_like(s).copy_(s)
    perm = torch.zeros_like(s)

    if nrows is not None:
        for b in range(batch_size):
            tmp_s[b, nrows[b]:, :] = -float('inf')
    if ncols is not None:
        for b in range(batch_size):
            tmp_s[b, :, ncols[b]:] = -float('inf')

    for i in range(tmp_s.shape[1]):
        max_idx = torch.argmax(tmp_s.view(tmp_s.size(0), -1), dim=-1)
        row_idx = max_idx // tmp_s.shape[2]
        col_idx = max_idx % tmp_s.shape[2]
        perm[torch.arange(batch_size), row_idx, col_idx] = 1
        tmp_s[torch.arange(batch_size), row_idx, :] = -float('inf')
        tmp_s[torch.arange(batch_size), :, col_idx] = -float('inf')

    if nrows is not None:
        for b in range(batch_size):
            perm[b, nrows[b]:, :] = 0
    if ncols is not None:
        for b in range(batch_size):
            perm[b, :, ncols[b]:] = 0

    return perm
