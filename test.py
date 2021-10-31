from functools import reduce

import pytest
import tensorly as tl
import torch
from tensorly.random import random_tucker

import hosvd

tl.set_backend('pytorch')


def random_n_rank_matrix(rows, cols, rank):
    """Create a random low rank matrix."""
    A = torch.randn(rows, rank)
    B = torch.randn(rank, cols)
    return A @ B


def test_truncated_svd():
    """Test the truncated SVD."""
    rank = 10
    A = random_n_rank_matrix(2 ** 12, 2 ** 10, rank=rank)
    U, s, Vt = hosvd.truncated_svd(A, k=rank)
    assert U.shape == (A.shape[0], rank)
    assert s.shape == (rank,)
    assert Vt.shape == (rank, A.shape[1])
    reconstructed = U @ torch.diag(s) @ Vt
    assert (torch.norm(A - reconstructed).item() / torch.norm(A)).item() < 1e-6


def test_sthosvd():
    """Test the sequentially truncated svd."""
    core_size = (10, 10, 4)
    tensor = random_tucker((2 ** 8, 2 ** 5, 2 ** 3), rank=core_size, full=True)
    core, svecs, _ = hosvd.sthosvd(tensor, core_size)
    assert core.shape == core_size

    reconstructed = tl.tucker_to_tensor((core, svecs))
    assert torch.norm(tensor - reconstructed).item() / torch.norm(tensor).item() < 1e-6
