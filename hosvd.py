"""Higher order Singular Value Decomposition (HOSVD)."""
from typing import Tuple, List

import torch
import tensorly as tl

tl.set_backend('pytorch')


def truncated_svd(
    A: torch.Tensor,
    k: int,
    n_iter: int = 2,
    n_oversamples: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the Truncated SVD.

    Based on fbpca's version.

    Parameters
    ----------
    A : (M, N) torch.Tensor
    k : int
    n_iter : int
    n_oversamples : int

    Returns
    -------
    u : (M, k) torch.Tensor
    s : (k,) torch.Tensor
    vt : (k, N) torch.Tensor
    """
    m, n = A.shape
    Q = torch.randn(n, k + n_oversamples)
    Q = A @ Q

    Q, _ = torch.linalg.qr(Q)

    # Power iterations
    for _ in range(n_iter):
        Q = (Q.t() @ A).t()
        Q, _ = torch.linalg.qr(Q)
        Q = A @ Q
        Q, _ = torch.linalg.qr(Q)

    QA = Q.t() @ A
    # Transpose QA to make it tall-skinny as MAGMA has optimisations for this
    # (USVt)t = VStUt
    Va, s, R = torch.linalg.svd(QA.t(), full_matrices=False)
    U = Q @ R.t()

    return U[:, :k], s[:k], Va.t()[:k, :]


def sthosvd(
    tensor: torch.Tensor,
    core_size: List[int]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Seqeuntially Truncated Higher Order SVD.

    Parameters
    ----------
    tensor : torch.Tensor,
        arbitrarily dimensional tensor
    core_size : list of int

    Returns
    -------
    torch.Tensor
        core tensor
    List[torch.Tensor]
        list of singular vectors
    List[torch.Tensor]
        list of singular vectors
    """
    intermediate = tensor
    singular_vectors, singular_values = [], []
    for mode in range(len(tensor.shape)):
        to_unfold = intermediate
        svec, sval, _ = truncated_svd(tl.unfold(to_unfold, mode), core_size[mode])
        intermediate = tl.tenalg.mode_dot(intermediate, svec.t(), mode)
        singular_vectors.append(svec)
        singular_values.append(sval)
    return intermediate, singular_vectors, singular_values
