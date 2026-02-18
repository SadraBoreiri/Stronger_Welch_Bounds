import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from math import floor
import math



# %%
# Tools to build the symmetric projector and apply partial transpose

from typing import Iterable, List, Optional, Sequence, Tuple


def _compute_symbol_labels(dim: int, copies: int) -> Tuple[np.ndarray, int]:
    """Return orbit labels for computational basis states under qudit permutations."""
    if dim <= 0 or copies <= 0:
        raise ValueError("Both dim and copies must be positive integers.")

    total_dim = dim ** copies
    # Encode occupation vectors in mixed radix to avoid storing a dense (total_dim x dim) array.
    base = copies + 1
    weights = (base ** np.arange(dim, dtype=np.int64))
    labels = np.zeros(total_dim, dtype=np.int64)
    work = np.arange(total_dim, dtype=np.int64)
    for _ in range(copies):
        digits = work % dim
        labels += weights[digits]
        work //= dim

    _, inverse = np.unique(labels, return_inverse=True)
    return inverse, total_dim


def symmetric_projection(dim: int, p_val: int = 2, partial: bool = False) -> sparse.csr_matrix:
    """
    Sparse projector onto the symmetric subspace of (C^d)^{⊗ p_val}.

    NOTE: This returns a normalized projector (divided by the number of orbits)
    """
    if partial:
        raise NotImplementedError("partial=True is not supported in the sparse variant.")
    if p_val == 1:
        return sparse.eye(dim, format="csr", dtype=np.float64) / dim

    labels, total_dim = _compute_symbol_labels(dim, p_val)
    order = np.argsort(labels, kind="stable")
    split_points = np.flatnonzero(np.diff(labels[order])) + 1
    groups = np.split(order, split_points)
    num_groups = len(groups)

    row_parts: List[np.ndarray] = []
    col_parts: List[np.ndarray] = []
    data_parts: List[np.ndarray] = []
    for group in groups:
        if group.size == 0:
            continue
        m = group.size
        coeff = 1.0 / m
        row_parts.append(np.repeat(group, m))
        col_parts.append(np.tile(group, m))
        data_parts.append(np.full(m * m, coeff, dtype=np.float64))

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.concatenate(data_parts)
    projector = sparse.coo_matrix((data, (rows, cols)), shape=(total_dim, total_dim)).tocsr()
    return projector / num_groups


def _to_tuple_dims(dim: int, k: int) -> Sequence[int]:
    return tuple([dim] * k)


def _partial_transpose_sparse(
    mat: sparse.spmatrix,
    sys: Iterable[int],
    dims: Sequence[int],
) -> sparse.csr_matrix:
    """Apply the partial transpose to a sparse matrix over specified subsystems."""
    sys = tuple(sys)
    if not sys:
        return mat.tocsr()

    dims = tuple(dims)
    order = len(dims)
    if max(sys, default=-1) >= order:
        raise ValueError("Subsystem index out of range for provided dimensions.")

    coo = mat.tocoo()
    row_multi = np.vstack(np.unravel_index(coo.row, dims)).T
    col_multi = np.vstack(np.unravel_index(coo.col, dims)).T

    # swap row/column indices for the subsystems being transposed
    for subsystem in sys:
        temp = row_multi[:, subsystem].copy()
        row_multi[:, subsystem] = col_multi[:, subsystem]
        col_multi[:, subsystem] = temp

    new_rows = np.ravel_multi_index(row_multi.T, dims)
    new_cols = np.ravel_multi_index(col_multi.T, dims)
    return sparse.coo_matrix((coo.data, (new_rows, new_cols)), shape=mat.shape).tocsr()



# %%
# Eigenvalue computation utilities

def _compute_eigenvalues(
    hermitian: sparse.spmatrix,
    num_eigs: Optional[int],
) -> np.ndarray:
    """Compute eigenvalues using either dense or sparse routines."""
    size = hermitian.shape[0]
    if num_eigs is None or num_eigs <= 0 or num_eigs >= size - 1:
        dense = hermitian.toarray()
        evals = np.linalg.eigvalsh(dense)
        return np.sort(evals.real)[::-1]

    max_k = size - 2
    if max_k <= 0:
        dense = hermitian.toarray()
        evals = np.linalg.eigvalsh(dense)
        return np.sort(evals.real)[::-1]

    k = min(num_eigs, max_k)
    evals = spla.eigsh(hermitian, k=k, which="LA", return_eigenvectors=False)
    return np.sort(evals.real)[::-1]


def summarize_eigenspectrum(evals: np.ndarray, decimals: int = 8) -> List[Tuple[float, int]]:
    """Group eigenvalues by rounding to the requested precision."""
    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    rounded = np.round(evals, decimals=decimals)
    unique_vals, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(unique_vals)[::-1]
    return [(float(unique_vals[idx]), int(counts[idx])) for idx in order]


def format_degeneracy_table(summary: Sequence[Tuple[float, int]], decimals: int = 8) -> str:
    """Render a text table of eigenvalues and degeneracies."""
    if not summary:
        return "(no eigenvalues)"

    header = f"{'Eigenvalue':>20}  {'Degeneracy':>10}"
    separator = "-" * len(header)
    lines = [header, separator]
    for value, multiplicity in summary:
        lines.append(f"{value:>20.{decimals}f}  {multiplicity:>10d}")
    return "\n".join(lines)



# %%
# Build the partially transposed symmetric projector and get its spectrum

def symmetric_proj_partial_transpose_spectrum(
    d: int,
    k: int,
    num_eigs: Optional[int] = None,
    return_dense: bool = False,
):
    """
    Construct the symmetric projector for (C^d)^{⊗ k}, partially transpose floor(k/2)
    subsystems, and return the spectrum.

    Returns:
        evals: Eigenvalues sorted in decreasing order.
        P_sym_pt: Partially transposed matrix, sparse by default (or dense if return_dense=True).
    """
    dims_list = _to_tuple_dims(d, k)

    # 1) Symmetric projector -> sparse format
    P_sym = symmetric_projection(dim=d, p_val=k)

    # 2) Determine subsystems to partially transpose
    num_pt = floor(k / 2)
    pt_subsystems = list(range(num_pt))

    # 3) Sparse partial transpose
    P_sym_pt = _partial_transpose_sparse(P_sym, pt_subsystems, dims_list)
    P_sym_pt = (P_sym_pt + P_sym_pt.getH()) * 0.5
    P_sym_pt = P_sym_pt.tocsr()

    # 4) Eigenvalues via sparse/dense pipeline
    evals = _compute_eigenvalues(P_sym_pt, num_eigs=num_eigs)

    matrix_out = P_sym_pt.toarray() if return_dense else P_sym_pt
    return evals, matrix_out



# %%
# Distinct eigenvalues (lambdas) and degeneracies

def distinct_eigenvalues(evals, tol: float = 1e-12):
    """
    Given an array of eigenvalues (with degeneracy), find
    the distinct eigenvalues (lambdas) and their degeneracies.

    Parameters
    ----------
    evals : array_like
        Eigenvalues, possibly with repetitions.
    tol : float
        Numerical tolerance for grouping (two eigenvalues whose
        difference is <= tol are treated as equal).

    Returns
    -------
    lambdas : np.ndarray
        Sorted distinct eigenvalues (descending).
    degeneracies : np.ndarray
        Corresponding degeneracies (same order as lambdas).
    """
    evals = np.asarray(evals, dtype=float)
    # sort in decreasing order
    evals_sorted = np.sort(evals)[::-1]

    lambdas = []
    degeneracies = []

    for val in evals_sorted:
        if not lambdas or abs(val - lambdas[-1]) > tol:
            # new distinct eigenvalue
            lambdas.append(val)
            degeneracies.append(1)
        else:
            # same as last eigenvalue (within tol)
            degeneracies[-1] += 1

    lambdas = np.array(lambdas)
    degeneracies = np.array(degeneracies, dtype=int)

    # print(f"Number of distinct eigenvalues: {len(lambdas)}")
    return lambdas, degeneracies



# %%
# Function that, for a given d and k, computes WB, NB, and MUB

def compute_WB_NB(d: int, k: int, N: int, tol: float = 1e-10):
    """
    For given d, k, compute:
      - WB  : Welch bound-like quantity (N^2 / C(d+k-1, k))
      - NB : our bound using lambdas and tail (Delta1, Delta2)
    """
    # 1) Spectrum of partially transposed symmetric projector
    evals, _ = symmetric_proj_partial_transpose_spectrum(
        d=d,
        k=k,
        num_eigs=None,
        return_dense=False,
    )

    # 2) Distinct eigenvalues (lambdas) and degeneracies
    lambdas, degeneracies = distinct_eigenvalues(evals, tol=tol)

    
    
    # Dimension of the matrix (number of eigenvalues)
    D = np.sum(evals > 1e-7)
    #print("D is = ", D)
    #print(degeneracies)
    
    T = float(N)       # coefficient T 
  
    # Welch bound-like quantity
    base_dim = math.comb(d + k - 1, k)
    WB = (N ** 2) / base_dim

    # Reconstruct full spectrum with degeneracies
    full_lambdas = np.repeat(lambdas, degeneracies)
    full_lambdas = np.sort(full_lambdas)[::-1]   # λ_1 ≥ λ_2 ≥ ...

    

    if full_lambdas.size < D:
        raise ValueError(f"Need at least D={D} eigenvalues, got {full_lambdas.size}")

    # Tail from i = N+1 to D (Python indices N .. D-1)
    if N < D:
        tail = full_lambdas[N:D]
        Delta1 = tail.sum()
        Delta2 = np.dot(tail, tail)
    else:
        # No tail
        Delta1 = 0.0
        Delta2 = 0.0

    # NB bound formula:
    NB = WB + (N ** 2) * (Delta2 + (Delta1 ** 2 / T))

    

    return WB, NB, {
        "evals": evals,
        "lambdas": lambdas,
        "degeneracies": degeneracies,
        "Delta1": Delta1,
        "Delta2": Delta2,
        "base_dim": base_dim,
        "N": N,
        "D": D,
    }
