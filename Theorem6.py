import numpy as np     
import math
from math import comb
from tqdm import tqdm



# %%
# MUB overlap moment formula

def mub_overlap_moment_formula(d: int, m: int, k: int, include_diagonal: bool = True) -> float:
    """
    Compute S_k = sum_{i,j} |<psi_i, psi_j>|^{2k} for an m-element set of MUBs in C^d,
    using only the defining properties of MUBs (no explicit vectors).

    d: dimension
    m: number of mutually unbiased bases
    k: positive integer, we compute |<psi_i, psi_j>|^(2k)
    include_diagonal: if False, sum only over i != j
    """
    # total number of vectors
    N = m * d

    # cross-basis contribution: m(m-1)*d^2 pairs, each with (1/d)^k
    cross = m * (m - 1) * (d ** (2 - k))

    if include_diagonal:
        # diagonal: N terms with value 1
        diag = N
        return diag + cross
    else:
        return cross

    


def comb_safe(n: int, r: int) -> int:
    """Binomial(n, r) with out-of-range returning 0."""
    if r < 0 or n < 0 or r > n:
        return 0
    return math.comb(n, r)

def D_kd(k: int, d: int) -> int:
    """D(k,d)"""
    a = k // 2                 # floor(k/2)
    b = (k + 1) // 2           # ceil(k/2)
    if a == 0 or b == 0:
        raise ValueError("This formula needs k >= 2 so that floor/ceil(k/2)-1 are nonnegative.")
    return comb_safe(d + a - 1, a ) * comb_safe(d + b - 1, b)

def eta_r(k: int, d: int, r: int) -> float:
    """eta_r(k,d) from Eq. (49)."""
    a = (k + 1) // 2           # ceil(k/2)
    b = k // 2                 # floor(k/2)
    pref = (k - 2*r + d - 1) / (d - 1)
    # For d=2, the choose(..., d-2) is choose(..., 0) = 1 (handled by comb_safe).
    t1 = comb_safe(a - r + d - 2, d - 2)
    t2 = comb_safe(b - r + d - 2, d - 2)
    return pref * t1 * t2

def C_r(k: int, d: int, r: int) -> float:
    """C_r(k,d) from Eq. (50)."""
    
    denom = comb_safe(k, k // 2) * comb_safe(k+d-1, k)
    if denom == 0:
        raise ZeroDivisionError("Denominator vanished; check k,d.")
    return comb_safe(k + d - 1, r) / denom

def Delta(k: int, d: int, N: int) -> float:
    """Delta(k,d,N) from Eq. (72), using r=floor(k/2)."""
    #r = k // 2
    D = float(D_kd(k, d))
    #C = float(C_r(k, d, r))
    #eta = float(eta_r(k, d, r))
    
    
    C = C_r(k, d, 0) #min([C_r(k, d, r) for r in range(k//2+1)]) #C_min
    eta = eta_r(k, d, 0) #max([eta_r(k, d, r) for r in range(k//2+1)]) #eta_max
    
    
    

    denom = eta + N - D  # (eta_{floor(k/2)} + N - D(k,d))
    if denom == 0:
        
        return float("inf") if (D - N) != 0 else float("nan")

    return (C**2) * (D - N) * (1.0 + (D-N) / denom)

def our_bound(k: int, d: int, N: int) -> float:
    """
    Computes the RHS of Eq. (71):
        ||g_k||_2^2 >= N^2 / binom(d+k-1, k) + N^2 * Delta(k,d,N)
    """
    base_den = comb_safe(d + k - 1, k)
    if base_den == 0:
        raise ZeroDivisionError("binom(d+k-1, k) is zero; check k,d.")
    base = (N**2) / base_den
    return base + (N**2) * Delta(k, d, N)


def Welch_bound(k: int, d: int, N: int) -> float:
    
    base_den = comb_safe(d + k - 1, k)
        
    return (N**2) / base_den
