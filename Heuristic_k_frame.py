import numpy as np
from scipy.optimize import minimize

# =========================================================
# k-frame potential objective:
# FP_k = sum_{i,j=1}^N |<psi_i|psi_j>|^(2k)
# =========================================================
def kframe_potential(states, k):
    S = np.asarray(states, dtype=np.complex128)
    # safety normalization
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-18)
    G = S @ S.conj().T
    return float(np.sum(np.abs(G) ** (2.0 * k)))

# =========================================================
# Unit vector in C^d from 2d-2 params
# (d-1 amplitude angles + d-1 relative phases)
# =========================================================
def state_from_params(params, d):
    a = np.asarray(params[:d-1], dtype=np.float64)      # in [0, pi/2]
    ph = np.asarray(params[d-1:], dtype=np.float64)     # in [0, 2pi]

    amps = np.empty(d, dtype=np.float64)
    prod_sin = 1.0
    for j in range(d - 1):
        amps[j] = prod_sin * np.cos(a[j])
        prod_sin *= np.sin(a[j])
    amps[d - 1] = prod_sin

    phases = np.empty(d, dtype=np.float64)
    phases[0] = 0.0      # fix global phase
    phases[1:] = ph

    v = amps * np.exp(1j * phases)
    v /= np.linalg.norm(v)
    return v

def build_states(x, d, N, fix_first=True):
    per_state = 2 * d - 2
    S = np.empty((N, d), dtype=np.complex128)

    if fix_first:
        S[0] = np.eye(d, dtype=np.complex128)[0]  # |0>
        start, ptr = 1, 0
    else:
        start, ptr = 0, 0

    for i in range(start, N):
        S[i] = state_from_params(x[ptr:ptr + per_state], d)
        ptr += per_state

    return S

def objective(x, d, N, k, fix_first=True):
    S = build_states(x, d, N, fix_first=fix_first)
    return kframe_potential(S, k)

def make_bounds(d, N, fix_first=True):
    n_var_states = N - 1 if fix_first else N
    b = []
    for _ in range(n_var_states):
        b += [(0.0, 0.5 * np.pi)] * (d - 1)  # amplitudes
        b += [(0.0, 2.0 * np.pi)] * (d - 1)  # phases
    return b

def random_x0(d, N, rng, fix_first=True):
    n_var_states = N - 1 if fix_first else N
    per_state = 2 * d - 2
    x0 = np.empty(n_var_states * per_state, dtype=np.float64)

    idx = 0
    for _ in range(n_var_states):
        x0[idx:idx + d - 1] = rng.uniform(0.0, 0.5 * np.pi, size=d - 1)
        idx += d - 1
        x0[idx:idx + d - 1] = rng.uniform(0.0, 2.0 * np.pi, size=d - 1)
        idx += d - 1
    return x0

def optimize_kframe(d, N, k, n_restarts=80, seed=42, fix_first=True):
    rng = np.random.default_rng(seed)
    bounds = make_bounds(d, N, fix_first=fix_first)

    opts = {
        "maxiter": 30000,
        "maxfun": 300000,
        "ftol": 1e-15,
        "gtol": 1e-12,
        "eps": 1e-8,
        "maxls": 100,
    }
    opts_polish = {
        "maxiter": 60000,
        "maxfun": 600000,
        "ftol": 1e-16,
        "gtol": 1e-13,
        "eps": 5e-9,
        "maxls": 150,
    }

    best = None
    best_val = np.inf

    for _ in range(n_restarts):
        x0 = random_x0(d, N, rng, fix_first=fix_first)
        res = minimize(
            objective, x0,
            args=(d, N, k, fix_first),
            method="L-BFGS-B",
            bounds=bounds,
            options=opts
        )
        if res.fun < best_val:
            best_val = res.fun
            best = res

    # polish
    best = minimize(
        objective, best.x,
        args=(d, N, k, fix_first),
        method="L-BFGS-B",
        bounds=bounds,
        options=opts_polish
    )

    S_opt = build_states(best.x, d, N, fix_first=fix_first)
    fp_k = kframe_potential(S_opt, k)
    return fp_k, S_opt, best
