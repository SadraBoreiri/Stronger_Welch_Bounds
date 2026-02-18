import math
import torch

# ----------------------------
# Unitary projection via QR
# ----------------------------

def qr_unitary(Z: torch.Tensor) -> torch.Tensor:
    """
    Project a complex matrix Z (dxd) to a unitary via QR.
    Q is adjusted so diag(R) phases are absorbed, making the map smoother.
    """
    Q, R = torch.linalg.qr(Z)
    diag = torch.diagonal(R, 0, -2, -1)
    phase = diag / (diag.abs() + 1e-12)
    Q = Q * phase.conj().unsqueeze(0)
    return Q


def params_to_unitaries(params: torch.Tensor) -> torch.Tensor:
    """
    params: (d, d, d, 2) real tensor representing d complex matrices (excluding U0).
    Returns:
      U: (d+1, d, d) complex tensor, U[0]=I fixed, U[1..d]=unitaries.
    """
    device = params.device
    d = params.shape[1]
    Z = params[..., 0] + 1j * params[..., 1]  # (d, d, d) complex
    U_list = [torch.eye(d, dtype=torch.cfloat, device=device)]
    for a in range(d):
        U_list.append(qr_unitary(Z[a]))
    return torch.stack(U_list, dim=0)  # (d+1, d, d)


# ----------------------------
# Efficient E2 / E3 for union of bases
# ----------------------------

def Ek_union_of_bases(U: torch.Tensor, k: int) -> torch.Tensor:
    """
    U: (B, d, d) complex, each U[b] unitary; columns = basis vectors.
    B = d+1, N = B*d.
    Uses block structure:
      - within same basis: overlaps are delta => contributes exactly N to sum_{i,j} |<.|.>|^{2k}
      - cross bases: overlaps = entries of U[a]^† U[b]
    """
    device = U.device
    B, d, _ = U.shape
    N = B * d

    Udag = U.conj().transpose(-2, -1)
    M = torch.einsum('aij,bjk->abik', Udag, U)  # (B,B,d,d), M[a,b]=U[a]^†U[b]

    abs_2k = (M.abs() ** (2 * k)).sum(dim=(-1, -2))  # (B,B)

    # exclude diagonal blocks a=b (we account for within-basis exactly by N)
    mask = 1.0 - torch.eye(B, device=device)
    S_cross = (abs_2k * mask).sum()

    Ek = (N + S_cross) / (N * N)
    return Ek


def haar_Ek(d: int, k: int) -> float:
    # Haar moment for projective k-design: 1 / binom(d+k-1, k)
    return 1.0 / math.comb(d + k - 1, k)


# ----------------------------
# Objective with 2-design penalty
# ----------------------------

def objective_with_2design_penalty(params: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    f = (E3 - 1/d^2) + lam * (E2 - HaarE2)^2
    Returns (f, E2, E3)
    """
    d = params.shape[1]
    U = params_to_unitaries(params)

    E2 = Ek_union_of_bases(U, k=2)
    E3 = Ek_union_of_bases(U, k=3)

    target_E2 = haar_Ek(d, k=2)           # = 2/(d(d+1))
    target_E3 = 1.0 / (d * d)             # sharpened bound value for 2-designs at N=d(d+1)

    f = (E3 - target_E3) + lam * (E2 - target_E2) ** 2
    return f, E2, E3


# ----------------------------
# Heuristic optimizer
# ----------------------------

def search_MUBs_with_2design_penalty(
    d: int,
    steps: int = 12000,
    lr: float = 0.03,
    restarts: int = 10,
    lam: float = 500.0,
    seed: int = 0,
    device=None,
    print_every: int = 6000,
    e3_tol: float = 1e-8,                    # NEW: early-stop threshold on |E3-target_E3|
    stop_all_restarts_when_hit: bool = True, # NEW: stop full search once threshold is hit
    min_steps_before_stop: int = 0,          # optional: avoid stopping too early
):
    """
    Optimize over unions of d+1 orthonormal bases (U0 fixed to identity),
    with a penalty enforcing approximate 2-design (via E2).

    Early stop condition:
        abs(E3 - target_E3) <= e3_tol
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    target_E2 = haar_Ek(d, 2)
    target_E3 = 1.0 / (d * d)

    best = {"f": float("inf"), "E2": None, "E3": None, "params": None}
    global_hit = False

    for r in range(restarts):
        params = torch.randn(d, d, d, 2, device=device, dtype=torch.float32) * 0.2
        params = torch.nn.Parameter(params)
        opt = torch.optim.Adam([params], lr=lr)

        hit_this_restart = False
        hit_step = None

        for t in range(steps):
            opt.zero_grad(set_to_none=True)
            f, E2, E3 = objective_with_2design_penalty(params, lam=lam)
            f.backward()
            opt.step()

            # Values from current forward pass
            with torch.no_grad():
                E2v = E2.item()
                E3v = E3.item()
                gap2 = E2v - target_E2
                gap3 = E3v - target_E3
                gap3_abs = abs(gap3)

            if (t % print_every == 0) or (t == steps - 1):
                print(
                    f"[d={d} restart {r+1}/{restarts} step {t:5d}] "
                    f"E2={E2v:.10e} (Haar {target_E2:.10e}, diff {gap2:.3e}) | "
                    f"E3={E3v:.10e} (target {target_E3:.10e}, diff {gap3:.3e}, |diff| {gap3_abs:.3e}) | "
                    f"f={f.item():.3e}"
                )

            # NEW: early stop on E3 closeness to target
            if (t >= min_steps_before_stop) and (gap3_abs <= e3_tol):
                hit_this_restart = True
                hit_step = t
                print(
                    f"--> Early stop at restart {r+1}, step {t}: "
                    f"|E3 - target_E3| = {gap3_abs:.3e} <= {e3_tol:.1e}"
                )
                break

        # Re-evaluate at the final params of this restart
        with torch.no_grad():
            f_fin, E2_fin, E3_fin = objective_with_2design_penalty(params, lam=lam)
            fv = f_fin.item()

            if fv < best["f"]:
                best["f"] = fv
                best["E2"] = E2_fin.item()
                best["E3"] = E3_fin.item()
                best["params"] = params.detach().cpu()

        # Optionally stop all restarts once threshold is hit
        if hit_this_restart and stop_all_restarts_when_hit:
            global_hit = True
            break

    print("\nBEST RESULT")
    print(f"  d={d}")
    print(f"  E2={best['E2']:.12e}  (Haar {target_E2:.12e})  diff={best['E2']-target_E2:.3e}")
    print(f"  E3={best['E3']:.12e}  (target {target_E3:.12e})  diff={best['E3']-target_E3:.3e}")
    print(f"  |E3-target|={abs(best['E3']-target_E3):.3e}")
    print(f"  f ={best['f']:.3e}")
    if global_hit:
        print(f"  Early-stop criterion met: |E3-target_E3| <= {e3_tol:.1e}")

    return best
