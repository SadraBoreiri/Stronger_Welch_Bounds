# Code for Figure 2 — *Stronger Welch Bounds and Optimal Approximate k-Designs*

This repository provides code to compute the numerical results presented in the paper and to reproduce **Figure 2**.

- **“Stronger Welch Bounds and Optimal Approximate k-Designs”**  
  Riccardo Castellano, Dmitry Grinko, Sadra Boreiri, Nicolas Brunner, Jef Pauwels  
  arXiv:2602.13099v1: https://arxiv.org/html/2602.13099v1

Figure 2 compares:
- the **standard Welch bound**,
- the **strengthened bounds** from **Theorem 5** (general case) and the **sharpened bounds** from **Theorem 6** (design-constrained regime),
- and **heuristic minima** obtained via non-convex optimization,
for the **off-diagonal overlap moment**

$$
\sum_{i\neq j} |\langle \psi_i|\psi_j\rangle|^{2k} \;=\; N^2 \mathcal{E}_k(\chi) - N.
$$

---

## Repository contents

- `Stronger Welch Bounds-Numerical Results.ipynb`  
  Main notebook that reproduces the plots used for **Figure 2** and saves them to disk.

- `Theorem5.py`  
  Implements the **Theorem 5** strengthened Welch bound (general case) via the spectrum of the **partially transposed symmetric-subspace projector**.

  **Important:** the function  
  `symmetric_proj_partial_transpose_spectrum(d, k, num_eigs=None, return_dense=False)`  
  can be used to compute the **complete spectrum** (and, optionally, the operator) of the partially transposed symmetric-subspace projector (i.e., the partially transposed projector onto
   $\vee^k \mathbb{C}^d$ ).

  The helper `compute_WB_NB(d, k, N)` computes (i) Welch bound and (ii) Theorem 5 bound in the “scaled” form used for Figure 2.

- `Theorem6.py`  
  Implements the **Theorem 6** bound in the design-constrained regime (used for the right panel of Fig. 2 for \(k=3\), \(N=d(d+1)\)).

- `Heuristic_k_frame.py`  
  Heuristic optimization for general frames (left panel). Minimizes the k-frame potential using multi-start L-BFGS-B.

- `Heuristic_MUBs.py`  
  Heuristic optimization over **unions of \(d+1\) orthonormal bases** with a 2-design penalty (right panel). Implemented in PyTorch.

---
