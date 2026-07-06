# Anisotropic Adaptive Poisson Solver for P1 and P2 finite elements

This repository implements a 3D anisotropic mesh adaptation loop for the Poisson equation using [FEniCSx](https://fenicsproject.org/) and [MMG3D](https://www.mmgtools.org/). The adaptation is driven by gradient-recovery error estimators (Zienkiewicz–Zhu and Naga-Zhang/PPR) and produces anisotropic metric tensors fed to MMG3D for remeshing.

---

## Adaptive algorithm

`eta_estimator1.py` is the core estimator module. It contains the gradient estimation and recovery for ZZ, and the computation of the $\tilde{G}_K$ tensor** for the NZ. It also contains the metric building utilities. `nz_eta_estimatorP2.py` implements the NZ $G_h(u_h)$ for P2 finite elements on tetrahedral meshes in 3D. The `mesh_hepers.py` contains utility functions for mesh I/O and metric-based adaptation (Important funciton: `build_dolfinx_to_medit_map` returns the permutation `perm` such that `M_medit[perm[i]] = M_dolfinx[i]`, accounting for the difference between DOLFINx's internal vertex ordering and the original Medit file ordering.). `problems.py` defines all Poisson problem instances used in the adaptive loop (`adaptive_algo.py`). The problems have the following identifiers which are passed from the runner `poisson_cube/adaptive_poisson_cube` which executes the full anisotropic adaptive loop and arguments can be defined from `poisson_cube/run.sh` for ease of use:

| Name | Description |
| --- | --- |
| `"1d"` | 1D tanh boundary layer along $x_0$, $\epsilon$ |
| `"sphere"` | Cosine-smoothed Heaviside shell at radius $R=0.5$, $\epsilon$ |
| `"plan"` | Tanh layer on the diagonal plane $x+y=0$, $\epsilon$ |
| `"tok-sphere"` | tanhd toroidal spherical shell in tokamak geometry |
| `"tok-wall"` | Axisymmetric cosh wall-adaptation problem, radial layer at $R_c = 500$ mm |

`solver.py` contains the FEM solver used by the adaptive loop. The main generic solver is `solve_poisson_generic`, which solves $-\Delta u = f$ with Dirichlet boundary conditions using **CG-$k$ elements** (P1 or P2 depending on the `k` parameter). It uses a **CG + AMG (BoomerAMG)** linear solver and accepts either a UFL expression or a numpy callable for $f$.



<!-- #### `NZ_diagnosis/`
#
#Folder collecting all diagnostic tools for the Naga-Zhang / PPR gradient recovery:
#
#| File | Description |
#|---|---|
#| `ppr_tracker.py` | `PPRConvergenceTracker` — records six $L^2$ gradient errors (A–F) across adaptive runs and produces convergence tables and log-log plots. Also contains `_save_patch_cond_hist` for plotting $\kappa(A^T A)$ histograms. |
#| `diagnose_ppr.py` | Lightweight per-iteration diagnostic that prints quantities A–F at each adaptive loop step. |
#| `ppr_cellwise_diagnostic.py` | Writes per-cell DG-0 contributions of the A–F errors to XDMF for inspection in ParaView. |
#
#The six diagnostic quantities are:
#
#| Symbol | Quantity |
#|---|---|
#| $A$ | $\|\nabla u_h - \nabla u\|$ — true FE gradient error |
#| $B$ | $\|\nabla u_h - G_h(u_h)\|$ — PPR–FE residual |
#| $C$ | $\|G_h(u_h) - \nabla u\|$ — PPR recovery error vs. exact |
#| $D$ | $\|\nabla u - G_h(I^2 u)\|$ — recovery error on the P2 interpolant |
#| $E$ | $\|G_h(u_h) - G_h(I^2 u)\|$ — PPR sensitivity |
#| $F$ | $\|\nabla(I^2 u) - \nabla u_h\|$ — supercloseness |
#
#Expected asymptotic rates for P2 elements: $A, B \sim O(h^2)$, $C, D, F \sim O(h^3)$, $E \sim o(h^2)$. -->
