# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false, reportReturnType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false
import numpy as np

import dolfinx
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import form
from dolfinx.fem.petsc import LinearProblem
import ufl
import basix.ufl

def compute_iso_eta(u_h: fem.Function, f: ufl.core.expr.Expr, g_N: ufl.core.expr.Expr | None = None) -> np.ndarray: # type: ignore

    domain = u_h.function_space.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    V0 = fem.functionspace(domain, ("DG", 0))
    v0 = ufl.TestFunction(V0)
    h = ufl.CellDiameter(domain)
    n = ufl.FacetNormal(domain)

    R_K = ufl.div(ufl.grad(u_h)) + f
    b1 = fem.assemble_vector(form(h**2 * ufl.inner(R_K, R_K) * v0 * ufl.dx)) # pyright: ignore[reportOperatorIssue]

    h_K = h('+')
    jump_n = ufl.jump(ufl.grad(u_h), n)
    b2 = fem.assemble_vector(
        form(0.25 * h_K * ufl.inner(jump_n, jump_n) * (v0('+') + v0('-')) * ufl.dS) # type: ignore
    )

    if g_N is not None:
        neumann_res = ufl.dot(ufl.grad(u_h), n) - g_N
        b3 = fem.assemble_vector(
            form(h * ufl.inner(neumann_res, neumann_res) * v0 * ufl.ds)
        )
        b3_array = b3.array
    else:
        # pure Dirichlet: boundary contribution is exactly zero
        b3_array = np.zeros_like(b2.array)

    # no n_owned slicing: all entries are owned on a single process
    term1 = np.sqrt(b1.array)
    term2 = np.sqrt(b2.array + b3_array)
    return term2 + term1


def compute_gradient_dg0(u_h: fem.Function):

    domain = u_h.function_space.mesh
    gdim = domain.geometry.dim

    dg0_element = basix.ufl.element("DG", domain.topology.cell_name(), 0, shape = (gdim,))
    V_dg0_vec = fem.functionspace(domain, dg0_element)
    interp_points = V_dg0_vec.element.interpolation_points
    grad_expr = fem.Expression(ufl.grad(u_h), interp_points)
    grad_dg0 = fem.Function(V_dg0_vec)
    grad_dg0.interpolate(grad_expr)

    return grad_dg0

def compute_zz_grad(u_h: fem.Function):
    domain = u_h.function_space.mesh
    tdim = domain.topology.dim
    gdim = domain.geometry.dim

    domain.topology.create_entities(0)
    domain.topology.create_connectivity(tdim, 0)
    domain.topology.create_connectivity(0, tdim)
    cell_to_vertex = domain.topology.connectivity(tdim, 0)
    vertex_to_cell = domain.topology.connectivity(0, tdim)

    V0 = fem.functionspace(domain, ("DG", 0))
    v0 = ufl.TestFunction(V0)
    n_cells = V0.dofmap.index_map.size_local
    one = fem.Constant(domain, default_scalar_type(1.0))
    b_vol = fem.assemble_vector(form(one * v0 * ufl.dx))
    vol = b_vol.array

    grad_dg0 = compute_gradient_dg0(u_h)
    V_cg = fem.functionspace(domain, ("Lagrange", 1))
    n_vertices = domain.topology.index_map(0).size_local

    # Build vertex→DOF map without a Python loop.
    # cell_to_vertex.array and dofmap.list (shape n_cells×4) are both in the
    # same (cell, local-vertex) order for P1 Lagrange, so a direct scatter is
    # correct: every vertex is written consistently (same DOF each time).
    ctv_flat = cell_to_vertex.array                          # (n_cells*4,) global vertex ids
    dof_flat = np.asarray(V_cg.dofmap.list).ravel()         # (n_cells*4,) global DOF ids
    vertex_to_dof_arr = np.empty(n_vertices, dtype=np.intp)
    vertex_to_dof_arr[ctv_flat] = dof_flat

    # Expand vertex index once: vertex_rep[k] = owner vertex of vtc_cells[k]
    vtc_cells   = vertex_to_cell.array                       # flat cell indices (variable patch size)
    vtc_offsets = vertex_to_cell.offsets                     # (n_vertices+1,)
    vertex_rep  = np.repeat(np.arange(n_vertices, dtype=np.intp), np.diff(vtc_offsets))

    grad_arr = grad_dg0.x.array[:n_cells * gdim].reshape(n_cells, gdim)

    # Volume per patch entry — computed once, reused across directions
    vol_entries = vol[vtc_cells]
    den = np.bincount(vertex_rep, weights=vol_entries, minlength=n_vertices)

    Pi_funcs = []
    for i in range(gdim):
        num = np.bincount(vertex_rep,
                          weights=vol_entries * grad_arr[vtc_cells, i],
                          minlength=n_vertices)
        Pi_gi = fem.Function(V_cg)
        Pi_gi.x.array[vertex_to_dof_arr] = num / den
        Pi_funcs.append(Pi_gi)

    return tuple(Pi_funcs)

def compute_eta_zz(u_h: fem.Function):

    domain = u_h.function_space.mesh
    gdim = domain.geometry.dim
    Pi_funcs = compute_zz_grad(u_h)
    etas = []

    for i in range(gdim):
        eta_i = ufl.grad(u_h)[i] - Pi_funcs[i]
        etas.append(eta_i)

    return etas

def compute_G_tilde(
    u_h: fem.Function,
) -> tuple[dict[tuple[int, int], np.ndarray], fem.Function]:
    """Compute per-cell G̃_K matrices and the ZZ error as a storable Function.

    Returns
    -------
    G : dict
        Mapping ``(i, j) -> array(n_cells)`` of G̃_K matrix entries.
    eta_zz_fn : fem.Function
        DG0 vector function representing ``grad(u_h) - Pi_h(grad(u_h))``
        (the ZZ gradient error), stored for later I/O.
    """
    domain = u_h.function_space.mesh
    gdim = domain.geometry.dim
    etas = compute_eta_zz(u_h)  # list of gdim UFL expressions: grad(u_h)[i] - Pi_h[i]
    V0 = fem.functionspace(domain, ("DG", 0))
    v0 = ufl.TestFunction(V0)
    n_cells = V0.dofmap.index_map.size_local

    G = {}
    for i in range(gdim):
        for j in range(i, gdim):
            b = fem.assemble_vector(form(etas[i] * etas[j] * v0 * ufl.dx))
            G[(i, j)] = b.array[:].copy()

    # Build a storable DG0 vector Function for eta_zz
    dg0_vec_el = basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(gdim,))
    V_dg0_vec = fem.functionspace(domain, dg0_vec_el)
    eta_zz_fn = fem.Function(V_dg0_vec, name="eta_zz")
    eta_zz_expr = ufl.as_vector([etas[i] for i in range(gdim)])
    eta_zz_fn.interpolate(fem.Expression(eta_zz_expr, V_dg0_vec.element.interpolation_points))

    return G, eta_zz_fn


def compute_G_tilde_nz(
    u_h: fem.Function,
) -> tuple[dict[tuple[int, int], np.ndarray], fem.Function]:
    """Like compute_G_tilde but uses the Naga-Zhang recovered gradient (P2).

    Replaces the ZZ recovery Pi_h with Gh(u_h) from nz_eta_estimatorP2.
    The rest of the metric pipeline (compute_G_P, adapt_h, …) is unchanged.
    """
    from nz_eta_estimatorP2 import Gh  # local import: requires numba

    domain = u_h.function_space.mesh
    gdim = domain.geometry.dim

    G_nz = Gh(u_h)  # fem.Function in P2 vector space
    etas = [ufl.grad(u_h)[i] - G_nz[i] for i in range(gdim)]

    V0 = fem.functionspace(domain, ("DG", 0))
    v0 = ufl.TestFunction(V0)
    n_cells = V0.dofmap.index_map.size_local

    G = {}
    for i in range(gdim):
        for j in range(i, gdim):
            b = fem.assemble_vector(form(etas[i] * etas[j] * v0 * ufl.dx))
            G[(i, j)] = b.array[:n_cells].copy()

    dg0_vec_el = basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(gdim,))
    V_dg0_vec = fem.functionspace(domain, dg0_vec_el)
    eta_nz_fn = fem.Function(V_dg0_vec, name="eta_nz")
    eta_nz_expr = ufl.as_vector([etas[i] for i in range(gdim)])
    eta_nz_fn.interpolate(fem.Expression(eta_nz_expr, V_dg0_vec.element.interpolation_points))

    return G, eta_nz_fn


def get_G_matrix(G: dict[tuple[int,int], np.ndarray], K: int, gdim: int) -> np.ndarray:

    mat = np.zeros((gdim, gdim))
    for i in range(gdim):
        for j in range(i, gdim):
            mat[i, j] = G[(i, j)][K]
            mat[j, i] = G[(i, j)][K]

    return mat


def compute_G_P(
    u_h: fem.Function,
    G: dict[tuple[int, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    domain = u_h.function_space.mesh
    tdim = domain.topology.dim
    gdim = domain.geometry.dim

    domain.topology.create_entities(0)
    domain.topology.create_connectivity(tdim, 0)
    cell_to_vertex = domain.topology.connectivity(tdim, 0)

    n_cells = domain.topology.index_map(tdim).size_local
    n_vertices = domain.topology.index_map(0).size_local

    if G is None:
        G, _ = compute_G_tilde(u_h)

    # Build (cell, vertex) flat CSR — cell_rep[k] is the cell owning ctv_flat[k]
    ctv_flat = cell_to_vertex.array                                      # (n_cells*(tdim+1),)
    cell_rep  = np.repeat(np.arange(n_cells, dtype=np.intp), tdim + 1)  # same length

    # Scatter symmetric G entries to vertex patches: one bincount per unique (i,j)
    vertex_sums = np.zeros((n_vertices, gdim, gdim))
    for i in range(gdim):
        for j in range(i, gdim):
            s = np.bincount(ctv_flat,
                            weights=G[(i, j)][cell_rep],
                            minlength=n_vertices)
            vertex_sums[:, i, j] = s
            vertex_sums[:, j, i] = s

    # Batched eigh over all vertices at once — no Python loop
    _, Q = np.linalg.eigh(vertex_sums)   # Q: (n_vertices, gdim, gdim)

    return vertex_sums, Q


def compute_sigma_P(
    u_h: fem.Function,
    eta_k_i: np.ndarray,
) -> np.ndarray:

    domain = u_h.function_space.mesh
    tdim = domain.topology.dim

    domain.topology.create_entities(0)
    domain.topology.create_connectivity(tdim, 0)
    cell_to_vertex = domain.topology.connectivity(tdim, 0)

    n_cells = domain.topology.index_map(tdim).size_local
    n_vertices = domain.topology.index_map(0).size_local

    # Per-cell contributions — vectorized over all cells at once
    num_per_cell = np.sum(eta_k_i ** 2, axis=0)            # (n_cells,)
    den_per_cell = np.sqrt(np.sum(eta_k_i ** 4, axis=0))   # (n_cells,)

    # Scatter to vertex patches: repeat each cell's value once per patch vertex
    ctv_flat = cell_to_vertex.array                                      # (n_cells*(tdim+1),)
    cell_rep  = np.repeat(np.arange(n_cells, dtype=np.intp), tdim + 1)

    numerator   = np.bincount(ctv_flat, weights=num_per_cell[cell_rep], minlength=n_vertices)
    denominator = np.bincount(ctv_flat, weights=den_per_cell[cell_rep], minlength=n_vertices)

    return numerator / denominator


def compute_lambda_P(
    u_h: fem.Function,
    svd: dict | None = None,
) -> np.ndarray:

    domain = u_h.function_space.mesh
    tdim = domain.topology.dim

    domain.topology.create_entities(0)
    domain.topology.create_connectivity(tdim, 0)
    cell_to_vertex = domain.topology.connectivity(tdim, 0)

    n_cells = domain.topology.index_map(tdim).size_local
    n_vertices = domain.topology.index_map(0).size_local

    if svd is None:
        svd = compute_jacobian_svd(domain)

    lam_arr = svd["lambda"]  # (n_cells, tdim) numpy array

    # Scatter singular values to vertex patches using bincount
    ctv_flat = cell_to_vertex.array                                      # (n_cells*(tdim+1),)
    cell_rep  = np.repeat(np.arange(n_cells, dtype=np.intp), tdim + 1)

    count = np.bincount(ctv_flat, minlength=n_vertices).astype(float)   # cells per vertex

    h_sum = np.zeros((n_vertices, tdim))
    for i in range(tdim):
        h_sum[:, i] = np.bincount(ctv_flat,
                                   weights=lam_arr[cell_rep, i],
                                   minlength=n_vertices)

    return h_sum / count[:, None]  # (n_vertices, tdim)


def adapt_h(
    msh,
    eta_k_i: np.ndarray,
    u_h: fem.Function,
    TOL: float,
    lambda_p: np.ndarray,
    ALPHA: float,
    correction_factor: float,
    sigma_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (h_p, coarsen_any, refine_any).

    coarsen_any[P] is True if at least one direction of vertex P is to be coarsened.
    refine_any[P]  is True if at least one direction of vertex P is to be refined.
    """
    tdim = msh.topology.dim
    gdim = msh.geometry.dim
    n_verts = msh.topology.index_map(0).size_local

    energy_sq = fem.assemble_scalar(form(ufl.inner(ufl.grad(u_h), ufl.grad(u_h)) * ufl.dx))

    msh.topology.create_entities(0)
    msh.topology.create_connectivity(tdim, 0)
    msh.topology.create_connectivity(0, tdim)
    vertex_to_cell = msh.topology.connectivity(0, tdim)

    eta_i_at_P = np.zeros((gdim, n_verts))
    for P in range(n_verts):
        cells_of_P = vertex_to_cell.links(P)
        for i in range(gdim):
            eta_i_at_P[i, P] = np.sum(eta_k_i[i, cells_of_P])

    h_p = np.sqrt(2) * lambda_p.copy()
    
    coarsen_any = np.zeros(n_verts, dtype=bool)
    refine_any  = np.zeros(n_verts, dtype=bool)

    for i_dir in range(gdim):
        coeff = (4.0 * sigma_p) / (3.0 * n_verts)
        lower = coeff * ((1 - ALPHA) ** 2) * (TOL ** 2) * energy_sq
        upper = coeff * ((1 + ALPHA) ** 2) * (TOL ** 2) * energy_sq
        eta_sum = eta_i_at_P[i_dir, :]

        ok_mask = (lower <= eta_sum) & (eta_sum <= upper)
        n_ok = int(np.sum(ok_mask))

        coarsen = lower > eta_sum
        h_p[coarsen, i_dir] = correction_factor * lambda_p[coarsen, i_dir]
        coarsen_any |= coarsen

        refine = eta_sum > upper
        h_p[refine, i_dir] = lambda_p[refine, i_dir] / correction_factor
        refine_any |= refine

        print(
            f"  Dir {i_dir + 1}:  satisfied {n_ok}/{n_verts} "
            f"({100 * n_ok / n_verts:.1f}%),  "
            f"coarsen {int(np.sum(coarsen))},  refine {int(np.sum(refine))}"
        )

    return h_p, coarsen_any, refine_any


def compute_jacobian_svd(msh: mesh.Mesh) -> dict:

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    conn = msh.topology.connectivity(tdim, 0)
    coords = msh.geometry.x[:, :tdim]
    n_cells = msh.topology.index_map(tdim).size_local
    nverts = tdim + 1

    # Vectorized Jacobian construction: gather all cell vertices in one shot
    cell_verts = conn.array.reshape(-1, nverts)              # (n_cells, 4)
    pts = coords[cell_verts]                                  # (n_cells, 4, 3)
    J_all = (pts[:, 1:, :] - pts[:, 0:1, :]).transpose(0, 2, 1)  # (n_cells, 3, 3)

    # Single batched SVD call over the entire mesh
    U_all, Sigma_all, _ = np.linalg.svd(J_all)               # shapes: (n_cells,3,3), (n_cells,3), (n_cells,3,3)

    lam_arr = Sigma_all                                       # (n_cells, tdim)
    r_arr   = U_all.transpose(0, 2, 1)                       # r_arr[c] = U_all[c].T → (n_cells, tdim, tdim)

    return {
        "M_k":    J_all,                    # (n_cells, tdim, tdim)
        "lambda": lam_arr,                  # (n_cells, tdim)
        "r":      r_arr,                    # (n_cells, tdim, tdim)
        "AR":     lam_arr[:, 0] / lam_arr[:, -1],  # (n_cells,)
    }

def compute_anisotropic_eta(u_h: fem.Function, f: ufl.core.expr.Expr, g_N: ufl.core.expr.Expr | None = None) -> np.ndarray:

    G, _ = compute_G_tilde(u_h)
    domain = u_h.function_space.mesh
    tdim = domain.topology.dim

    V0 = fem.functionspace(domain, ("DG", 0))
    v0 = ufl.TestFunction(V0)
    n = ufl.FacetNormal(domain)

    # Pre-evaluate f once at DG0 interpolation points (one per cell, at the
    # barycentre). This replaces expensive per-quadrature-point evaluation of
    # the full UFL expression tree (trig functions, chain-rule products, etc.)
    # with a plain coefficient lookup during assembly.
    # fem.Function is itself a UFL coefficient, so this works whether f is a
    # raw UFL expression OR already a fem.Function of any space.
    f_dg0 = fem.Function(V0)
    f_dg0.interpolate(fem.Expression(f, V0.element.interpolation_points))

    svd     = compute_jacobian_svd(domain)
    lam_arr = svd["lambda"]   # (n_cells, tdim)
    r_arr   = svd["r"]        # (n_cells, tdim, tdim)

    # Residual term — f_dg0 is a simple coefficient lookup, not a trig kernel
    R_K = ufl.div(ufl.grad(u_h)) + f_dg0
    b1 = fem.assemble_vector(form(ufl.inner(R_K, R_K) * v0 * ufl.dx))
    res_norm = np.sqrt(b1.array)

    # Gradient jump term -> internal edges
    jump_n = ufl.jump(ufl.grad(u_h), n)
    b2 = fem.assemble_vector(form(ufl.inner(jump_n, jump_n) * (v0('+') + v0('-')) * ufl.dS))
    jump_norm = np.sqrt(b2.array)
    lam_min = lam_arr[:, -1]
    jump_norm /= (2 * np.sqrt(lam_min))

    # Boundary term -> if it exists
    if g_N is not None:
        neumann_res = ufl.dot(ufl.grad(u_h), n) - g_N
        b3 = fem.assemble_vector(form(ufl.inner(neumann_res, neumann_res) * v0 * ufl.ds))
        bound_norm = np.sqrt(b3.array)
        bound_norm /= (2 * np.sqrt(lam_min))
    else:
        # pure Dirichlet: boundary contribution is exactly zero
        bound_norm = np.zeros_like(b2.array)

    res1 = res_norm + jump_norm + bound_norm

    # Compute Omega_k term
    n_cells = V0.dofmap.index_map.size_local
    omega_sq = np.zeros(n_cells)
    omegas = np.zeros((tdim, n_cells))
    for i in range(tdim):
        q_i = np.zeros(n_cells)
        r_vec = r_arr[:, i, :]
        for j in range(tdim):
            for k in range(j, tdim):
                factor = 1 if j == k else 2
                q_i += factor * r_vec[:, j] * r_vec[:, k] * G[j, k]
        omegas[i] = lam_arr[:, i] * np.sqrt(q_i)
        omega_sq += omegas[i]**2

    omega_tilde = np.sqrt(omega_sq)
    eta_K = res1 * omega_tilde

    return eta_K, res1, omegas
