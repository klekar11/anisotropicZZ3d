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
    n_dofs = V_cg.dofmap.index_map.size_local
    
    vertex_to_dof = {}
    for c in range(n_cells):
        verts = cell_to_vertex.links(c)
        dofs = V_cg.dofmap.cell_dofs(c)
        for v, d in zip(verts, dofs):
            vertex_to_dof[int(v)] = int(d)
            
    grad_arr = grad_dg0.x.array[:n_cells * gdim].reshape(n_cells, gdim)
    Pi_funcs = []
    for i in range(gdim):
        num = np.zeros(n_dofs)
        den = np.zeros(n_dofs)
        for v, dof in vertex_to_dof.items():
            patch_cells = vertex_to_cell.links(v)
            vols_patch = vol[patch_cells]
            grads_patch = grad_arr[patch_cells, i]
            num[dof] += np.dot(vols_patch, grads_patch)
            den[dof] += np.sum(vols_patch)
        Pi_gi = fem.Function(V_cg)
        Pi_gi.x.array[:] = num / den
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
            G[(i, j)] = b.array[:n_cells].copy()

    # Build a storable DG0 vector Function for eta_zz
    dg0_vec_el = basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(gdim,))
    V_dg0_vec = fem.functionspace(domain, dg0_vec_el)
    eta_zz_fn = fem.Function(V_dg0_vec, name="eta_zz")
    eta_zz_expr = ufl.as_vector([etas[i] for i in range(gdim)])
    eta_zz_fn.interpolate(fem.Expression(eta_zz_expr, V_dg0_vec.element.interpolation_points))

    return G, eta_zz_fn

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

    vertex_sums = np.zeros((n_vertices, gdim, gdim))
    for cell in range(n_cells):
        G_k = get_G_matrix(G, cell, gdim)
        for vertex in cell_to_vertex.links(cell):
            vertex_sums[vertex] += G_k

    # Eigenvectors of each vertex patch G-tilde sum; columns of Q[p] are eigenvectors
    Q = np.zeros((n_vertices, gdim, gdim))
    for p in range(n_vertices):
        _, Q[p] = np.linalg.eigh(vertex_sums[p])

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

    numerator = np.zeros(n_vertices)
    denominator = np.zeros(n_vertices)

    for K in range(n_cells):
        eta_i_K = eta_k_i[:, K]                    # shape (tdim,)
        num_contrib = np.sum(eta_i_K ** 2)
        den_contrib = np.sqrt(np.sum(eta_i_K ** 4))
        for P in cell_to_vertex.links(K):
            numerator[P]   += num_contrib
            denominator[P] += den_contrib

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

    lam = svd["lambda"]
    lam_arr = np.stack([lam[i].x.array for i in range(tdim)], axis=1)  # (n_cells, tdim)

    h_sum = np.zeros((n_vertices, tdim))
    count = np.zeros(n_vertices, dtype=int)

    for K in range(n_cells):
        for P in cell_to_vertex.links(K):
            h_sum[P] += lam_arr[K]
            count[P] += 1

    return h_sum / count[:, None]  # shape (n_vertices, tdim)


def adapt_h(
    msh,
    eta_k_i: np.ndarray,
    u_h: fem.Function,
    TOL: float,
    lambda_p: np.ndarray,
    ALPHA: float,
    correction_factor: float,
    sigma_p: np.ndarray,
) -> np.ndarray:

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

    h_p = lambda_p.copy()

    for i_dir in range(gdim):
        coeff = (4.0 * sigma_p) / (3.0 * n_verts)
        lower = coeff * ((1 - ALPHA) ** 2) * (TOL ** 2) * energy_sq
        upper = coeff * ((1 + ALPHA) ** 2) * (TOL ** 2) * energy_sq
        eta_sum = eta_i_at_P[i_dir, :]

        ok_mask = (lower <= eta_sum) & (eta_sum <= upper)
        n_ok = int(np.sum(ok_mask))

        coarsen = lower > eta_sum
        h_p[coarsen, i_dir] = correction_factor * lambda_p[coarsen, i_dir]

        refine = eta_sum > upper
        h_p[refine, i_dir] = lambda_p[refine, i_dir] / correction_factor

        print(
            f"  Dir {i_dir + 1}:  satisfied {n_ok}/{n_verts} "
            f"({100 * n_ok / n_verts:.1f}%),  "
            f"coarsen {int(np.sum(coarsen))},  refine {int(np.sum(refine))}"
        )

    return h_p


def compute_jacobian_svd(msh: mesh.Mesh) -> dict:

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    conn = msh.topology.connectivity(tdim, 0)
    coords = msh.geometry.x[:, :tdim]
    n_cells = msh.topology.index_map(tdim).size_local
    nverts = tdim + 1

    cell_name = msh.topology.cell_name()
    el = basix.ufl.element("Lagrange", cell_name, 1)
    ref_centre = np.full((1, tdim), 1 / (tdim + 1))
    dphidX = el.tabulate(1, ref_centre)[1:].reshape(tdim, nverts)

    # Build elements and function spaces
    scalar_el = basix.ufl.element("DG", cell_name, 0)
    vector_el = basix.ufl.element("DG", cell_name, 0, shape = (tdim,))
    tensor_el = basix.ufl.element("DG", cell_name, 0, shape = (tdim, tdim))

    V_s = fem.functionspace(msh, scalar_el)
    V_v = fem.functionspace(msh, vector_el)
    V_t = fem.functionspace(msh, tensor_el)

    M_k = fem.Function(V_t, name = "M_k")
    lam = [fem.Function(V_s, name = f"lambda_{i}") for i in range(tdim)]
    r = [fem.Function(V_v, name = f"r_{i}") for i in range(tdim)]
    AR = fem.Function(V_s, name = "AR")

    J_arr = np.empty((n_cells, tdim, tdim))
    lam_arr = np.empty((n_cells, tdim))
    r_arr = np.empty((n_cells, tdim, tdim))

    for c in range(n_cells):

        pts = coords[conn.links(c)]
        J = (pts[1:] - pts[0]).T
        J_arr[c] = J
        U, Sigma, _ = np.linalg.svd(J)
        lam_arr[c] = Sigma
        r_arr[c] = U.T

    AR_arr = lam_arr[:, 0] / lam_arr[:, -1]

    M_k.x.array[:] = J_arr.ravel()
    for i in range(tdim):
        lam[i].x.array[:] = lam_arr[:, i]
        r[i].x.array[:] = r_arr[:, i, :].ravel()
    AR.x.array[:] = AR_arr

    return {"M_k": M_k, "lambda": lam, "r": r, "AR": AR}

def compute_anisotropic_eta(u_h: fem.Function, f: ufl.core.expr.Expr, g_N: ufl.core.expr.Expr | None = None) -> np.ndarray:

    G, _ = compute_G_tilde(u_h)
    domain = u_h.function_space.mesh
    tdim = domain.topology.dim

    V0 = fem.functionspace(domain, ("DG", 0))
    v0 = ufl.TestFunction(V0)
    n = ufl.FacetNormal(domain)
                                                   
    
    svd = compute_jacobian_svd(domain)
    lam = svd["lambda"]
    r_vecs = svd["r"]

    # Residual term
    R_K = ufl.div(ufl.grad(u_h)) + f                                                  
    b1 = fem.assemble_vector(form(ufl.inner(R_K, R_K) * v0 * ufl.dx))
    res_norm = np.sqrt(b1.array)
    
    lam_arr = np.stack([lam[i].x.array for i in range(tdim)], axis=1)
    r_arr = np.stack([r_vecs[i].x.array.reshape(-1, tdim) for i in range(tdim)], axis=1)

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