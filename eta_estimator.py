import numpy as np
import dolfinx
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import form
from dolfinx.fem.petsc import LinearProblem
import ufl
import basix.ufl

def compute_iso_eta(u_h: fem.Function, f: ufl.core.expr.Expr, g_N: ufl.core.expr.Expr | None = None) -> np.ndarray:
    
    domain = u_h.function_space.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    V0 = fem.functionspace(domain, ("DG", 0))
    v0 = ufl.TestFunction(V0)
    h = ufl.CellDiameter(domain)
    n = ufl.FacetNormal(domain)

    R_K = ufl.div(ufl.grad(u_h)) + f
    b1 = fem.assemble_vector(form(h**2 * ufl.inner(R_K, R_K) * v0 * ufl.dx))

    h_K = h('+')
    jump_n = ufl.jump(ufl.grad(u_h), n)
    b2 = fem.assemble_vector(
        form(0.25 * h_K * ufl.inner(jump_n, jump_n) * (v0('+') + v0('-')) * ufl.dS)
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
            num[dof]   += np.dot(vols_patch, grads_patch)
            den[dof]   += np.sum(vols_patch)
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

def compute_G_tilde(u_h: fem.Function):

    domain = u_h.function_space.mesh
    gdim = domain.geometry.dim
    etas = compute_eta_zz(u_h)
    V0 = fem.functionspace(domain, ("DG", 0))
    v0 = ufl.TestFunction(V0)
    n_cells = V0.dofmap.index_map.size_local
    
    G = {}
    for i in range(gdim):
        for j in range(i, gdim):
            b = fem.assemble_vector(form(etas[i] * etas[j] * v0 * ufl.dx))
            G[(i, j)] = b.array[:n_cells].copy()
            
    return G

def get_G_matrix(G: dict[tuple[int,int], np.ndarray], K: int, gdim: int) -> np.ndarray:

    mat = np.zeros((gdim, gdim))
    for i in range(gdim):
        for j in range(i, gdim):
            mat[i, j] = G[(i, j)][K]
            mat[j, i] = G[(i, j)][K]
            
    return mat

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

    G = compute_G_tilde(u_h)
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
    for i in range(tdim):
        q_i = np.zeros(n_cells)
        r_vec = r_arr[:, i, :]
        for j in range(tdim):
            for k in range(j, tdim):
                factor = 1 if j == k else 2
                q_i += factor * r_vec[:, j] * r_vec[:, k] * G[(j, k)]
        omega_sq += lam_arr[:, i]**2 * q_i

    omega_tilde = np.sqrt(omega_sq)

    return res1 * omega_tilde