import numpy as np
import dolfinx
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import form
from dolfinx.fem.petsc import LinearProblem
import ufl
import basix.ufl
from mpi4py import MPI

def compute_eta_k(u_h: fem.Function, f: ufl.core.expr.Expr, cell_index: int) -> float:
    
    domain = u_h.function_space.mesh
    tdim   = domain.topology.dim
    fdim   = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(fdim, tdim)

    cell_to_facet = domain.topology.connectivity(tdim, fdim)
    facet_to_cell = domain.topology.connectivity(fdim, tdim)

    facets_of_K = cell_to_facet.links(cell_index)

    interior_facets = []
    boundary_facets = []
    for facet in facets_of_K:
        neighbouring_cells = facet_to_cell.links(facet)
        if len(neighbouring_cells) == 2:
            interior_facets.append(facet)
        else:
            boundary_facets.append(facet)
    

    cell_tags = mesh.meshtags(domain, tdim, np.array([cell_index], dtype=np.int32),
                              np.array([1], dtype=np.int32))
    dx_K = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

    h = ufl.CellDiameter(domain)
    n = ufl.FacetNormal(domain)

    interior_res = ufl.div(ufl.grad(u_h)) + f
    res_form = form(h**2 * ufl.inner(interior_res, interior_res) * dx_K(1))
    res = np.sqrt(fem.assemble_scalar(res_form))
    jump_term = 0.0

    if len(interior_facets) > 0:
        int_facet_indices = np.array(interior_facets, dtype=np.int32)
        int_facet_markers = np.ones(len(interior_facets), dtype=np.int32)
        int_facet_tags = mesh.meshtags(domain, fdim, int_facet_indices, int_facet_markers)

        dS_K = ufl.Measure("dS", domain=domain, subdomain_data=int_facet_tags)
        h_K = h('+')
        jump_n = ufl.jump(ufl.grad(u_h), n)

        int_edge_form = form((0.5)**2 * h_K * ufl.inner(jump_n, jump_n) * dS_K(1))
        jump_term += fem.assemble_scalar(int_edge_form)

    if len(boundary_facets) > 0:
        bnd_facet_indices = np.array(boundary_facets, dtype=np.int32)
        bnd_facet_markers = np.full(len(boundary_facets), 2, dtype=np.int32)
        bnd_facet_tags = mesh.meshtags(domain, fdim,
                                          bnd_facet_indices, bnd_facet_markers)
        ds_K = ufl.Measure("ds", domain=domain, subdomain_data=bnd_facet_tags)

        bnd_jump  = 2.0 * ufl.dot(ufl.grad(u_h), n)
        bnd_edge_form = form((0.5)**2 * h * bnd_jump**2 * ds_K(2))
        jump_term += fem.assemble_scalar(bnd_edge_form)

    eta_K = res + np.sqrt(max(jump_term, 0.0))
    return float(eta_K)

def compute_eta(u_h: fem.Function, f: ufl.core.expr.Expr) -> np.ndarray:
    domain = u_h.function_space.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    V0  = fem.functionspace(domain, ("DG", 0))
    v0 = ufl.TestFunction(V0)

    h = ufl.CellDiameter(domain)
    n = ufl.FacetNormal(domain)
    R_K = ufl.div(ufl.grad(u_h)) + f
    L1 = form(h**2 * ufl.inner(R_K, R_K) * v0 * ufl.dx)
    b1 = fem.assemble_vector(L1)
    b1.scatter_reverse(dolfinx.la.InsertMode.add)
    term1 = np.sqrt(np.maximum(b1.array, 0.0))
    
    h_K = h('+')
    jump_n = ufl.jump(ufl.grad(u_h), n)
    L2_int = form(2 * (0.5)**2 * h_K * ufl.inner(jump_n, jump_n) * ufl.avg(v0) * ufl.dS)
    b2 = fem.assemble_vector(L2_int)
    b2.scatter_reverse(dolfinx.la.InsertMode.add)

    bnd_jump = 2.0 * ufl.dot(ufl.grad(u_h), n)
    L2_bnd = form((0.5)**2 * h * bnd_jump**2 * v0 * ufl.ds)
    b3 = fem.assemble_vector(L2_bnd)
    b3.scatter_reverse(dolfinx.la.InsertMode.add)

    n_owned = V0.dofmap.index_map.size_local
    term1   = np.sqrt(np.maximum(b1.array[:n_owned], 0.0))
    term2   = np.sqrt(np.maximum(b2.array[:n_owned] + b3.array[:n_owned], 0.0))
    eta_local = term1 + term2
    #term2 = np.sqrt(np.maximum(b2.array + b3.array, 0.0))
    #eta_K = term1 + term2

    return eta_local
    
def compute_gradient_dg0(u_h: fem.Function):
    
    domain = u_h.function_space.mesh
    gdim   = domain.geometry.dim
    
    dg0_element  = basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(gdim,))
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

    n_cells_owned = V0.dofmap.index_map.size_local
    n_cells_ghost = V0.dofmap.index_map.num_ghosts
    n_cells_local = n_cells_owned + n_cells_ghost

    one = fem.Constant(domain, default_scalar_type(1.0))
    b_vol = fem.assemble_vector(form(one * v0 * ufl.dx))
    b_vol.scatter_reverse(dolfinx.la.InsertMode.add)
    vol_fn = fem.Function(V0)
    vol_fn.x.array[:] = b_vol.array[:]
    vol_fn.x.scatter_forward()
    vol = vol_fn.x.array 
    
    grad_dg0 = compute_gradient_dg0(u_h)
    
    V_cg = fem.functionspace(domain, ("Lagrange", 1))
    n_dofs_owned = V_cg.dofmap.index_map.size_local
    n_dofs_ghost = V_cg.dofmap.index_map.num_ghosts
    n_dofs_total = n_dofs_owned + n_dofs_ghost

    n_cells_owned = V0.dofmap.index_map.size_local
    n_cells_total = n_cells_owned + V0.dofmap.index_map.num_ghosts
    
    vertex_to_dof = {}    
    for c in range(n_cells_local):
        verts = cell_to_vertex.links(c)
        dofs  = V_cg.dofmap.cell_dofs(c)
        for v, d in zip(verts, dofs):
            vertex_to_dof[int(v)] = int(d)
            
    grad_arr = grad_dg0.x.array[:n_cells_total * gdim].reshape(n_cells_total, gdim)
    Pi_funcs = []    

    for i in range(gdim):
        num = np.zeros(n_dofs_total)   
        den = np.zeros(n_dofs_total)

        for v, dof in vertex_to_dof.items():
            patch_cells = vertex_to_cell.links(v)
            vols_patch  = vol[patch_cells]
            grads_patch = grad_arr[patch_cells, i]
            num[dof] += np.dot(vols_patch, grads_patch)
            den[dof] += np.sum(vols_patch)
    
        Pi_gi = fem.Function(V_cg)
        arr = Pi_gi.x.array
        arr[:n_dofs_owned] = (num[:n_dofs_owned]/den[:n_dofs_owned])
        #owned_mask = den[:n_dofs_owned] > 0.0
        #arr[:n_dofs_owned][owned_mask] = (num[:n_dofs_owned][owned_mask] / den[:n_dofs_owned][owned_mask])
        Pi_gi.x.scatter_forward()
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
    n_owned = V0.dofmap.index_map.size_local
    
    G = {}
    for i in range(gdim):
        for j in range(i, gdim):
            b = fem.assemble_vector(form(etas[i] * etas[j] * v0 * ufl.dx))
            b.scatter_reverse(dolfinx.la.InsertMode.add)
            # store only owned cell values, discard ghost padding
            G[(i, j)] = b.array[:n_owned].copy()
    return G

def get_G_matrix(G: dict[tuple[int,int], np.ndarray], K: int, gdim: int) -> np.ndarray:
    mat = np.zeros((gdim, gdim))
    for i in range(gdim):
        for j in range(i, gdim):
            mat[i, j] = G[(i, j)][K]
            mat[j, i] = G[(i, j)][K]
    return mat
    
        
# Simple Poisson same as in tutorial
domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, mesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("Lagrange", 1))

uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, default_scalar_type(-6))
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Poisson",
)
uh = problem.solve()

n_dofs_owned = V.dofmap.index_map.size_local
# Print for compute_eta
all_eta = compute_eta(uh, f)
local_sum_sq  = np.sum(all_eta**2)
global_sum_sq = domain.comm.allreduce(local_sum_sq, op=MPI.SUM)

# Test to see if ZZ function is Ok
Pi = compute_zz_grad(uh)
Pi_gx, Pi_gy = Pi
print(f"Rank {domain.comm.rank} Pi_gx owned values: {Pi_gx.x.array[:n_dofs_owned]}")
print(f"Rank {domain.comm.rank} Pi_gy owned values: {Pi_gy.x.array[:n_dofs_owned]}")

G = compute_G_tilde(uh)
mat = get_G_matrix(G, 0, gdim=2)
#print(mat)
#if domain.comm.rank == 0:
    #print(f"Global estimator = {np.sqrt(global_sum_sq):.6e}")

# Each rank prints its own owned cells (no zeros, correct lengths)
#print(f"Rank {domain.comm.rank}: {all_eta}")

# Print for compute_eta_k
#eta_42 = compute_eta_k(uh, f, cell_index=5)
##print(f"eta_5 = {eta_42:.6e}")
#
#n_cells = domain.topology.index_map(domain.topology.dim).size_local
#etas    = np.array([compute_eta_k(uh, f, k) for k in range(n_cells)])
#
#print(f"Global estimator (H1) = {np.sqrt(np.sum(etas**2)):.6e}")
#print(f"Max eta_K             = {etas.max():.6e}  at cell {np.argmax(etas)}")
#print(f"Min eta_K             = {etas.min():.6e}  at cell {np.argmin(etas)}")
