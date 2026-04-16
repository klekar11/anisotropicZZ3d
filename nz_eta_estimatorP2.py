### Naga-Zhang / PPR post-processing for P2 finite elements in 3D ###
# Adapted from the 2D version.


import numpy as np
from numba import njit, prange

import dolfinx
from dolfinx import fem

from nz_mesh_helpers import (
    get_nodes_to_elem_array,
    vertex_neighbors,
    find_boundary,
)


# Minimum number of DOFs we want in a patch before trusting the fit.
# Cubic basis in 3D has 20 monomials; 2x oversampling is a common rule.
N_MIN_PATCH = 40

# Tolerance below which a directional extent h_k is considered degenerate
# (patch is flat in direction k, so we need to expand more).
H_DIR_TOL = 1e-14


def vertex_to_dof_map(mesh, V):
    '''
    Map from vertex index to the DOF index of V sitting on that vertex.
    Works for any tdim because it uses dof_layout.entity_dofs(0, i), which
    is the local vertex -> local DOF mapping on the reference cell.
    Unchanged from the 2D code.
    '''
    num_vertices_per_cell = dolfinx.cpp.mesh.cell_num_entities(
        mesh.topology.cell_type, 0
    )
    dof_layout2 = np.empty((num_vertices_per_cell,), dtype=np.int32)
    for i in range(num_vertices_per_cell):
        var = V.dofmap.dof_layout.entity_dofs(0, i)
        assert len(var) == 1
        dof_layout2[i] = var[0]

    num_vertices = (
        mesh.topology.index_map(0).size_local
        + mesh.topology.index_map(0).num_ghosts
    )
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    assert (
        c_to_v.offsets[1:] - c_to_v.offsets[:-1] == c_to_v.offsets[1]
    ).all(), "Single cell type supported"

    v2d = np.empty(num_vertices, dtype=np.int32)
    v2d[c_to_v.array] = V.dofmap.list[:, dof_layout2].reshape(-1)
    return v2d


# --- Find DOFs on edges emanating from vertex i -----------------------------
@njit(cache=True)
def find_closest_dofs(nodes, elems, nodes_to_elem_array, dofs, i):
    '''
    For vertex i, find all DOFs sitting at the midpoint of an edge (i, j)
    where j is a vertex connected to i. These are the P2 mid-edge DOFs
    that need to be updated by the cubic fit around i.
    
    In 3D this typically finds up to ~12 DOFs (average degree in a
    tetrahedral mesh), but the loop is dimension-agnostic.
    '''
    cn = vertex_neighbors(elems, nodes_to_elem_array, i)
    cn = cn[cn != i]
    mid_edges = (nodes[cn] + nodes[i]) / 2.0  # shape (n_neigh, 3)
    
    closest_dofs = []
    for k in range(cn.shape[0]):
        for j in range(dofs.shape[0]):
            dx = mid_edges[k, 0] - dofs[j, 0]
            dy = mid_edges[k, 1] - dofs[j, 1]
            dz = mid_edges[k, 2] - dofs[j, 2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < 1e-14:
                closest_dofs.append(j)
    return np.array(closest_dofs)


# --- Boundary patch expansion -----------------------------------------------
@njit(cache=True)
def process_boundary_3d(elems, nodes_to_elem_array, j, B, on_boundary,
                        dofs, dofmap_list, node_i):
    '''
    Expand the vertex patch j until three conditions are met:
      (a) at least one vertex in j is interior (B[v] = False),
      (b) the collected DOF patch has at least N_MIN_PATCH DOFs,
      (c) the patch has nonzero extent in all 3 coordinate directions
          relative to node_i.
    
    Condition (a) is the 2D criterion. (b) guards against under-determined
    least squares (cubic basis has 20 monomials). (c) guards against a
    "flat" patch (e.g. a vertex on a face with only one layer of neighbours
    all in the same plane) which would make h_k = 0 in that direction.
    
    We also cap the number of expansion rounds to prevent runaway patches.
    '''
    MAX_ROUNDS = 6
    
    for _ in range(MAX_ROUNDS):
        # (a) Interior vertex reached?
        has_interior = False
        for v in j:
            if not B[v]:
                has_interior = True
                break
        
        # (b) Build the candidate DOF patch to count DOFs.
        L = _patch(nodes_to_elem_array, dofmap_list, j)
        enough_dofs = (L.shape[0] >= N_MIN_PATCH)
        
        # (c) Check extents in each direction.
        # We inline the extent computation to stay inside numba.
        full_extent = True
        if enough_dofs:
            hx = 0.0
            hy = 0.0
            hz = 0.0
            for m in range(L.shape[0]):
                dx = abs(dofs[L[m], 0] - node_i[0])
                dy = abs(dofs[L[m], 1] - node_i[1])
                dz = abs(dofs[L[m], 2] - node_i[2])
                if dx > hx: hx = dx
                if dy > hy: hy = dy
                if dz > hz: hz = dz
            if hx < H_DIR_TOL or hy < H_DIR_TOL or hz < H_DIR_TOL:
                full_extent = False
        
        if has_interior and enough_dofs and full_extent:
            return j
        
        # Expand patch: add all neighbours of current patch vertices.
        j_new = j.copy()
        for k in j:
            j_new = np.union1d(j_new, vertex_neighbors(elems, nodes_to_elem_array, k))
        j = j_new
    
    return j


@njit(cache=True)
def _patch(nodes_to_elem_array, dofmap_list, j):
    '''Collect all DOFs of all cells touching any vertex in j.'''
    l = np.zeros(0, dtype=np.int64)
    for k in j:
        N = nodes_to_elem_array[k, 0]
        cells = nodes_to_elem_array[k, 1:N + 1]
        l = np.union1d(l, np.unique(dofmap_list[cells].ravel()))
    return l.astype(np.int64)


def patch(nodes_to_elem_array, dofmap_list, j):
    '''Python-visible wrapper for _patch.'''
    return _patch(nodes_to_elem_array, dofmap_list, j)


# --- Build the 20-column Vandermonde for complete P3 in 3D ------------------
@njit(cache=True)
def _build_vandermonde(p):

    n = p.shape[0]
    A = np.empty((n, 20))
    for k in range(n):
        x = p[k, 0]
        y = p[k, 1]
        z = p[k, 2]
        A[k, 0] = 1.0
        A[k, 1] = x
        A[k, 2] = y
        A[k, 3] = z
        A[k, 4] = x * x
        A[k, 5] = x * y
        A[k, 6] = x * z
        A[k, 7] = y * y
        A[k, 8] = y * z
        A[k, 9] = z * z
        A[k, 10] = x * x * x
        A[k, 11] = x * x * y
        A[k, 12] = x * x * z
        A[k, 13] = x * y * y
        A[k, 14] = x * y * z
        A[k, 15] = x * z * z
        A[k, 16] = y * y * y
        A[k, 17] = y * y * z
        A[k, 18] = y * z * z
        A[k, 19] = z * z * z
    return A


@njit(cache=True)
def _eval_grad(a, x, y, z):
    '''
    Analytical gradient of the cubic p(x,y,z) = sum_i a_i * m_i(x,y,z),
    using the same monomial ordering as _build_vandermonde.
    Returns (dp/dx, dp/dy, dp/dz) in the rescaled local coordinates.
    '''
    # d/dx
    gx = (a[1]
          + 2.0 * a[4] * x + a[5] * y + a[6] * z
          + 3.0 * a[10] * x * x + 2.0 * a[11] * x * y + 2.0 * a[12] * x * z
          + a[13] * y * y + a[14] * y * z + a[15] * z * z)
    # d/dy
    gy = (a[2]
          + a[5] * x + 2.0 * a[7] * y + a[8] * z
          + a[11] * x * x + 2.0 * a[13] * x * y + a[14] * x * z
          + 3.0 * a[16] * y * y + 2.0 * a[17] * y * z + a[18] * z * z)
    # d/dz
    gz = (a[3]
          + a[6] * x + a[8] * y + 2.0 * a[9] * z
          + a[12] * x * x + a[14] * x * y + 2.0 * a[15] * x * z
          + a[17] * y * y + 2.0 * a[18] * y * z + 3.0 * a[19] * z * z)
    return gx, gy, gz


# --- Public driver ----------------------------------------------------------
def Gh(uh):
    '''
    Naga-Zhang / PPR recovered gradient for a P2 FE solution on a
    tetrahedral mesh. Returns a fem.Function in P2 vector space of dim 3.
    '''
    V = uh.function_space
    mesh = V.mesh
    assert mesh.topology.dim == 3, "This module is for 3D meshes."
    
    dofs = V.tabulate_dof_coordinates()[:, :3]
    v2d = vertex_to_dof_map(mesh, V)
    
    nodes = mesh.geometry.x[:, :3]
    conn = mesh.topology.connectivity(mesh.topology.dim, 0)
    elems = conn.array.reshape((-1, 4))  # tetrahedra: 4 vertices each
    n2e_arr = get_nodes_to_elem_array(mesh)
    boundary = find_boundary(mesh)
    
    Ghuh_dofs = _Gh(
        nodes, elems, dofs, uh.x.array,
        n2e_arr, boundary, v2d,
        V.dofmap.list,
    )
    
    V2 = fem.functionspace(mesh, ("Lagrange", 2, (3,)))
    Ghuh = fem.Function(V2)
    Ghuh.x.array[:] = Ghuh_dofs.flatten()
    return Ghuh


# --- The parallel core ------------------------------------------------------
@njit(parallel=True, cache=True)
def _Gh(nodes, elems, dofs, uh, nodes_to_elem_array, B, v2d, dofmap_list):
    '''
    Parallel evaluation of the recovered gradient at every P2 DOF.
    
    Key differences from 2D:
      - per-direction rescaling h_x, h_y, h_z
      - 20-column Vandermonde
      - gradient evaluated and divided component-wise
    '''
    Nnodes = nodes.shape[0]
    Ndofs = dofs.shape[0]
    Ghuh = np.zeros((Ndofs, 3))
    
    for i in prange(Nnodes):
        on_boundary = B[i]
        node_i = nodes[i]
        
        # Build the vertex patch around i (with expansion near boundaries).
        j = np.array([i])
        j = process_boundary_3d(
            elems, nodes_to_elem_array, j, B, on_boundary,
            dofs, dofmap_list, node_i,
        )
        # For interior vertices we still want a patch with enough DOFs.
        # The first test inside process_boundary_3d already handles this
        # because has_interior is True immediately; but if the 1-ring is
        # too small for a cubic fit, we expand one more time.
        L = _patch(nodes_to_elem_array, dofmap_list, j)
        if L.shape[0] < N_MIN_PATCH:
            j_new = j.copy()
            for k in j:
                j_new = np.union1d(j_new, vertex_neighbors(elems, nodes_to_elem_array, k))
            j = j_new
            L = _patch(nodes_to_elem_array, dofmap_list, j)
        
        # Per-direction patch extents (never zero here thanks to the
        # boundary expansion loop, but we guard anyway).
        hx = 0.0
        hy = 0.0
        hz = 0.0
        for m in range(L.shape[0]):
            dx = abs(dofs[L[m], 0] - node_i[0])
            dy = abs(dofs[L[m], 1] - node_i[1])
            dz = abs(dofs[L[m], 2] - node_i[2])
            if dx > hx: hx = dx
            if dy > hy: hy = dy
            if dz > hz: hz = dz
        # Floor to avoid division by zero on ill-posed patches.
        if hx < H_DIR_TOL: hx = 1.0
        if hy < H_DIR_TOL: hy = 1.0
        if hz < H_DIR_TOL: hz = 1.0
        
        # Rescaled local coordinates of the patch DOFs.
        p = np.empty((L.shape[0], 3))
        for m in range(L.shape[0]):
            p[m, 0] = (dofs[L[m], 0] - node_i[0]) / hx
            p[m, 1] = (dofs[L[m], 1] - node_i[1]) / hy
            p[m, 2] = (dofs[L[m], 2] - node_i[2]) / hz
        
        A = _build_vandermonde(p)
        b = np.empty(L.shape[0])
        for m in range(L.shape[0]):
            b[m] = uh[L[m]]
        
        # Normal equations. A^T A is 20x20 — tiny.
        AtA = A.T @ A
        Atb = A.T @ b
        a = np.linalg.solve(AtA, Atb)
        
        # Vertex value: gradient of p at the local origin is (a1, a2, a3).
        k_vertex = v2d[i]
        Ghuh[k_vertex, 0] = a[1] / hx
        Ghuh[k_vertex, 1] = a[2] / hy
        Ghuh[k_vertex, 2] = a[3] / hz
        
        # Mid-edge DOFs connected to i: accumulate with weight 1/2.
        # The other 1/2 comes from the other endpoint when prange visits it.
        L_edge = find_closest_dofs(nodes, elems, nodes_to_elem_array, dofs, i)
        
        Ghuh_temp = np.zeros((Ndofs, 3))
        for idx in range(L_edge.shape[0]):
            l = L_edge[idx]
            x = (dofs[l, 0] - node_i[0]) / hx
            y = (dofs[l, 1] - node_i[1]) / hy
            z = (dofs[l, 2] - node_i[2]) / hz
            gx, gy, gz = _eval_grad(a, x, y, z)
            Ghuh_temp[l, 0] += 0.5 * gx / hx
            Ghuh_temp[l, 1] += 0.5 * gy / hy
            Ghuh_temp[l, 2] += 0.5 * gz / hz
        
        Ghuh += Ghuh_temp
    
    return Ghuh