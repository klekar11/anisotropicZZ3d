### Mesh functions — 3D version ###
# Adapted from the 2D mesh_functions.py for tetrahedral meshes.


import numpy as np
import dolfinx
from numba import njit
from dolfinx import fem


def get_nodes_to_elem(msh):
    '''
    For each mesh vertex, the tetrahedra that contain it.
    '''
    msh.topology.create_connectivity(0, 3)
    return msh.topology.connectivity(0, 3)


def get_nodes_to_elem_array(msh):

    Nnodes = msh.geometry.x.shape[0]
    n2e = get_nodes_to_elem(msh)
    max_valence = max(len(n2e.links(i)) for i in range(Nnodes))
    arr = np.zeros((Nnodes, 1 + max_valence), dtype=np.int64)
    for i in range(Nnodes):
        links = n2e.links(i)
        arr[i, 0] = len(links)
        arr[i, 1:1 + len(links)] = links
    return arr


@njit(cache=True)
def vertex_neighbors(elems: np.ndarray, nodes_to_elem_array: np.ndarray, i: int):
    '''
    All vertices sharing a tetrahedron with vertex i (including i itself).
    '''
    N = nodes_to_elem_array[i, 0]
    cells = nodes_to_elem_array[i, 1:N + 1]
    return np.unique(elems[cells])


def find_boundary(msh):
    '''
    Boolean array of length n_vertices, True on boundary vertices.
    '''
    V = fem.functionspace(msh, ("CG", 1))
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    boundary = np.zeros(msh.geometry.x.shape[0], dtype=bool)
    boundary[boundary_dofs] = True
    return boundary