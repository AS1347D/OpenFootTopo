# adios_read.py
from dolfinx.fem import (Function, functionspace)
from dolfinx import plot
from mpi4py import MPI
import adios4dolfinx
import dolfinx
from pathlib import Path
import pyvista as pv
import numpy as np

path_out = Path("adios_dir/checkpoint.bp")
path_pyvista = Path("adios_dir/isoline.png")

domain = adios4dolfinx.read_mesh(path_out, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

S = functionspace(domain, ("CG", 1))

rho_import = Function(S)

adios4dolfinx.read_function(path_out, rho_import, engine="BP4")

S_cg2 = functionspace(domain, ("CG", 2))
rho_cg2 = Function(S_cg2)
rho_cg2.interpolate(rho_import)

# Create plotter and pyvista grid
import pyvista
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, tdim)
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(S_cg2)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = rho_cg2.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter(off_screen=False)
u_plotter.add_mesh(u_grid, cmap="jet", show_edges=False)
u_plotter.view_xy()
u_plotter.show()
u_plotter.screenshot(path_pyvista)

iso = u_grid.threshold(0.9, scalars="u")
edges = iso.extract_feature_edges(
    boundary_edges=True, feature_edges=False,
    manifold_edges=False, non_manifold_edges=False
).clean()

coords = edges.points
# step = 6
# coords = coords[::step]

import matplotlib.pyplot as plt
import numpy as np
x = coords[:, 0]
y = coords[:, 1]

plt.figure()
plt.scatter(x, y)
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.show()