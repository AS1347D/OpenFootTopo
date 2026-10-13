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

import numpy as np
from rdp import rdp
from scipy.interpolate import splprep, splev
from sklearn.neighbors import NearestNeighbors

def process_multiple_loops(isoline_mesh, epsilon=0.01):
    # 1. Separate the mesh into disconnected components (the "holes")
    # This adds a "RegionId" array to the points
    separated = isoline_mesh.connectivity(label_mode='all')
    region_ids = separated.point_data["RegionId"]
    num_regions = len(np.unique(region_ids))
    
    all_splines = []

    for i in range(num_regions):
        # Extract just this specific loop/hole
        indices = np.where(region_ids == i)[0]
        region_points = separated.points[indices]
        
        if len(region_points) < 4:  # Skip tiny noise artifacts
            continue
            
        # 2. Sort points for this specific loop
        sorted_coords = sort_points(region_points)
        
        # 3. RDP Simplification (with the list-to-array fix)
        rdp_points = np.array(rdp(sorted_coords, epsilon=epsilon))
        
        # 4. Fit B-Spline (Degree 2)
        # We check if it's a closed loop by comparing start and end points
        is_closed = np.linalg.norm(rdp_points[0] - rdp_points[-1]) < epsilon * 2
        
        try:
            # per=is_closed ensures the spline closes smoothly without a kink
            tck, u = splprep([rdp_points[:, 0], rdp_points[:, 1], rdp_points[:, 2]], 
                             k=2, s=0, per=is_closed)
            
            u_fine = np.linspace(0, 1, 500)
            smooth_coords = np.vstack(splev(u_fine, tck)).T
            all_splines.append(smooth_coords)
        except Exception as e:
            print(f"Skipping region {i} due to spline error: {e}")
            
    return all_splines

def sort_points(pts):
    """Sorts points into a sequence using nearest neighbors."""
    clf = NearestNeighbors(n_neighbors=2).fit(pts)
    unused = list(range(1, len(pts)))
    sorted_idx = [0]
    curr = 0
    while unused:
        dists = np.linalg.norm(pts[unused] - pts[curr], axis=1)
        closest = np.argmin(dists)
        curr = unused.pop(closest)
        sorted_idx.append(curr)
    return pts[sorted_idx]

# --- Usage ---
# isoline = grid.contour([0.5], scalars="rho")
spline_loops = process_multiple_loops(edges, epsilon=0.02)
print(len(spline_loops))
# Visualization
plotter = pv.Plotter()
for loop in spline_loops:
    plotter.add_mesh(pv.MultipleLines(loop), color="cyan", line_width=4)
plotter.show()
plotter.screenshot(path_pyvista)

# --- Visualization with PyVista ---
# import pyvista as pv
# plotter = pv.Plotter()
# plotter.add_mesh(pv.MultipleLines(coords), color="red", label="Original Zigzag")
# plotter.add_mesh(pv.MultipleLines(smooth_line), color="cyan", line_width=5, label="RDP + CG2 Spline")
# plotter.add_legend()
# plotter.show()
