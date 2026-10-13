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
# u_plotter.show(screenshot=path_pyvista)

iso = u_grid.threshold(0.9, scalars="u")
edges = iso.extract_feature_edges(
    boundary_edges=True, feature_edges=False,
    manifold_edges=False, non_manifold_edges=False
).clean()

from postprocessing_utils import get_ordered_polyline_points

loops = get_ordered_polyline_points(edges, close=True)
if len(loops) == 0:
    raise RuntimeError("No boundary loops found. Try a different threshold.")

outer_loop = loops[0][:, :2]              # largest loop
hole_loops = [ln[:, :2] for ln in loops[1:]]

import matplotlib.pyplot as plt
import numpy as np
x = hole_loops[0][:, 0]
y = hole_loops[0][:, 1]

from scipy.interpolate import splprep, splev


x = outer_loop[:, 0]
y = outer_loop[:, 1]

tck, u = splprep([x, y], s=50, per=True)

u_new = np.linspace(0, 1, 500)
x_smooth, y_smooth = splev(u_new, tck)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b.', alpha=0.1, label='Original Jaggy')
plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'Smoothed Spline (s=50)')
plt.axis("equal")
plt.legend()
plt.show()

import trimesh
from shapely.geometry import Polygon

points_2d = np.column_stack((x_smooth, y_smooth))
poly = Polygon(points_2d)

height = 80.0
mesh = trimesh.creation.extrude_polygon(poly, height=height)

output_file = "printing/splined.obj"
mesh.export(output_file)
mesh.show()
print(f"Success! Extruded mesh saved as {output_file}")


from build123d import *
import numpy as np

# --- 1. Data Cleaning (Crucial for CAD) ---
points_2d = np.column_stack((x_smooth, y_smooth))

# Remove duplicate end point if it exists
if np.allclose(points_2d[0], points_2d[-1]):
    points_2d = points_2d[:-1]

# Subsample to keep the CAD kernel happy (e.g., take every 5th point)
# High-density points from a spline often cause "micro-jags"
points_2d = points_2d[::5]

# Convert to Build123d Vector objects
pts = [Vector(p[0], p[1]) for p in points_2d]

# --- 2. Create the Part ---
try:
    with BuildPart() as foot:
        with BuildSketch() as s:
            # Create a smooth periodic spline edge
            # Build123d's make_spline is very robust for closures
            edge = Edge.make_spline(pts, periodic=True)
            
            # Turn the edge into a closed wire and then a face
            add(edge)
            make_face()
            
        # Extrude the sketch by 80mm along the Z axis
        extrude(amount=80.0)
        
        # --- 3. Robust Filleting ---
        # Selection is easy here: we filter for all edges parallel to the Z axis
        vertical_edges = foot.edges().filter_by(Axis.Z)
        
        try:
            # Start with a conservative fillet (1.0 or 2.0 mm)
            fillet(vertical_edges, radius=2.0)
            print("Fillet applied successfully.")
        except:
            print("Fillet failed: The geometry may have sharp corners or self-intersections.")

    # --- 4. Export to STEP ---
    foot.part.export_step("printing/foot_build123d.step")
    print("Success! STEP file saved as printing/foot_build123d.step")

except Exception as e:
    print(f"Build123d failed: {e}")