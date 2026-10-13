# adios_read.py
from dolfinx.fem import (Function, functionspace)
from dolfinx import plot
from mpi4py import MPI
from postprocessing_utils import get_ordered_polyline_points, footMesh
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from pathlib import Path
import adios4dolfinx
import numpy as np
import dolfinx
import trimesh
import pyvista
import sys


path_out = Path("adios_dir/checkpoint.bp")
path_pyvista = Path("adios_dir_smoothing/isoline.png")

domain = adios4dolfinx.read_mesh(path_out, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

S = functionspace(domain, ("CG", 1))

rho_import = Function(S)

adios4dolfinx.read_function(path_out, rho_import, engine="BP4")

# Create plotter and pyvista grid
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, tdim)
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(S)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = rho_import.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter(off_screen=False)
u_plotter.add_mesh(u_grid, cmap="jet", show_edges=False)
u_plotter.view_xy()
if "-nopopup" not in sys.argv:
    u_plotter.show(screenshot=path_pyvista)

rho_threshold = 0.5 

iso = u_grid.threshold(rho_threshold, scalars="u")
edges = iso.extract_feature_edges(
    boundary_edges=True, feature_edges=False,
    manifold_edges=False, non_manifold_edges=False
).clean()

## Get coordinates of the extracted edges 
loops = get_ordered_polyline_points(edges, close=True)

if len(loops) == 0:
    raise RuntimeError("No boundary loops found. Try a different threshold.")

outer_loop = loops[0][:, :2] # largest loop

hole_loops = [ln[:, :2] for ln in loops[1:]]

## Jaggy plot

plt.figure(figsize=(8, 6))

# Plot Raw Outer Loop
plt.plot(outer_loop[:, 0], outer_loop[:, 1], 'k-', drawstyle='steps-post', linewidth=1.5, label='Raw Outer Boundary')
plt.fill(outer_loop[:, 0], outer_loop[:, 1], color='gray', alpha=0.2)

# Plot Raw Hole Loops
for i, hole in enumerate(hole_loops):
    label = "Raw Holes" if i == 0 else None
    plt.plot(hole[:, 0], hole[:, 1], 'b-', drawstyle='steps-post', linewidth=1, label=label)
    plt.fill(hole[:, 0], hole[:, 1], color='white')

plt.axis("equal")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.savefig("plots/jaggy_boundaries.png", dpi=300, bbox_inches='tight', 
            pad_inches=0,
            transparent=False) # Optional: Makes the background transparent)
if "-nopopup" not in sys.argv:
    plt.show()

## mesh for further calculations
import gmsh
import os

path = os.path.dirname(os.path.abspath(__file__))
mesh_dir = "/MeshDir/"

meshJaggy = path + mesh_dir + "meshJaggy"

lc_jaggy = 0.7

footMesh(meshJaggy, outer_loop, None, lc = lc_jaggy)

## Smoothing starts here

def get_smooth_points(data, s_factor=50, n_points=500):
    """Smooths a 2D array of points and returns them as a (N, 2) array."""
    # Ensure data is numpy array
    data = np.array(data)
    # splprep requires data in (2, N) format
    tck, u = splprep([data[:, 0], data[:, 1]], s=s_factor, per=True)
    u_new = np.linspace(0, 1, n_points)
    x_new, y_new = splev(u_new, tck)
    return np.column_stack((x_new, y_new))

# Smooth the outer loop (largest)
outer_smooth = get_smooth_points(outer_loop, s_factor=100)

# Smooth all hole loops using a list comprehension
holes_smooth = [get_smooth_points(h, s_factor=50) for h in hole_loops]

plt.figure(figsize=(8, 6))

# A. Plot the Outer Boundary
plt.plot(outer_smooth[:, 0], outer_smooth[:, 1], 'r-', linewidth=2, label='Outer Boundary')
# Optional: Fill the background to show the solid part
plt.fill(outer_smooth[:, 0], outer_smooth[:, 1], facecolor='lightgray', edgecolor='none', alpha=0.3)

# B. Plot all Holes
for i, hole in enumerate(holes_smooth):
    label = "Internal Holes" if i == 0 else None # Avoid duplicate legend entries
    plt.plot(hole[:, 0], hole[:, 1], 'b-', linewidth=1.5, label=label)
    # Fill holes with white to "cut" them out visually
    plt.fill(hole[:, 0], hole[:, 1], facecolor='white', edgecolor='none')

# C. Formatting
plt.axis("equal")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.legend(loc='upper right')

# Save and Show
plt.savefig("plots/smooth_boundaries.png", dpi=300, bbox_inches='tight', 
            pad_inches=0,
            transparent=False) # Optional: Makes the background transparent)
if "-nopopup" not in sys.argv:    
    plt.show()

# Shapely Polygon takes (shell, [hole1, hole2, ...])
# The smoothing ensures the geometry isn't "jaggy" before the extrusion
poly = Polygon(outer_smooth, holes_smooth)

# Extrude and Export ---
# Height is 80mm
extrusion_height = 80.0
mesh = trimesh.creation.extrude_polygon(poly, height=extrusion_height)

# Save as STL or OBJ
mesh.export("adios_dir_smoothing/smoothedFull.stl")
if "-nopopup" not in sys.argv:
    mesh.show()
print("Success! Extruded mesh with holes saved as 'final_model_with_holes.stl'")

## gmsh mesh for further calculations
meshSmoothed = path + mesh_dir + "meshSmoothed"

footMesh(meshSmoothed, outer_smooth, None, lc = None)