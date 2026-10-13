# adios_read.py
from dolfinx.fem import (Function, functionspace)
from dolfinx import plot
from mpi4py import MPI
import adios4dolfinx
import dolfinx
from pathlib import Path
import basix

path_out = Path("adios_dir/checkpoint.bp")
path_pyvista = Path("adios_dir/laplacian.png")
path_obj = Path("printing/preparedGeometry.obj")
path_stl = Path("printing/preparedGeometry.stl")

domain = adios4dolfinx.read_mesh(path_out, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

S = functionspace(domain, ("CG", 1))

rho_import = Function(S)

adios4dolfinx.read_function(path_out, rho_import, engine="BP4")

from topolegy.io_utils import pyvista_visualizer

pyvista_visualizer(path_pyvista, rho_import, show_mesh=False)

dg0_el = basix.ufl.element("DG", domain.basix_cell(), 0)
Q = functionspace(domain, dg0_el)
rho0 = Function(Q)
rho0.interpolate(rho_import)  # cellwise constant approximation

thickness = 80 #mm
rho_threshold = 0.60

import pyvista as pv

topo_dim = domain.topology.dim
cells, cell_types, points = plot.vtk_mesh(domain, topo_dim)

grid = pv.UnstructuredGrid(cells, cell_types, points)

# DG0 has one value per cell
grid.cell_data["rho"] = rho0.x.array.real

# --- 3) Threshold “solid” cells
solid = grid.threshold(value=rho_threshold, scalars="rho")

if solid.n_cells == 0:
    raise RuntimeError(
        f"No cells found with rho >= {rho_threshold}. Try lowering threshold (e.g. 0.95)."
    )

# --- 4) Extract surface and triangulate
surf = solid.extract_surface().triangulate()


# surf = surf.extract_feature_edges(
#     boundary_edges=True,
#     feature_edges=True,
#     manifold_edges=True,
#     non_manifold_edges=True
# )
surf = surf.connectivity(largest=True).triangulate().clean()
surf.save("printing/surface2d.stl")


import vtk


def vtk_extrude_cap(poly: pv.PolyData, thickness: float) -> pv.PolyData:
    f = vtk.vtkLinearExtrusionFilter()
    f.SetInputData(poly)
    f.SetExtrusionTypeToVectorExtrusion()
    f.SetVector(0.0, 0.0, float(thickness))
    f.SetCapping(True)  # <-- makes top/bottom caps
    f.Update()
    return pv.wrap(f.GetOutput()).triangulate().clean()

def assert_watertight(poly: pv.PolyData, name="mesh"):
    """Raise if there are open boundary edges."""
    edges = poly.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False
    )
    if edges.n_cells > 0:
        raise RuntimeError(f"{name} is NOT watertight: {edges.n_cells} boundary edge segments found.")
    return True


def make_watertight(poly: pv.PolyData,
                    fill_hole_size=1e12,
                    clean_tol=1e-8,
                    fix_normals=True) -> pv.PolyData:
    """
    Make a closed surface suitable as a 'solid' for printing:
      - triangulate + clean
      - fill holes
      - clean again
      - orient normals consistently
    """
    poly = poly.triangulate()
    poly = poly.clean(tolerance=clean_tol, absolute=True)

    # Fill any remaining holes (including extrusion side gaps)
    poly = poly.fill_holes(fill_hole_size).triangulate().clean()

    # Optional: fix normals (important for slicers)
    if fix_normals:
        poly = poly.compute_normals(
            auto_orient_normals=True,
            consistent_normals=True,
            split_vertices=False
        )

    # One more clean
    poly = poly.clean(tolerance=clean_tol, absolute=True)
    return poly

# region_boundary = solid.extract_surface().triangulate().clean()

body = vtk_extrude_cap(surf, thickness)

# Make it watertight / "solid" for printing
body = make_watertight(body, fill_hole_size=1e12, clean_tol=1e-8, fix_normals=True)

# Optional: keep largest connected component
body = body.connectivity(largest=True).triangulate().clean()

# Sanity check (raises if not watertight)
assert_watertight(body, "extruded body")

# body = body.smooth(n_iter=1, relaxation_factor=0.01,
#                    feature_angle=45, boundary_smoothing=True, feature_smoothing=False, inplace=False)

if domain.comm.rank == 0:
    body.save(path_stl)
    body.save(path_obj)
    print(f"Wrote {path_stl} (triangles: {body.n_cells})")
