# adios_read.py
from dolfinx.fem import (Function, functionspace)
from dolfinx import plot
from mpi4py import MPI
import adios4dolfinx
import dolfinx
from pathlib import Path
import pyvista as pv
import basix

path_out = Path("adios_dir/checkpoint.bp")
path_pyvista = Path("adios_dir/isoline.png")
path_stl = Path("printing/jaggyFull.stl")

domain = adios4dolfinx.read_mesh(path_out, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

S = functionspace(domain, ("CG", 1))

rho_import = Function(S)

adios4dolfinx.read_function(path_out, rho_import, engine="BP4")

dg0_el = basix.ufl.element("DG", domain.basix_cell(), 0)
Q = functionspace(domain, dg0_el)
rho0 = Function(Q)
rho0.interpolate(rho_import)  # cellwise constant approximation

thickness = 80 #mm
rho_threshold = 0.90

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
        f"No cells found with rho >= {rho_threshold}. "
        "Try lowering threshold (e.g. 0.95)."
    )

# --- 4) Extract surface and triangulate
surf = solid.extract_surface().triangulate()

import vtk


def vtk_extrude_cap(poly: pv.PolyData, thickness: float) -> pv.PolyData:
    f = vtk.vtkLinearExtrusionFilter()
    f.SetInputData(poly)
    f.SetExtrusionTypeToVectorExtrusion()
    f.SetVector(0.0, 0.0, float(thickness))
    f.SetCapping(True)  # <-- makes top/bottom caps
    f.Update()
    return pv.wrap(f.GetOutput()).triangulate().clean()

# region_boundary = solid.extract_surface().triangulate().clean()


body = vtk_extrude_cap(surf, thickness)

body = body.fill_holes(1e6).triangulate().clean()

body = body.connectivity(largest=True)

# body = body.smooth_taubin(
#     n_iter=2,
#     pass_band=0.2   # 0.05–0.2 typical
# )

if domain.comm.rank == 0:
    body.save(path_stl)
    print(f"Wrote {path_stl} (triangles: {body.n_cells})")
