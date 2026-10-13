# adios_write.py
from dolfinx.fem import (Function, functionspace, Expression)
from basix.ufl import element
from mpi4py import MPI
import dolfinx.io
import adios4dolfinx
from pathlib import Path

def f(x):
    return x[0]**2+x[1]

path = Path("adios_dir/mesh.bp")

mesh = adios4dolfinx.read_mesh(path, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

v_cg = element("Lagrange", mesh.basix_cell(), 2)
V = functionspace(mesh, v_cg)

kappa = Function(V)
kappa.interpolate(f)

path_out = Path("adios_dir/checkpoint.bp")
adios4dolfinx.write_mesh(path_out, mesh, engine="BP4")
adios4dolfinx.write_function(path_out, kappa, engine="BP4")

path_vis = Path("adios_dir/visualization.bp")
with dolfinx.io.VTXWriter(MPI.COMM_WORLD, path_vis, kappa, "bp4") as vtx_file:
    vtx_file.write(0)