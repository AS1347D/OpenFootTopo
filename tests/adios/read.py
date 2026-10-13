# adios_read.py
from dolfinx.fem import (Function, functionspace)
from basix.ufl import element

from mpi4py import MPI
import adios4dolfinx
import dolfinx
import numpy as np
from pathlib import Path

def f(x):
    return x[0]**2+x[1]
    
path_out = Path("adios_dir/checkpoint.bp")

mesh = adios4dolfinx.read_mesh(path_out, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

v_cg = element("Lagrange", mesh.basix_cell(), 2)
V = functionspace(mesh, v_cg)

kappa = Function(V)
kappa.interpolate(f)


kappa_import = Function(V)

adios4dolfinx.read_function(path_out, kappa_import, engine="BP4")

np.testing.assert_allclose(kappa.x.array, kappa_import.x.array, atol=1e-15)