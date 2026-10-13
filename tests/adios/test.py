# adios_mesh.py
from dolfinx.mesh import create_unit_square
from dolfinx.fem import (Function, functionspace, Expression)

import adios4dolfinx
from mpi4py import MPI
from pathlib import Path

mesh = create_unit_square(MPI.COMM_WORLD, 30, 30)

V = functionspace(mesh, ("Lagrange", 1))

kappa = Function(V)

def f(x):
    return x[0]**2+x[1]

kappa.interpolate(f)