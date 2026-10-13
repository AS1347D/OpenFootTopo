# adios_mesh.py
from dolfinx.mesh import create_unit_square
import adios4dolfinx
from mpi4py import MPI
from pathlib import Path

path = Path("adios_dir/mesh.bp")
mesh = create_unit_square(MPI.COMM_WORLD, 30, 30)
adios4dolfinx.write_mesh(path, mesh, engine='BP4')