import numpy as np
from mpi4py import MPI
from topolegy.topopt import topopt
from dolfinx.io.gmshio import read_from_msh
from math import sin, cos
import adios4dolfinx
from pathlib import Path
import dolfinx
import argparse

# --- Command Line Argument Parsing ---
parser = argparse.ArgumentParser(description="Run topology optimization with custom parameters.")
parser.add_argument("-vol_frac", type=float, default=0.75, help="Volume fraction")
parser.add_argument("-filter_radius", type=float, default=10.0, help="Filter radius")
parser.add_argument("-save_log", type=str, default="results", help="Filename for the log")
parser.add_argument("-E", type=float, default=3000, help="Elasticity modulus in N/mm2")
parser.add_argument("-condition", type=str, default="P3")
parser.add_argument("-thickness", type=float, default=80.0)

args = parser.parse_args()

pathMesh = Path("adios_dir/mesh.bp")

mesh = adios4dolfinx.read_mesh(pathMesh, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

alpha = np.deg2rad(15)
beta = np.deg2rad(20)

condition = args.condition
thickness = args.thickness

if condition =="P3":
    F1 = 1610/thickness #N
    F2 = 1610/thickness #N
elif condition =="P4":
    F1 = 2065/thickness #N
    F2 = 2065/thickness #N
elif condition =="P5":
    F1 = 2240/thickness #N
    F2 = 2240/thickness #N
elif condition =="P6":
    F1 = 2490/thickness #N
    F2 = 2490/thickness #N
elif condition =="P7":
    F1 = 2900/thickness #N
    F2 = 2900/thickness #N
elif condition =="P8":
    F1 = 3200/thickness #N
    F2 = 3200/thickness #N

fem = {  # FEM parameters
    "mesh": mesh,
    "mesh_serial": mesh,
    "young's modulus": 3000,
    "poisson's ratio": 0.35,
    "thickness": thickness,
    "disp_bc": lambda x: np.greater_equal(x[1], 119.0),
    "traction_bcs": [
                    [(-F1*sin(alpha), F1*cos(alpha)),
                      lambda x: (np.greater_equal(x[1], 8) & np.less_equal(x[1], 12) & 
                                 np.greater_equal(x[0], 244) & np.less_equal(x[0], 248))],
                    [(F2*sin(beta), F2*cos(beta)),
                      lambda x: (np.greater_equal(x[1], 7) & np.less_equal(x[1], 10) & 
                                 np.greater_equal(x[0], 2) & np.less_equal(x[0], 8))],
                    ],
    "body_force": (0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
    },
}

opt = {  # Topology optimization parameters
    "max_iter": 299,
    "opt_tol": 1e-5,
    "vol_frac": args.vol_frac,
    "void_zone": lambda x: np.full(x.shape[1], False),
    "solid_zone": lambda x: (np.greater_equal(x[0], 225.0) & np.less_equal(x[1], 13.0) ) |
                            (np.greater_equal(x[1], 110.0)) |
                            (np.greater_equal(x[0], 3.0) & np.less_equal(x[0], 70.0) & np.less_equal(x[1], 16.0)),
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": args.filter_radius,
    "beta_interval": 50,
    "beta_max": 128,
    "use_oc": True,
    "move": 0.02,
    "opt_compliance": True,
    "save_log": args.save_log
}

if __name__ == "__main__":
    topopt(fem, opt)

# Execute the code in parallel:
# mpirun -n 8 python3 scripts/beam_2d.py
