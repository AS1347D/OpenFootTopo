from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from pathlib import Path
from math import sin, cos
import dolfinx
from dolfinx import fem
from dolfinx.io import gmshio, XDMFFile
from dolfinx.fem import (
    dirichletbc,
    locate_dofs_topological,
    Constant,
    Function,
    functionspace,
)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import adios4dolfinx
from dolfinx.mesh import locate_entities_boundary, meshtags
import basix
import argparse

parser = argparse.ArgumentParser(description="Hyperelastic (Mooney-Rivlin) force-displacement check with varying F.")
parser.add_argument("-F", type=float, default=2900, help="Force Magnitude")
parser.add_argument("-nsteps", type=int, default=2,
                     help="Number of load increments for Newton continuation (raise this if a run fails to converge)")
parser.add_argument("-c10frac", type=float, default=0.75,
                     help="Fraction of the small-strain shear modulus assigned to C10 (rest goes to C01). "
                          "C10frac=1.0 reduces exactly to the Neo-Hookean model. This split is NOT determined "
                          "by (E, nu) alone -- without real biaxial/shear test data it is a free modeling "
                          "choice that only affects the response beyond the small-strain limit.")
args = parser.parse_args()

# Mesh from Gmsh .msh file / ADIOS checkpoint (unchanged from the Neo-Hookean script)

option = 2

if option == 1:
    pathMesh = Path("adios_dir/mesh.bp")
    domain = adios4dolfinx.read_mesh(pathMesh, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
    resultsFolder = "original/"
elif option == 2:
    domain, _, _ = gmshio.read_from_msh("MeshDir/meshSmoothed.msh", MPI.COMM_WORLD, gdim=2)
    resultsFolder = "smoothed/"
elif option == 3:
    domain, _, _ = gmshio.read_from_msh("MeshDir/meshJaggy.msh", MPI.COMM_WORLD, gdim=2)
    resultsFolder = "jaggy/"
# NOTE: as with the Neo-Hookean script, run this on the SMOOTHED, single-material
# solid geometry (option 2), not a raw SIMP density-field mesh -- see the note in
# force_displacement_neohookean.py for why.

dim = domain.topology.dim
fdim = dim - 1

alpha = np.deg2rad(15)
beta = np.deg2rad(20)

thickness = 80  # mm

F_target = args.F
F1 = F_target / thickness  # N
F2 = F_target / thickness  # N

fem_params = {  # FEM parameters
    "young's modulus": 3000,
    "poisson's ratio": 0.35,
    "disp_bc": lambda x: np.greater_equal(x[1], 119.0),
    "traction_bcs": [
        [(-F1 * sin(alpha), F1 * cos(alpha)),
         lambda x: (np.greater_equal(x[1], 8) & np.less_equal(x[1], 12) &
                    np.greater_equal(x[0], 244) & np.less_equal(x[0], 248))],
        [(F2 * sin(beta), F2 * cos(beta)),
         lambda x: (np.greater_equal(x[1], 7) & np.less_equal(x[1], 10) &
                    np.greater_equal(x[0], 2) & np.less_equal(x[0], 8))],
    ],
    "body_force": (0, 0),
    "quadrature_degree": 4,
}

# Material parameters and thickness
E = fem_params["young's modulus"]
nu = fem_params["poisson's ratio"]
h = thickness  # thickness

# Same (lambda, mu) as the linear and Neo-Hookean models.
_lambda, mu = E * nu / (1 + nu) / (1 - 2 * nu), E / (2 * (1 + nu))  # Lame constants
K = _lambda + (2.0 / 3.0) * mu  # bulk modulus, same small-strain volumetric response as before

# Mooney-Rivlin split: mu = 2*(C10 + C01) reproduces the correct small-strain
# shear modulus for ANY split -- the split itself is a free choice (see the
# -c10frac help text above) that only shows up once strains are no longer small.
C10 = args.c10frac * mu / 2.0
C01 = (1.0 - args.c10frac) * mu / 2.0

element = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
V = functionspace(domain, element)

u = Function(V, name="Displacement")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

# --- Finite-strain kinematics ---
d = len(u)
Ivec = ufl.Identity(d)
Fdef = ufl.variable(Ivec + ufl.grad(u))  # deformation gradient
C = Fdef.T * Fdef                        # right (2D, in-plane) Cauchy-Green tensor
J = ufl.det(Fdef)                        # out-of-plane stretch = 1 under plane strain

# Plane-strain correction, generalized to BOTH invariants I1 and I2 (not just
# I1 as in the Neo-Hookean script), since Mooney-Rivlin needs both:
#   3D block-diagonal C_3D = diag(C_2D, 1)  =>
#   I1(C_3D) = tr(C_2D) + 1
#   I2(C_3D) = det(C_2D) + tr(C_2D)
# (derived from I2 = 1/2*(tr(A)^2 - tr(A^2)) applied to the block-diagonal form;
#  for a 2x2 block, 1/2*(tr(C_2D)^2 - tr(C_2D^2)) = det(C_2D) exactly.)
Ic1 = ufl.tr(C) + 1
Ic2 = ufl.tr(C) + ufl.det(C)

# Isochoric (volume-preserving) invariants, decoupling deviatoric and
# volumetric response -- standard for compressible Mooney-Rivlin.
I1_bar = (J ** (-2.0 / 3.0)) * Ic1
I2_bar = (J ** (-4.0 / 3.0)) * Ic2

# Compressible Mooney-Rivlin strain energy density. With C01=0 (c10frac=1.0)
# this reduces exactly to a compressible Neo-Hookean model with the same
# (lambda, mu); with c10frac<1 it introduces genuine (but small at these
# strain levels) deviation from the Neo-Hookean curve.
psi = C10 * (I1_bar - 3) + C01 * (I2_bar - 3) + (K / 2) * (J - 1) ** 2

# First Piola-Kirchhoff stress via automatic differentiation of psi w.r.t. F.
P = ufl.diff(psi, Fdef)

disp_facets = locate_entities_boundary(domain, fdim, fem_params["disp_bc"])
bc = dirichletbc(Constant(domain, np.full(dim, 0.0)),
                  locate_dofs_topological(V, fdim, disp_facets), V)
bcs = [bc]

tractions, facets, markers = [], [], []
for marker, (traction, traction_bc) in enumerate(fem_params["traction_bcs"]):
    tractions.append(Constant(domain, np.array(traction, dtype=float)))
    current_facets = locate_entities_boundary(domain, fdim, traction_bc)
    facets.extend(current_facets)
    markers.extend([marker, ] * len(current_facets))

facets = np.array(facets, dtype=np.int32)
markers = np.array(markers, dtype=np.int32)
_, unique_indices = np.unique(facets, return_index=True)
facets, markers = facets[unique_indices], markers[unique_indices]
sorted_indices = np.argsort(facets)
facet_tags = meshtags(domain, fdim, facets[sorted_indices], markers[sorted_indices])

b = Constant(domain, np.array(fem_params["body_force"], dtype=float))

load_factor = Constant(domain, 0.0)

metadata = {"quadrature_degree": fem_params["quadrature_degree"]}
dx = ufl.Measure("dx", metadata=metadata)
ds = ufl.Measure("ds", domain=domain, metadata=metadata, subdomain_data=facet_tags)

Fres = ufl.inner(P, ufl.grad(v)) * dx - ufl.dot(load_factor * b, v) * dx
for marker, t in enumerate(tractions):
    Fres += -ufl.dot(load_factor * t, v) * ds(marker)

Jac = ufl.derivative(Fres, u, du)

problem = NonlinearProblem(Fres, u, bcs=bcs, J=Jac)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 50
solver.report = True

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.prefix
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

# --- Load-stepping / Newton continuation ---
n_steps = args.nsteps
for step in range(1, n_steps + 1):
    load_factor.value = step / n_steps
    n_its, converged = solver.solve(u)
    if not converged:
        raise RuntimeError(
            f"Newton solver did not converge at load step {step}/{n_steps} "
            f"(F={F_target * step / n_steps:.1f} N). Try increasing -nsteps."
        )
    u.x.scatter_forward()
    if MPI.COMM_WORLD.rank == 0:
        print(f"  load step {step}/{n_steps} (F={F_target * step / n_steps:.1f} N): "
              f"converged in {n_its} Newton iterations")

# print maximum displacement magnitude
bs = V.dofmap.index_map_bs
u_values = u.x.array.reshape(-1, bs)
local_magnitude = np.linalg.norm(u_values, axis=1)

if len(local_magnitude) > 0:
    local_max = np.max(local_magnitude)
else:
    local_max = 0.0

u_max = MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)

print(F_target, u_max)

file_path = "plots/force_displacement_mooneyrivlin.txt"

with open(file_path, "a") as f:
    f.write(f"{F_target:.6e}, {u_max:.6f}\n")

from topolegy.io_utils import calculate_stress
stresses = calculate_stress(domain, u, E, nu)
local_max_stress = np.max(stresses.x.array) if len(stresses.x.array) > 0 else 0.0
max_stress = MPI.COMM_WORLD.allreduce(local_max_stress, op=MPI.MAX)
print(F_target, max_stress)

# from dolfinx import io
# vtx_s = io.VTXWriter(domain.comm, "adios_dir/stresses_MR.bp", stresses, engine="BP4")
# vtx_s.write(0)
# vtx_s.close()