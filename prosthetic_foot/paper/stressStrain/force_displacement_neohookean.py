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

parser = argparse.ArgumentParser(description="Hyperelastic (Neo-Hookean) force-displacement check with varying F.")
parser.add_argument("-F", type=float, default=2900, help="Force Magnitude")
parser.add_argument("-nsteps", type=int, default=2,
                     help="Number of load increments for Newton continuation (raise this if a run fails to converge)")
args = parser.parse_args()

# Mesh from Gmsh .msh file / ADIOS checkpoint (unchanged from the linear script)

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
# NOTE: for the hyperelastic verification run, use the SMOOTHED, single-material
# solid geometry (option 2 / Fig. 7d), not a raw SIMP density-field mesh. Near-zero
# density "void" elements can make the compressible Neo-Hookean energy density
# (which contains a -mu*ln(J) term) ill-conditioned or singular under load, since
# nothing physically prevents those near-zero-stiffness elements from collapsing
# (J -> 0). The smoothed solid geometry is also the physically meaningful one to
# check, since it's what actually gets printed.

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
    # Bumped from 2 -> 4: the Neo-Hookean energy density is nonlinear in u
    # (via ln(J) and C = F^T F), so the integrand is no longer a low-order
    # polynomial the way the linear-elastic bilinear form was.
    "quadrature_degree": 4,
}

# Material parameters and thickness
E = fem_params["young's modulus"]
nu = fem_params["poisson's ratio"]
h = thickness  # thickness

# Same (lambda, mu) as the linear model. Calibrating the hyperelastic model to
# these exact values is what makes it a *finite-strain generalization* of your
# existing linear model rather than a different material: as strain -> 0, this
# Neo-Hookean model's tangent stiffness recovers your linear sigma(u) exactly.
_lambda, mu = E * nu / (1 + nu) / (1 - 2 * nu), E / (2 * (1 + nu))  # Lame constants

element = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
V = functionspace(domain, element)

# u is now the unknown itself (nonlinear problem) rather than a TrialFunction.
# du is only used to build the Jacobian via automatic differentiation below.
u = Function(V, name="Displacement")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

# --- Finite-strain kinematics ---
d = len(u)
Ivec = ufl.Identity(d)
# ufl.diff (used below to get P from psi) only differentiates with respect to
# a ufl.variable, not an arbitrary expression -- so Fdef must be wrapped here,
# and every downstream quantity (C, J, Ic, psi) must be built from this same
# variable rather than from a fresh "Ivec + ufl.grad(u)" expression.
Fdef = ufl.variable(Ivec + ufl.grad(u))  # deformation gradient
C = Fdef.T * Fdef                  # right Cauchy-Green tensor
J = ufl.det(Fdef)                  # out-of-plane stretch = 1 under plane strain,
                                    # so this 2D determinant already equals det(F_3D)

# Plane-strain correction: the true 3D deformation gradient has a unit
# out-of-plane stretch (lambda_3 = 1), contributing an extra "+1" to the
# trace invariant that a naive 2D tr(C) would miss. Without this term the
# small-strain limit of this model would NOT exactly recover the plane-strain
# linear elasticity used elsewhere in the paper (Sec. A.1).
Ic = ufl.tr(C) + 1

# Compressible Neo-Hookean strain energy density (DOLFINx hyperelasticity-
# tutorial form), calibrated to (lambda, mu) above.
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (_lambda / 2) * (ufl.ln(J)) ** 2

# First Piola-Kirchhoff stress obtained by automatic differentiation of psi
# w.r.t. the deformation gradient -- this replaces the hand-written sigma(u).
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

# Load factor for Newton continuation: ramped 0 -> 1 over `nsteps` increments
# below so the nonlinear solver always starts each step from a converged,
# nearby state rather than jumping straight to the full ISO load.
load_factor = Constant(domain, 0.0)

metadata = {"quadrature_degree": fem_params["quadrature_degree"]}
dx = ufl.Measure("dx", metadata=metadata)
ds = ufl.Measure("ds", domain=domain, metadata=metadata, subdomain_data=facet_tags)

# Residual (virtual work principle): internal - external = 0
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
# Unlike the linear problem (one direct solve), the nonlinear problem is
# solved incrementally: ramp load_factor from 0 to 1 in `nsteps` steps,
# reusing the previous step's converged displacement as the initial guess
# for the next. If a run fails to converge, increase -nsteps rather than
# loosening the tolerances.
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

# Assuming F and u_max are your force and displacement values
file_path = "plots/force_displacement_neohookean.txt"

with open(file_path, "a") as f:
    f.write(f"{F_target:.6e}, {u_max:.6f}\n")


from topolegy.io_utils import calculate_stress
stresses = calculate_stress(domain, u, E, nu)
local_max_stress = np.max(stresses.x.array) if len(stresses.x.array) > 0 else 0.0
max_stress = MPI.COMM_WORLD.allreduce(local_max_stress, op=MPI.MAX)
print(F_target, max_stress)

from dolfinx import io
vtx_s = io.VTXWriter(domain.comm, "adios_dir/stresses_neohookean.bp", stresses, engine="BP4")
vtx_s.write(0)
vtx_s.close()