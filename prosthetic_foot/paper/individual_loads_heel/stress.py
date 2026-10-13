from dolfinx.io import gmshio, XDMFFile, VTXWriter
from dolfinx.fem import (
    dirichletbc,
    assemble_scalar,
    locate_dofs_topological,
    form,
    Constant,
    functionspace
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags
from topolegy.io_utils import traction_visualizer, calculate_stress, calculate_strain, vtx_writer
from math import sin, cos
from pathlib import Path
from mpi4py import MPI

import adios4dolfinx
import numpy as np
import dolfinx
import basix
import ufl

option = 2

if option == 1:
    pathMesh = Path("adios_dir/mesh.bp")
    domain = adios4dolfinx.read_mesh(pathMesh, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
    resultsFolder = "original/"
elif option == 2:
    domain ,_, _ = gmshio.read_from_msh("MeshDir/meshJaggy.msh", MPI.COMM_WORLD, gdim=2)
    resultsFolder = "jaggy/"
elif option == 3:
    domain ,_, _ = gmshio.read_from_msh("MeshDir/meshSmoothed.msh", MPI.COMM_WORLD, gdim=2)
    resultsFolder = "smoothed/"

dim = domain.topology.dim
fdim = dim - 1

alpha = np.deg2rad(15)
beta = np.deg2rad(20)

condition =  "P8"
thickness = 80 #mm

if condition =="P3":
    F1 = 1610/thickness #N volfrac 0.34
    F2 = 1610/thickness #N
elif condition =="P4":
    F1 = 2065/thickness #N
    F2 = 2065/thickness #N
elif condition =="P5":
    F1 = 2240/thickness #N volfrac 0.45
    F2 = 2240/thickness #N
elif condition =="P6":
    F1 = 2490/thickness #N
    F2 = 2490/thickness #N
elif condition =="P7":
    F1 = 2900/thickness #N volfrac 0.61
    F2 = 2900/thickness #N
elif condition =="P8":
    F1 = 3200/thickness #N
    F2 = 3200/thickness #N

fem_params = {  # FEM parameters
    "young's modulus": 3000,
    "poisson's ratio": 0.35,
    "disp_bc": lambda x: np.greater_equal(x[1], 119.0),
    "traction_bcs": [
                    [(-F1*sin(alpha), F1*cos(alpha)),
                      lambda x: (np.greater_equal(x[1], 8) & np.less_equal(x[1], 12) & 
                                 np.greater_equal(x[0], 244) & np.less_equal(x[0], 248))],
                    # [(F2*sin(beta), F2*cos(beta)),
                    #   lambda x: (np.greater_equal(x[1], 7) & np.less_equal(x[1], 10) & 
                    #              np.greater_equal(x[0], 2) & np.less_equal(x[0], 8))],
                    ],
    "body_force": (0, 0),
    "quadrature_degree": 2,
    "thickness" : thickness,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
    },
}

# Material parameters and thickness
E = fem_params["young's modulus"]
nu = fem_params["poisson's ratio"]
h = thickness  # thickness
L_f = 260 #mm
# # Plane-stress lambda (as in some dolfinx elasticity tutorials):contentReference[oaicite:2]{index=2}
_lambda, mu = E*nu/(1+nu)/(1-2*nu), E/(2*(1+nu))  # Lame constants

element = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
V = functionspace(domain, element)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Kinematics
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):  # 3D or plane strain
    return 2*mu*epsilon(u) + _lambda*ufl.tr(epsilon(u))*ufl.Identity(len(u))

dim = domain.topology.dim
fdim = dim - 1
disp_facets = locate_entities_boundary(domain, fdim, fem_params["disp_bc"])
bc = dirichletbc(Constant(domain, np.full(dim, 0.0)),
                    locate_dofs_topological(V, fdim, disp_facets), V)
bcs = [bc]

# A safe, small tolerance. Use 1e-12 or 1e-10.
tol_dof = 1e-3

tractions, facets, markers = [], [], []
for marker, (traction, traction_bc) in enumerate(fem_params["traction_bcs"]):
    tractions.append(Constant(domain, np.array(traction, dtype=float)))
    current_facets = locate_entities_boundary(domain, fdim, traction_bc)
    facets.extend(current_facets)
    markers.extend([marker,]*len(current_facets))

facets = np.array(facets, dtype=np.int32)
markers = np.array(markers, dtype=np.int32)
_, unique_indices = np.unique(facets, return_index=True)
facets, markers = facets[unique_indices], markers[unique_indices]
sorted_indices = np.argsort(facets)
facet_tags = meshtags(domain, fdim, facets[sorted_indices], markers[sorted_indices])

metadata = {"quadrature_degree": fem_params["quadrature_degree"]}
dx = ufl.Measure("dx", metadata=metadata)
ds = ufl.Measure("ds", domain=domain, metadata=metadata, subdomain_data=facet_tags)

filePath = "stressDir/tractions_visual.xdmf"
traction_visualizer(domain, facet_tags, fem_params, filePath)

b = Constant(domain, np.array(fem_params["body_force"], dtype=float))

lhs = ufl.inner(sigma(u), epsilon(v))*dx
rhs = ufl.dot(b, v)*dx
for marker, t in enumerate(tractions):
    if dim==2:
        line_length =  assemble_scalar(form(Constant(domain, 1.0)*ds(marker)))
        rhs += ufl.dot(t, v)/line_length*ds(marker)
    elif dim==3:
        rhs += ufl.dot(t, v)*ds(marker)

# Assemble system
a = form(lhs)
L = form(rhs)

problem = LinearProblem(
    lhs,
    rhs,
    bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    # petsc_options_prefix="linear_elasticity",
)
uh = problem.solve()

with XDMFFile(domain.comm, "stressDir/"+resultsFolder+"u.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

# Compliance calculation
Comp = assemble_scalar(form(ufl.inner(sigma(uh), epsilon(uh))*dx))
print(f"Compliance = {Comp:.6f}")

# gamma = Comp*h*E*L_f/(F1*h)**2
# print(f"Gamma = {gamma:.6f} and log(Gamma) = {np.log10(gamma):.6f}")

stresses = calculate_stress(domain, uh, E, nu)
max_stress = domain.comm.allreduce(np.max(stresses.x.array), op=MPI.MAX)
print(f"Max Equivalent Stress = {max_stress:.6f}")

vtx_writer("stressDir/"+resultsFolder+"stresses", domain, stresses)

strains = calculate_strain(domain, uh, E, nu)
max_strain = domain.comm.allreduce(np.max(strains.x.array), op=MPI.MAX)
print(f"Max Equivalent Strain = {max_strain:.6f}")

vtx_writer("stressDir/"+resultsFolder+"strains", domain, strains)
