from mpi4py import MPI
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
    form,
    Constant,
)

import adios4dolfinx
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags
import basix
import argparse

parser = argparse.ArgumentParser(description="Run force displacement check with varying F.")
parser.add_argument("-F", type=float, default=1, help="Force Magnitude")
args = parser.parse_args()

# Mesh from Gmsh .msh file

option = 2

if option == 1:
    pathMesh = Path("adios_dir/mesh.bp")
    domain = adios4dolfinx.read_mesh(pathMesh, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
    resultsFolder = "original/"
elif option == 2:
    domain ,_, _ = gmshio.read_from_msh("MeshDir/meshSmoothed.msh", MPI.COMM_WORLD, gdim=2)
    resultsFolder = "smoothed/"
elif option == 3:
    domain ,_, _ = gmshio.read_from_msh("MeshDir/meshJaggy.msh", MPI.COMM_WORLD, gdim=2)
    resultsFolder = "jaggy/"

dim = domain.topology.dim
fdim = dim - 1

alpha = np.deg2rad(15)
beta = np.deg2rad(20)

thickness = 80 #mm

F = args.F
F1 = F/thickness # N
F2 = F/thickness # N


fem_params = {  # FEM parameters
    "young's modulus": 3000,
    "poisson's ratio": 0.35,
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

# Material parameters and thickness
E = fem_params["young's modulus"]
nu = fem_params["poisson's ratio"]
h = thickness  # thickness

# # Plane-stress lambda (as in some dolfinx elasticity tutorials):contentReference[oaicite:2]{index=2}
_lambda, mu = E*nu/(1+nu)/(1-2*nu), E/(2*(1+nu))  # Lame constants

element = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
V = fem.functionspace(domain, element)

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


# Visualize traction vectors
V_vis = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
t_vis = fem.Function(V_vis, name="Traction_Visual")

# Initialize to zero (so non-boundary areas have 0 vectors)
t_vis.x.array[:] = 0.0

# Get the Block Size (2 for 2D, 3 for 3D) to handle vector indexing
bs = V_vis.dofmap.index_map_bs

# Iterate over your input dictionary again to map Values -> DOFs
for marker, (traction_val, _) in enumerate(fem_params["traction_bcs"]):
    # Find the facets associated with this specific marker
    # (Note: 'facet_tags' is the MeshTags object you created in your snippet)
    facets = facet_tags.find(marker)

    if len(facets) > 0:
        # Locate the Degrees of Freedom (nodes) on these facets
        dofs = fem.locate_dofs_topological(V_vis, fdim, facets)
        # Set the vector components for these DOFs
        # traction_val is your tuple, e.g., (0.0, -40.0)
        for i, val in enumerate(traction_val):
            # We access the flat array. 
            # The pattern is: [dof0_x, dof0_y, dof1_x, dof1_y, ...]
            t_vis.x.array[dofs * bs + i] = val

# Write to XDMF for ParaView
with XDMFFile(domain.comm, "stressDir/tractions_visual.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(t_vis)

b = Constant(domain, np.array(fem_params["body_force"], dtype=float))

metadata = {"quadrature_degree": fem_params["quadrature_degree"]}
dx = ufl.Measure("dx", metadata=metadata)
ds = ufl.Measure("ds", domain=domain, metadata=metadata, subdomain_data=facet_tags)

lhs = ufl.inner(sigma(u), epsilon(v))*dx
rhs = ufl.dot(b, v)*dx
for marker, t in enumerate(tractions):
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

# print maximum displacement magnitude
u_values = uh.x.array.reshape(-1, bs)
local_magnitude = np.linalg.norm(u_values, axis=1)

if len(local_magnitude) > 0:
    local_max = np.max(local_magnitude)
else:
    local_max = 0.0
    
u_max = MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)

print(F, u_max)

# Assuming F and u_max are your force and displacement values
file_path = "plots/force_displacement.txt"

with open(file_path, "a") as f:
    # Use an f-string to format the output clearly
    f.write(f"{F:.6e}, {u_max:.6f}\n")