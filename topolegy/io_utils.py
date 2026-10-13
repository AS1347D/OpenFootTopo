from dolfinx.fem import Function, functionspace, form, assemble_scalar, locate_dofs_topological, Constant, petsc
from dolfinx.io import XDMFFile, VTKFile, VTXWriter
from mpi4py import MPI
from math import sqrt
import meshio
import os
import json
import ast
import ufl

import datetime

def start_time():
    return datetime.datetime.now()

def execution_time(start_time):
    if MPI.COMM_WORLD.rank == 0:
        print("Total Execution Time: ", datetime.datetime.now()-start_time)

def info(str):
    """Only prints information message for once. Useful for logging,

    Args:
        str ('str'): log entry
    """
    if MPI.COMM_WORLD.Get_rank()==0:
        print(str)

def dict_writer(filename, dictionary, extension = ".txt"):
    """Writes dictionary object into a text file.

    Args:
        filename ('str'): file path
        dictionary ('dict'): dictionary object
        extension (str, optional): file extension. Defaults to ".txt".
    """
    with open(filename+extension, 'w') as file:
        file.write(json.dumps(str(dictionary))) 
    if MPI.COMM_WORLD.rank==0:
        print(filename+extension, " is saved.")

def dict_loader(filename, extension = ".txt"):
    """Loads dictionary into python script

    Args:
        filename ('str'): file path
        extension (str, optional): file extension. Defaults to ".txt".

    Returns:
        dictionary ('dict'): dictionary object
    """
    with open(filename+extension) as f:
        data = json.load(f)
    data = ast.literal_eval(data)
    if MPI.COMM_WORLD.rank==0:
        print(filename+extension, " is loaded.")
    return data

def xdmf_writer(name, mesh, function):
    """ writes functions into xdmf file

    Args:
        name (string): name of the file
        mesh (dolfinx.mesh.Mesh]): Dolfinx mesh
        function (dolfinx.fem.function.Function): Dolfinx function to be saved.
    """
    el = function.function_space.element 
    if el.basix_element.degree>1:
        el_type = el.basix_element.family
        V = functionspace(mesh, (el_type,1))
        function_interpolation = Function(V)
        function_interpolation.interpolate(function)
    else:
        function_interpolation = function


    with XDMFFile(MPI.COMM_WORLD, name+".xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(function_interpolation)

def vtk_writer(name, mesh, function):
    """ writes functions into xdmf file

    Args:
        name (string): name of the file
        mesh (dolfinx.mesh.Mesh]): Dolfinx mesh
        function (dolfinx.fem.function.Function): Dolfinx function to be saved.
    """
    with VTKFile(MPI.COMM_WORLD, name+".pvd", "w") as vtk:
        vtk.write_mesh(mesh)
        vtk.write_function(function)

def vtx_writer(name, mesh, function):
    """ writes functions into adios bp file

    Args:
        name (string): name of the file
        mesh (dolfinx.mesh.Mesh]): Dolfinx mesh
        function (dolfinx.fem.function.Function): Dolfinx function to be saved.
    """
    with VTXWriter(mesh.comm, name+".bp", function, engine="BP4") as vtx_s:
        vtx_s.write(0)
        vtx_s.close()
        
def create_mesh(mesh, cell_type, prune_z):
    """Subroutine for mesh creation by using meshio library

    Args:
        mesh (meshio._mesh.Mesh): mesh to be converted into Dolfinx mesh
        cell_type ('str'): type of cell (it becomes tetrahedral most of the time)
        prune_z ('bool'): whether consider the 3th dimension's coordinate or not, (it should be False for 2D cases)

    Returns:
        meshio._mesh.Mesh: converted dolfinx mesh
    """

    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    
    if cell_type=="tetra":
        print("Number of 3D cells:  {:,}".format(len(cells)))
    elif cell_type=="triangle":
        print("Number of 2D cells (facets):  {:,}".format(len(cells)))

    return out_mesh

def write_xdmf_mesh(name, dimension, write_edge=False):
    """Writes gmsh (.msh) mesh as an .xdmf mesh

    Args:
        name ('str'): filename
        dimension ('int'): Dimension of the mesh (2 or 3)
    """

    if MPI.COMM_WORLD.Get_rank() == 0:
        msh_name = name + ".msh"
        msh = meshio.read(msh_name)

        if dimension == 1:
            prune_z = True
            volume_mesh = create_mesh(msh, "line",prune_z)
            tag_mesh = create_mesh(msh, "vertex",prune_z)

        if dimension == 2:
            prune_z = True
            volume_mesh = create_mesh(msh, "triangle",prune_z)
            tag_mesh = create_mesh(msh, "line",prune_z)

        elif dimension == 3:
            prune_z = False
            volume_mesh = create_mesh(msh, "tetra",prune_z)
            tag_mesh = create_mesh(msh, "triangle",prune_z)
            if write_edge:
                edge_mesh = create_mesh(msh, "line",prune_z)
                xdmf_edge_name = name + "_edgetags.xdmf"
                meshio.write(xdmf_edge_name, edge_mesh)
            
        # Create and save one file for the mesh, and one file for the facets and one file for the edges
        xdmf_name = name + ".xdmf"
        xdmf_tags_name = name + "_tags.xdmf"
        
        meshio.write(xdmf_name, volume_mesh)
        meshio.write(xdmf_tags_name, tag_mesh)

        print(str(dimension)+"D XDMF mesh is generated.")

def load_xdmf_mesh(name):
    """Loads xdmf mesh into python script

    Args:
        name ('str'): Name of the .xdmf file

    Returns:
        tuple: mesh, boundary tags and volume tags of the geometry
    """
    mesh_loader_name = name + ".xdmf"
    tag_loader_name = name + "_tags.xdmf"
    with XDMFFile(MPI.COMM_WORLD, mesh_loader_name, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    with XDMFFile(MPI.COMM_WORLD, tag_loader_name, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")
    
    if MPI.COMM_WORLD.rank == 0:
        print("XDMF Mesh is loaded.")
    return mesh, cell_tags, facet_tags

class XDMFReader:
    """This class generates geometry objec to load its instances and information about it (number of elements etc.)
    """
    def __init__(self, name):
        self.name = name
        self._mesh = None
        self._cell_tags = None
        self._facet_tags = None
        self._gdim = None
        mesh_loader_name = name + ".xdmf"
        tag_loader_name = name + "_tags.xdmf"
        edgetag_loader_name = name + "_edgetags.xdmf"

        with XDMFFile(MPI.COMM_WORLD, mesh_loader_name, "r") as xdmf:
            self._mesh = xdmf.read_mesh(name="Grid")
            self._cell_tags = xdmf.read_meshtags(self.mesh, name="Grid")
        info("\nXDMF Mesh - Cell data loaded.")

        self._mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)        
        self._mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)        
        self._mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim-1)
        with XDMFFile(MPI.COMM_WORLD, tag_loader_name, "r") as xdmf:
            self._facet_tags = xdmf.read_meshtags(self.mesh, name="Grid")    
        info("XDMF Mesh - Facet data loaded.")

        if os.path.exists(edgetag_loader_name):
            self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim-2)
            with XDMFFile(MPI.COMM_WORLD, edgetag_loader_name, "r") as xdmf:
                self._edge_tags = xdmf.read_meshtags(self.mesh, name="Grid")
            info("XDMF Mesh - Edge data loaded.")
    
    @property
    def mesh(self):
        return self._mesh
    @property
    def subdomains(self):
        return self._cell_tags
    @property
    def facet_tags(self):
        return self._facet_tags   
    @property
    def edge_tags(self):
        return self._edge_tags 
    @property
    def dimension(self):
        return self._mesh.topology.dim    

    def getAll(self):
        return self.mesh, self.subdomains, self.facet_tags
    
    def getInfo(self):
        t_imap = self.mesh.topology.index_map(self.mesh.topology.dim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        total_num_cells = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM) #sum all cells and distribute to each process
        if MPI.COMM_WORLD.Get_rank()==0:
            print("Number of cells:  {:,}".format(total_num_cells))
            print("Number of cores: ", MPI.COMM_WORLD.Get_size(), "\n")
        return total_num_cells
    
import adios4dolfinx
import pathlib

def export_function(function: Function, directory: str, filename: str) -> None:
    """
    Export a dolfinx.fem.Function to file.

    Parameters
    ----------
    function
        Function to be exported.
    directory
        Directory where to export the function.
    filename
        Name of the file where to export the function.
    """
    comm = function.function_space.mesh.comm
    visualization_directory = os.path.join(directory, filename + ".bp")
    checkpointing_directory = os.path.join(directory, filename + ".bp", ".checkpoint")
    os.makedirs(visualization_directory, exist_ok=True)
    os.makedirs(checkpointing_directory, exist_ok=True)

    # Export for visualization
    with VTXWriter(comm, visualization_directory, function, "bp4") as vtx_file:
        vtx_file.write(0)

    # Export for checkpointing
    adios4dolfinx.write_function(function, pathlib.Path(checkpointing_directory), "bp4")

def import_function(function_space: functionspace, directory: str, filename: str) -> Function:
    """
    Import a dolfinx.fem.Function from file.

    Parameters
    ----------
    function_space
        Finite element space on which the function to be imported lives.
    directory
        Directory where to import the function from.
    filename
        Name of the file where to import the function from.

    Returns
    -------
    :
        Function imported from file.
    """
    function = Function(function_space)
    checkpointing_directory = os.path.join(directory, filename + ".bp", ".checkpoint")
    adios4dolfinx.read_function(function, pathlib.Path(checkpointing_directory), "bp4")
    return function


def pyvista_visualizer(Path, Function, show_mesh=True):
    import pyvista
    from dolfinx import plot

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Function.function_space)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["rho"] = Function.x.array.real
    u_grid.set_active_scalars("rho")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, cmap="jet", show_edges=show_mesh)
    u_plotter.view_xy()
    u_plotter.show()
    u_plotter.screenshot(Path)

def traction_visualizer(domain, facet_tags, fem_params, filePath):

    dim = domain.topology.dim
    fdim = dim - 1

    # Visualize traction vectors
    V_vis = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    t_vis = Function(V_vis, name="Traction_Visual")

    # Initialize to zero (so non-boundary areas have 0 vectors)
    t_vis.x.array[:] = 0.0

    bs = V_vis.dofmap.index_map_bs

    metadata = {"quadrature_degree": fem_params["quadrature_degree"]}
    ds = ufl.Measure("ds", domain=domain, metadata=metadata, subdomain_data=facet_tags)

    for marker, (traction_val, _) in enumerate(fem_params["traction_bcs"]):
        # Find the facets associated with this specific marker
        facets = facet_tags.find(marker)
        line_length =  assemble_scalar(form(Constant(domain, 1.0)*ds(marker)))

        if len(facets) > 0:

            # Locate the Degrees of Freedom (nodes) on these facets
            dofs = locate_dofs_topological(V_vis, fdim, facets)
            
            # Set the vector components for these DOFs
            for i, val in enumerate(traction_val):
                # We access the flat array. 
                # The pattern is: [dof0_x, dof0_y, dof1_x, dof1_y, ...]

                if dim==2 and fem_params.get("thickness"):
                    t_vis.x.array[dofs * bs + i] = val/line_length
                
                elif dim==3:
                    t_vis.x.array[dofs * bs + i] = val
                
            if dim==2 and fem_params.get("thickness"):
                force_x = assemble_scalar(form(t_vis[0] * ds(marker)))
                force_y = assemble_scalar(form(t_vis[1] * ds(marker)))
                
                magnitude = sqrt(force_x**2 + force_y**2)*fem_params["thickness"]
                print(f"Marker {marker} - Vector: ({force_x:.4e}, {force_y:.4e}) | Magnitude Check: {magnitude:.4e}")

            elif dim==3:
                t_vis.x.array[dofs * bs + i] = val
                
                force_x = assemble_scalar(form(t_vis[0] * ds(marker)))
                force_y = assemble_scalar(form(t_vis[1] * ds(marker)))
                force_z = assemble_scalar(form(t_vis[2] * ds(marker)))
                
                magnitude = sqrt(force_x**2 + force_y**2+ force_z**2)
                
                print(f"Marker {marker} - Vector: ({force_x:.4e}, {force_y:.4e}) | Magnitude Check: {magnitude:.4e}")

    # Write to XDMF for ParaView
    with XDMFFile(domain.comm, filePath, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(t_vis)

def calculate_stress(domain, uh, E, nu):
    # Stress calculation (plane strain, including out-of-plane sigma_zz)

    _lambda, mu = E*nu/(1+nu)/(1-2*nu), E/(2*(1+nu))  # Lame constants

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma_2d(u):  # in-plane (2x2) stress only
        return 2*mu*epsilon(u) + _lambda*ufl.tr(epsilon(u))*ufl.Identity(len(u))

    V_dg0 = functionspace(domain, ("DG", 0))

    sigma_inplane = sigma_2d(uh)
    # Out-of-plane Cauchy stress under plane strain (epsilon_zz = 0):
    sigma_zz = _lambda * ufl.tr(epsilon(uh))

    T = ufl.tr(sigma_inplane) + sigma_zz            # full 3D trace, includes zz
    s_inplane = sigma_inplane - (T / 3) * ufl.Identity(len(uh))
    s_zz = sigma_zz - T / 3

    von_Mises = ufl.sqrt(1.5 * (ufl.inner(s_inplane, s_inplane) + s_zz**2))

    stress = Function(V_dg0, name="VonMisesStress")
    v = ufl.TestFunction(V_dg0)
    u_trial = ufl.TrialFunction(V_dg0)
    a = ufl.inner(u_trial, v) * ufl.dx
    L = ufl.inner(von_Mises, v) * ufl.dx
    problem = petsc.LinearProblem(a, L, u=stress)
    problem.solve()
    return stress

def calculate_strain(domain, uh, E, nu):
    # Strain calculation (plane strain, including out-of-plane deviatoric term)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    V_dg0 = functionspace(domain, ("DG", 0))
    strains = Function(V_dg0)

    eps_inplane = epsilon(uh)
    tr_eps = ufl.tr(eps_inplane)  # = tr(eps_3D) too, since eps_zz = 0

    e_inplane = eps_inplane - (tr_eps / 3.0) * ufl.Identity(len(uh))
    e_zz = -tr_eps / 3.0          # = eps_zz - tr/3, with eps_zz = 0

    equivalent_strain = ufl.sqrt(2.0 / 3.0 * (ufl.inner(e_inplane, e_inplane) + e_zz**2))

    v_st = ufl.TestFunction(V_dg0)
    u_st = ufl.TrialFunction(V_dg0)
    a_st = ufl.inner(u_st, v_st) * ufl.dx
    L_st = ufl.inner(equivalent_strain, v_st) * ufl.dx
    problem_st = petsc.LinearProblem(a_st, L_st, u=strains)
    problem_st.solve()

    return strains