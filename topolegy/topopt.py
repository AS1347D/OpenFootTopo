"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. topopt: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

import time

import numpy as np
from mpi4py import MPI

from topolegy.fem import form_fem
from topolegy.parameterize import DensityFilter, Heaviside
from topolegy.sensitivity import Sensitivity
from topolegy.optimize import optimality_criteria, mma_optimizer
from topolegy.utility import Communicator, Plotter
from topolegy.io_utils import calculate_stress, calculate_strain
from dolfinx import io
import os
import adios4dolfinx
from pathlib import Path

def topopt(fem, opt):
    """Main function for topology optimization."""

    # Initialization
    comm = MPI.COMM_WORLD
    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                   opt["filter_radius"], fem["petsc_options"])
    heaviside = Heaviside(rho_phys_field)
    sens_problem = Sensitivity(comm, opt, linear_problem, u_field, lambda_field, rho_phys_field)
    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])
    if comm.rank == 0:
        plotter = Plotter(fem["mesh_serial"])
    num_consts = 1 if opt["opt_compliance"] else 2
    num_elems = rho_field.x.array.size
    if not opt["use_oc"]:
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None

    # Apply passive zones
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    rho_ini = np.full(num_elems, opt["vol_frac"])
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.x.array[:] = rho_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01

    # Start topology optimization
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    
    if opt.get("results_dir"):
        RESULTS_DIR = opt['results_dir']
        
    else:
        RESULTS_DIR = "iterations"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    xdmf = io.XDMFFile(fem["mesh"].comm, RESULTS_DIR+"/optimized_design.xdmf", "w")
    xdmf.write_mesh(fem["mesh"])

    bs = u_field.function_space.dofmap.index_map_bs

    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:

        opt_start_time = time.perf_counter()
        opt_iter += 1

        # Density filter and Heaviside projection
        density_filter.forward()
        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2
            change = opt["opt_tol"] * 2
        heaviside.forward(beta)

        # Solve FEM
        linear_problem.solve_fem()

        # Compute function values and sensitivities
        [C_value, V_value, U_value], sensitivities = sens_problem.evaluate()
        heaviside.backward(sensitivities)
        [dCdrho, dVdrho, dUdrho] = density_filter.backward(sensitivities)
        if opt["opt_compliance"]:
            g_vec = np.array([V_value-opt["vol_frac"]])
            dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
        else:
            g_vec = np.array([V_value-opt["vol_frac"], C_value-opt["compliance_bound"]])
            dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])

        # Update the design variables
        rho_values = rho_field.x.petsc_vec.array.copy()
        if opt["opt_compliance"] and opt["use_oc"]:
            rho_new, change = optimality_criteria(
                rho_values, rho_min, rho_max, g_vec, dJdrho, dgdrho[0], opt["move"])
        else:
            rho_new, change, low, upp = mma_optimizer(
                num_consts, num_elems, opt_iter, rho_values, rho_min, rho_max,
                rho_old1, rho_old2, dJdrho, g_vec, dgdrho, low, upp, opt["move"])
            rho_old2 = rho_old1.copy()
            rho_old1 = rho_values.copy()
        rho_field.x.petsc_vec.array = rho_new.copy()

        xdmf.write_function(rho_phys_field, opt_iter)

        # print maximum displacement magnitude
        u_values = u_field.x.array.reshape(-1, bs)
        local_magnitude = np.linalg.norm(u_values, axis=1)
        
        if len(local_magnitude) > 0:
            local_max = np.max(local_magnitude)
        else:
            local_max = 0.0
            
        u_max = comm.allreduce(local_max, op=MPI.MAX)
        
        # Output the histories
        opt_time = time.perf_counter() - opt_start_time
        if comm.rank == 0:
            print(f"opt_iter: {opt_iter}, opt_time: {opt_time:.3g} (s), "
                  f"beta: {beta}, C: {C_value:.3f}, V: {V_value:.3f}, "
                  f"U: {U_value:.3f}, change: {change:.3f}, Max ||u||_2 = {u_max:.6f}", flush=True)

    values = S_comm.gather(rho_phys_field)
    
    # if comm.rank == 0:
    #     plotter.plot(values)
    
    xdmf.close()
    
    # For checkpointing
    path_out = Path("adios_dir/checkpoint.bp")
    adios4dolfinx.write_mesh(path_out, fem["mesh"], engine="BP4")
    adios4dolfinx.write_function(path_out, rho_phys_field, engine="BP4")

    # For visualization
    path_vis = Path("adios_dir/visualization.bp")
    with io.VTXWriter(MPI.COMM_WORLD, path_vis, rho_phys_field, "bp4") as vtx_file:
        vtx_file.write(0)
    
    # Final displacement field - u
    path_vis = Path("adios_dir/u_field.bp")
    with io.VTXWriter(MPI.COMM_WORLD, path_vis, u_field, "bp4") as vtx_file:
        vtx_file.write(opt_iter)

    # Kinematics
    E, nu = fem["young's modulus"], fem["poisson's ratio"]

    stresses = calculate_stress(fem["mesh"], u_field, E, nu)

    max_stress = fem["mesh"].comm.allreduce(np.max(stresses.x.array), op=MPI.MAX)
    print(f"Max Equivalent Stress = {max_stress:.6e}")

    vtx_s = io.VTXWriter(fem["mesh"].comm, "adios_dir/stresses.bp", stresses, engine="BP4")
    vtx_s.write(0)
    vtx_s.close()

    # # Calculate Strain Field

    strains = calculate_strain(fem["mesh"], u_field, E, nu)

    # To find the maximum strain in the field:
    max_strain = fem["mesh"].comm.allreduce(np.max(strains.x.array), op=MPI.MAX)
    print(f"Max Equivalent Strain = {max_strain:.6e}")
    
    vtx_strain = io.VTXWriter(fem["mesh"].comm, "adios_dir/strains.bp", strains, engine="BP4")
    vtx_strain.write(0)
    vtx_strain.close()
    
    if opt.get("save_log"):
        
        os.makedirs("results", exist_ok=True)
        filepath = f"results/{opt['save_log']}.csv"
        
        t_imap = fem["mesh"].topology.index_map(fem["mesh"].topology.dim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        N_cell = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM) #sum all cells and distribute to each process
        
        # 1. Check if the file is new to write the Header
        with open(filepath, "w") as f:
            header = (
                "opt_iter,E,beta,C_value,V_value,"
                "target_vol_frac,filter_radius,penalty,u_max,sigma_max,N_cell\n"
            )
            f.write(header)

        # 2. Append the data row
        with open(filepath, "a") as f:
            # Using :.6e for precision values like Compliance and Change
            # Using :.3f for time and volume
            f.write(
                f"""{opt_iter},{fem["young's modulus"]},{beta},{C_value:.6e},{V_value:.6f},"""
                f"{opt['vol_frac']},{opt['filter_radius']},{opt['penalty']},{u_max:.6f},{max_stress:.6f},{N_cell}\n"
            )
    

    # save_xdmf(fem["mesh"], rho_phys_field)
