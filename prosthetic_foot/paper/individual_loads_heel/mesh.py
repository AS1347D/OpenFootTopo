import pandas as pd
import gmsh
import sys
import os
import numpy as np
from pathlib import Path


def isoFoot(lc=None, mesh_name="foot"):

    path = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = "/MeshDir/"
    COORDS_DIR = "GeomDir/coordinates"
    filename = "segmented_sach_flipped_edge_pixels.csv"  # Replace with your actual file name
    file_path = os.path.join(COORDS_DIR, filename)
    mesh_name = mesh_name

    df = pd.read_csv(file_path)
    outer = df.values

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(__name__)

    outer_points = []
    for outer_point in outer:
        outer_points.append(gmsh.model.occ.addPoint(outer_point[0], outer_point[1], 0))

    gmsh.model.occ.synchronize()

    outer_lines = []
    for i in range(len(outer_points)):
        if i == len(outer_points)-2:
            outer_lines.append(gmsh.model.occ.addLine(outer_points[i], outer_points[0]))
            break
        else:
            outer_lines.append(gmsh.model.occ.addLine(outer_points[i], outer_points[i+1]))

    outer_curve = gmsh.model.occ.addCurveLoop(outer_lines)

    gmsh.model.occ.synchronize()

    base = gmsh.model.occ.addPlaneSurface([outer_curve])
    gmsh.model.occ.synchronize()

    extension = gmsh.model.occ.addRectangle(138, 96, 0, 106, 24)
    gmsh.model.occ.synchronize()

    fused = gmsh.model.occ.fuse([(2, base)], [(2, extension)])
    gmsh.model.occ.synchronize()

    # # Physical tags
    line_tags = gmsh.model.getEntities(dim=1)
    for line in line_tags:
        gmsh.model.addPhysicalGroup(1, [line[1]], tag=line[1])

    gmsh.model.addPhysicalGroup(2, [fused[0][0][1]], tag=1)  # foot domain tag

    if lc:
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)

    gmsh.option.setNumber("Mesh.Optimize", 1)

    gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.refine()

    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    gmsh.write("{}.msh".format(path + mesh_dir + mesh_name))

    gmsh.finalize()

    return gmsh.model


if __name__ == "__main__":
    # isoFoot(lc=None)
    isoFoot(lc=1, mesh_name="foot")

    import adios4dolfinx
    from dolfinx.io.gmshio import read_from_msh
    from mpi4py import MPI

    pathMesh = Path("MeshDir/foot.msh")

    mesh ,_, _ = read_from_msh(pathMesh, MPI.COMM_WORLD, gdim=2)

    path = Path("adios_dir/mesh.bp")
    adios4dolfinx.write_mesh(path, mesh, engine='BP4')