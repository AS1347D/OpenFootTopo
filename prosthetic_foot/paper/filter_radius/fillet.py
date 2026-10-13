import gmsh
import sys

gmsh.initialize()
gmsh.model.add("ProstheticFoot")

# 1. Load the STL
gmsh.merge("printing/smoothedFull.stl")

# 2. Classify surfaces to create topological edges
# Angle is the threshold to detect an "edge"
angle = 40 
gmsh.model.mesh.classifySurfaces(angle * 3.14159 / 180., True, True)
gmsh.model.mesh.createGeometry()

# 3. Create a volume from the surfaces
surfaces = gmsh.model.occ.getEntities(2)
loop = gmsh.model.occ.addSurfaceLoop([s[1] for s in surfaces])
volume = gmsh.model.occ.addVolume([loop])

out = [(3,volume)]

gmsh.model.occ.synchronize()

e = gmsh.model.getBoundary(gmsh.model.getBoundary(out), False)
gmsh.model.occ.fillet([out[0][1]], [abs(i[1]) for i in e], [5])
gmsh.model.occ.synchronize()


if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

# 5. Re-mesh for dolfinx
# gmsh.model.mesh.generate(3)
# gmsh.write("filleted_foot.msh")
gmsh.finalize()