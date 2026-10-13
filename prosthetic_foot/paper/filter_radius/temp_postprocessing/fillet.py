# ------------------------------------------------------------------------------
#
#  Gmsh Python tutorial 19
#
#  Thrusections, fillets, pipes, mesh size from curvature
#
# ------------------------------------------------------------------------------

# The OpenCASCADE geometry kernel supports several useful features for solid
# modelling.

import gmsh
import math
import os
import sys

gmsh.initialize()

gmsh.model.add("t19")
gmsh.option.setNumber("General.Terminal", 0)


# Volumes can be constructed from (closed) curve loops thanks to the
# `addThruSections()' function
gmsh.model.occ.addCircle(0, 0, 0, 0.5, 1)
gmsh.model.occ.addCurveLoop([1], 1)
gmsh.model.occ.addCircle(0.1, 0.05, 1, 0.1, 2)
gmsh.model.occ.addCurveLoop([2], 2)
gmsh.model.occ.addCircle(-0.1, -0.1, 2, 0.3, 3)
gmsh.model.occ.addCurveLoop([3], 3)
gmsh.model.occ.addThruSections([1, 2, 3], 1)
gmsh.model.occ.synchronize()



# We copy the first volume, and fillet all its edges:
out = gmsh.model.occ.copy([(3, 1)])
gmsh.model.occ.translate(out, 4, 0, 0)
gmsh.model.occ.synchronize()
e = gmsh.model.getBoundary(gmsh.model.getBoundary(out), False)
gmsh.model.occ.fillet([out[0][1]], [abs(i[1]) for i in e], [0.1])
gmsh.model.occ.synchronize()

print(out)

gmsh.model.occ.synchronize()

# We can activate the calculation of mesh element sizes based on curvature
# (here with a target of 20 elements per 2*Pi radians):
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

# We can constraint the min and max element sizes to stay within reasonnable
# values (see `t10.py' for more details):
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)

gmsh.model.mesh.generate(3)
# gmsh.write("t19.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
