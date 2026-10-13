import gmsh
import sys

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

gmsh.model.add("surface2d")

# Import your STL surface
gmsh.merge("printing/surface2d.stl")

# --- Classify geometry (important!)
angle = 45 * 3.14159 / 180.0
gmsh.model.mesh.classifySurfaces(
    angle,
    boundary=True,
    forReparametrization=True,
    curveAngle=180 * 3.14159 / 180.0
)

gmsh.model.mesh.createGeometry()

# --- Set mesh size controls
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)

# --- Use high-quality meshing
gmsh.option.setNumber("Mesh.Algorithm", 6)          # Frontal-Delaunay
gmsh.option.setNumber("Mesh.Smoothing", 20)         # Laplacian smoothing
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

# --- Generate 2D mesh
gmsh.model.mesh.generate(2)
# gmsh.model.mesh.optimize("Laplace2D", force=True,niter=100)

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

# Save
# gmsh.write("surface2d_remeshed.msh")
# gmsh.write("surface2d_remeshed.stl")

gmsh.finalize()
