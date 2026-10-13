import cadquery as cq

# 1. Load the mesh
# Note: This is computationally expensive for high-poly TOP opt results!
mesh = cq.importers.importStl("../printing/smoothedFull.stl")

# 2. Convert to a solid (This "manicurizes" the triangles into a shape)
# This might fail if the mesh is too complex or has "holes"
solid = mesh.combine()

# 3. Fillet
# This will try to find edges based on an angle tolerance
result = solid.edges().fillet(0.5)

cq.exporters.export(result, "filleted_model.step")