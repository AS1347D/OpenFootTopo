# Segments the given foot image and saved the segmented file into GeomDir folder
python3 segmentation.py

# Extracts coordinates and prepares them for meshing
python3 coordinateExtractor.py

# Meshing
python3 mesh.py -nopopup

