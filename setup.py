from setuptools import setup

setup(
    name='topolegy',
    version = '0.0.0',
    author='Ekrem Ekici',
    author_email='ekrem.ekici@bogazici.edu.tr',
    packages=['topolegy'],
    install_requires=[
        'pyvista',
        'scipy',
        'gmsh',
        'meshio',
        'h5py',
        'pandas',
        'ipython',
        'svgpathtools',
        'shapely',
        'trimesh',
        'mapbox-earcut',
        'pyglet==1.5.31',
        # 'cadquery',
        # 'cadquery-ocp==7.9.3.0'
    ]
)
