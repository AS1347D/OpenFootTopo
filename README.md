# open-topolegy

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An open-source topology optimization framework for ISO 10328-compliant, image-driven prosthetic foot design, built on [FEniCSx](https://fenicsproject.org/).

**open-topolegy is built on top of [FEniTop](https://github.com/missionlab/fenitop)**, the FEniCSx topology optimization implementation by Jia et al. We use FEniTop as the optimization backbone and extend it for prosthetic foot design.

## Overview

open-topolegy accompanies the paper:

> Ekici, E., & Dalaq, A. "An open-source topology optimization framework for ISO 10328-compliant, image-driven prosthetic foot design." *International Journal for Numerical Methods in Biomedical Engineering*. DOI: [10.1002/cnm.70222](https://onlinelibrary.wiley.com/doi/10.1002/cnm.70222)

The framework:

- uses FEniCSx (DOLFINx) with FEniTop for linear-elastic, SIMP-based topology optimization;
- derives the design domain anatomically from images (with an optional segmentation step);
- applies ISO 10328 boundary conditions;
- produces a universal nondimensional design curve (CEL_f/F²) for patient-specific design selection;
- supports forward hyperelastic verification of the final topology.

## Acknowledgement of FEniTop

This project relies on FEniTop. If you use open-topolegy, please also cite FEniTop:

> Jia, Y., Wang, C., & Zhang, X. S. "FEniTop: a simple FEniCSx implementation for 2D and 3D topology optimization supporting parallel computing." *Structural and Multidisciplinary Optimization* (2024). DOI: [10.1007/s00158-024-03818-7](https://link.springer.com/article/10.1007/s00158-024-03818-7)

Please refer to the FEniTop repository for its own license terms.

## Installation

open-topolegy relies on the FEniCSx coding environment and a few other dependencies. To install the environment, follow the procedure below.

### Conda installation

```bash
sudo apt install libgl1-mesa-glx xvfb
conda create -n open-topolegy python=3.13.0 -y
# conda create -n topolegy python=3.11.0 -y # for laptop ubuntu 22.04
conda activate open-topolegy
conda install -c conda-forge fenics-dolfinx=0.9.0 pyvista=0.44.1 vtk=9.5.2 adios4dolfinx=0.9.0 -y # Linux and macOS
pip3 install -e .
```

### Segmentation algorithm (GPU)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx pandas
```

### Segmentation algorithm (CPU only)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx pandas
```

## Usage

The `prosthetic_foot/paper` folder contains the data from the published paper, so the reported results can be reproduced and inspected.

## Citation

If you use open-topolegy in your work, please cite the paper above and FEniTop. A BibTeX entry for the paper:

```bibtex
@article{ekici_dalaq_openfoot,
  author  = {Ekici, Ekrem and Dalaq, Ahmed},
  title   = {An open-source topology optimization framework for ISO 10328-compliant, image-driven prosthetic foot design},
  journal = {International Journal for Numerical Methods in Biomedical Engineering},
  doi     = {10.1002/cnm.70222}
}
```

## License

open-topolegy is released under the [Apache License 2.0](LICENSE). FEniTop and the other dependencies remain under their own licenses.

## TODO

- open-topolegy currenctly uses dolfinx v0.9.0. As of October 2026, v0.11.0 exists. Upgrade is needed.
- Parallelization is TODO.