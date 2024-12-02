# SynapseNet: Deep Learning for Automatic Synapse Reconstruction

SynapseNet is a tool for automatic segmentation and analysis of synapses in electron micrographs.
It provides deep neural networks for:
- Synaptic vesicle segmentation in ssTEM (2d data) and (cryo-)electron tomography (3d data)
- Active zone membrane segmentation in electron tomography
- Mitochondrion segmentation in electron tomography
- Synaptic compartment segmentation in electron tomography
- Synaptic ribbon and pre-synaptic density segmentation for ribbon synapses in electron tomography
It also offers functionality for quantifying synaptic ultrastructure based on segmentation results, for example by measuring vesicle or structure morphology, measuring distances between vesicles and structures, or assigning vesicles into different pools.
SynapseNet mainly targets electron tomography, but can also be appled to other types of electron microscopy,
especially throught the [domain adaptation](domain-adaptation) functionality.

SynapseNet offers a [napari plugin](napari-plugin), [command line interface](command-line-interface), and [python library](python-library).
Please cite our [bioRxiv preprint](TODO) if you use it in your research.

**The rest of the documentation will be updated in the next days!**

## Requirements & Installation

- Requirements: Tested on Linux but should work on Mac/Windows.
    - GPU needed to use 3d segmentation networks
- Installation via conda and local pip install
    - GPU support

- Make sure conda or mamba is installed.
    - If you don't have a conda installation yet we recommend [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- Create the environment with all required dependencies: `mamba env create -f environment.yaml`
- Activate the environment: `mamba activate synaptic-reconstruction`
- Install the package: `pip install -e .`

## Napari Plugin

lorem ipsum

## Command Line Functionality

- segmentation cli
- export to imod
    - vesicles / spheres
    - objects

## Python Library

- segmentation functions
- distance and morphology measurements
- imod

### Domain Adaptation

- explain domain adaptation
- link to the example script

### Network Training

- explain / diff to domain adaptation
- link to the example script
