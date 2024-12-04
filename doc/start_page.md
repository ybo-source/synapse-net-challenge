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


## Requirements & Installation

SynapseNet was developed and tested on Linux. It should be possible to install and use it on Mac or Windows, but we have not tested this.
Furthermore, SynapseNet requires a GPU for segmentation of 3D volumes.

You need a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) installation. Follow the instruction at the respective links if you have installed neither. We assume you have `conda` for the rest of the instructions. After installing it, you can use the `conda` command.

To install it you should follow these steps:
- First, download the SynapseNet repository via
```bash
git clone https://github.com/computational-cell-analytics/synapse-net
```
- Then, enter the `synapse-net` folder:
```bash
cd synapse-net
```
- Now you can install the environment for SynapseNet with `conda` from the environment file we proved:
```bash
conda env create -f environment.yaml
```
- You will need to confirm this step. It will take a while. Afterwards you can activate the environment:
```bash
conda activate synapse-net
```
- Finally, install SynapseNet itself into the environment:
```bash
pip install -e .
```

Now you can use all SynapseNet features. From now on, just activate the environment via
```
conda activate synapse-net
```
to use them.

> Note: If you use `mamba` instead of conda just replace `conda` in the commands above with `mamba`.

> Note: We also provide an environment for a CPU version of SynapseNet. You can install it by replacing `environment.yaml` with `environment_cpu.yaml` in the respective command above. This version can be used for 2D vesicle segmentation, but it does not work for 3D segmentation.

> Note: If you have issues with the CUDA version then install a PyTorch that matches your nvidia drivers. See [pytorch.org](https://pytorch.org/) for details.


## Napari Plugin

**The rest of the documentation will be updated in the next days!**


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
