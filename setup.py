import runpy
from setuptools import setup, find_packages

__version__ = runpy.run_path("synaptic_reconstruction/__version__.py")["__version__"]


setup(
    name="synaptic_reconstruction",
    packages=find_packages(exclude=["test"]),
    version=__version__,
    author="Constantin Pape; Sarah Muth; Luca Freckmann",
    url="https://github.com/computational-cell-analytics/synaptic_reconstruction",
    license="MIT",
    entry_points={
        "console_scripts": [
            # TODO add segmentation CLI
        ],
        "napari.manifest": [
            "synaptic_reconstruction = synaptic_reconstruction:napari.yaml",
        ],
    },
)
