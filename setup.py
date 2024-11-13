import runpy
from setuptools import setup, find_packages

__version__ = runpy.run_path("synaptic_reconstruction/__version__.py")["__version__"]


setup(
    name="synaptic_reconstruction",
    packages=find_packages(exclude=["test"]),
    version=__version__,
    author="Constantin Pape; Sarah Muth",
    url="https://github.com/computational-cell-analytics/synaptic_reconstruction",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sr_tools.correct_segmentation = synaptic_reconstruction.tools.segmentation_correction:main",
            "sr_tools.measure_distances = synaptic_reconstruction.tools.distance_measurement:main",
        ],
        "napari.manifest": [
            "synaptic_reconstruction = synaptic_reconstruction:napari.yaml",
        ],
    },
)
