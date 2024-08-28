# Segmentation of Synaptic Structures in RT EM Tomography

This folder contains functionality for segmenting and visualizing structures in EM tomograms.
It currently holds the following scripts:
- `run_vesicle_segmentation.py`: Segment vesicles from tomograms stored as mrc.
- `run_mitochondria_segmentation.py`: Segment mitochondria from tomograms stored as mrc.
- `run_cristae_segmentation.py`: Segment cristae from tomograms stored as mrc.
- `visualize_segmentation.py`: Visualize segmentation results with napari.
- `export_vesicles_to_imod.py`: Export the vesicle segmentation to IMOD point annotations.

## Usage

Before running any of the scripts you need to activate the python environment by running
```
$ micromamba activate synaptic-reconstruction
```

The segmentation scripts (`run_..._segmentation.py`) all work similarly and can either run segmentation for a single mrc file or for all mrcs in a folder structure.
For example, you can run vesicle segmentation like this:
```
$ python run_vesicle_segmentation.py -i /path/to/input_folder -o /path/to/output_folder -m /path/to/vesicle_model.pt
```
The filepath after `-i` specifices the location of the folder with the mrcs to be segmented, the segmentation results will be stored (as tifs) in the folder following `-o` and `-m` is used to specify the path to the segmentation model.

The segmentation scripts accept additional parameters, e.g. `--force` to overwrite existing segmentations in the output folder (by default these are skipped to avoid unnecessary computation) and `--tile_shape <TILE_Z> <TILE_Y> <TILE_X>` to specify a different tile shape (which may be necessary to avoid running out of GPU memory).
You can check out all parameters by printing the help message of the give string, e.g.
```
$ python run_vesicle_segmentation.py -h
```

To check segmentation results you can use the visualization script like this:
```
$ python visualize_segmentation.py -i /path/to/input_folder -s /path/to/vesicles /path/to/mitos -n vesicles mitos
```
This command will open all tomograms saved under `/path/to/input_folder` and the associated segmentations in the folders specified after `-s` (here vesicle and mito segmentations). The option `-n` is used to set the correct layer names for the segmentations in napari.

The script `export_vesicles_to_imod.py` can be used to export the vesicle segmentations to an imod point model.
You can run it like this:
```
$ python -i /path/to/input_folder -s /path/to/vesicles -o /path/to/imod_export --min_radius 10
```
The parameters have a similar meaning to the other scripts (`-i`: folder with mrc files, `-s`: folder with vesicle segmentation results, `-o`: folder for saving the imod export results); `--min_radius` controls the minimal vesicle radius in nanometer.
Note: export to imod is not yet implemented for other structures than vesicles.


## Data Transfer

We now have storage space to exchange tomogram data etc.
You can copy data over there with the following command:
```
$ rsync -avz --delete -e ssh /path/to/tomogram_folder constantin@sfb1286.ims.bio:/data/share/<NAME>
```
Here, `/path/to/tomogram_folder` is the folder with the tomograms (or any other data) you want to copy over.
Replace `<NAME>` with the name for storing the data on the server. This name can also be a nested folder.
E.g. use `all_tomograms/new` to store the data you copy in a nested folder on the server.

You will be prompted to enter a password after running the `rsync` command. You can find the password in the file `pwd.txt` in this folder.
