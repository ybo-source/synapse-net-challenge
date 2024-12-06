import os
from pathlib import Path

import imageio.v3 as imageio
import napari
from synapse_net.file_utils import get_data_path
from elf.io import open_file


tomos_with_outlier = [
    "WT strong stim/Mouse 1/modiolar/11",
    "WT strong stim/Mouse 1/pillar/3",
    "WT strong stim/Mouse 2/modiolar/2",
    "WT strong stim/Mouse 2/modiolar/2",
    "WT strong stim/Mouse 2/modiolar/3",
    "WT control/Mouse 1/modiolar/3",
    "WT control/Mouse 1/modiolar/3",
    "WT control/Mouse 2/modiolar/4",
]
root = "/home/pape/Work/data/moser/em-synapses/Electron-Microscopy-Susi/Analyse"

for tomo in tomos_with_outlier:
    folder = os.path.join(root, tomo)

    result_folder = os.path.join(folder, "manuell")
    if not os.path.exists(result_folder):
        result_folder = os.path.join(folder, "Manuell")
    if not os.path.exists(result_folder):
        continue

    raw_path = get_data_path(folder)
    with open_file(raw_path, "r") as f:
        raw = f["data"][:]

    fname = Path(raw_path).stem
    man_path = os.path.join(result_folder, "Vesikel.tif")
    seg = imageio.imread(man_path)

    az_path = os.path.join(result_folder, "Membrane.tif")
    az_seg = imageio.imread(az_path)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg)
    v.add_labels(az_seg)
    v.title = tomo
    napari.run()
