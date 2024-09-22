import os
from glob import glob
from pathlib import Path

import napari
import numpy as np
from elf.io import open_file


# TODO convert this to a format Arsen can read better.
def main():
    input_root = "/home/pape/Work/data/fernandez-busnadiego/tomos_actin_18924"
    output_root = "./predictions"

    files = sorted(glob(os.path.join(input_root, "*.mrc")))
    for ff in files:
        fname = Path(ff).stem
        seg_path = os.path.join(output_root, f"{fname}.h5")

        with open_file(ff, "r") as f:
            raw = f["data"][:]

        with open_file(seg_path, "r") as f:
            pred = f["actin_pred"][:]
            seg = f["actin_seg"][:]

        v = napari.Viewer()
        v.add_image(raw)
        v.add_image(pred)
        v.add_labels(seg)
        v.title = fname
        napari.run()


main()
