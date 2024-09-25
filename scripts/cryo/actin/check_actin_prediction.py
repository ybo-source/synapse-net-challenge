import os
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import napari
from elf.io import open_file


def main(view):
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

        if view:
            v = napari.Viewer()
            v.add_image(raw)
            v.add_image(pred)
            v.add_labels(seg)
            v.title = fname
            napari.run()
        else:
            imageio.imwrite(os.path.join(output_root, f"{fname}.tif"), seg, compression="zlib")


if __name__ == "__main__":
    main(view=False)
