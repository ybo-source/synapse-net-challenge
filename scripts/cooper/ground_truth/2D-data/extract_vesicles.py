import os
from glob import glob
from pathlib import Path

import napari
import numpy as np
from elf.io import open_file
from magicgui import magicgui
from synapse_net.imod.export import export_point_annotations

EXPORT_FOLDER = "./exported"


def export_vesicles(mrc, mod):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)

    fname = Path(mrc).stem
    output_path = os.path.join(EXPORT_FOLDER, f"{fname}.h5")
    if os.path.exists(output_path):
        return

    resolution = 0.592
    with open_file(mrc, "r") as f:
        data = f["data"][:]

    segmentation, labels, label_names = export_point_annotations(
        mod, shape=data.shape, resolution=resolution, exclude_labels=[7, 14]
    )
    data, segmentation = data[0], segmentation[0]

    with open_file(output_path, "a") as f:
        f.create_dataset("data", data=data, compression="gzip")
        f.create_dataset("labels/vesicles", data=segmentation, compression="gzip")


def export_all_vesicles():
    mrc_files = sorted(glob(os.path.join("./data/*.mrc")))
    mod_files = sorted(glob(os.path.join("./data/*.mod")))
    for mrc, mod in zip(mrc_files, mod_files):
        export_vesicles(mrc, mod)


def create_mask(file_path):
    with open_file(file_path, "r") as f:
        if "labels/mask" in f:
            return

        data = f["data"][:]
        vesicles = f["labels/vesicles"][:]

    mask = np.zeros_like(vesicles)

    v = napari.Viewer()
    v.add_image(data)
    v.add_labels(vesicles)
    v.add_labels(mask)

    @magicgui(call_button="Save Mask")
    def save_mask(v: napari.Viewer):
        mask = v.layers["mask"].data.astype("uint8")
        with open_file(file_path, "a") as f:
            f.create_dataset("labels/mask", data=mask, compression="gzip")

    v.window.add_dock_widget(save_mask)
    napari.run()


def create_all_masks():
    files = sorted(glob(os.path.join(EXPORT_FOLDER, "*.h5")))
    for ff in files:
        create_mask(ff)


def main():
    export_all_vesicles()
    create_all_masks()


if __name__ == "__main__":
    main()
