import os
from glob import glob

import h5py
import napari
import numpy as np
from magicgui import magicgui

from skimage.segmentation import watershed


INPUT_ROOT = "/home/pape/Work/data/fernandez-busnadiego/vesicle_gt/v2"
OUTPUT_ROOT = "/home/pape/Work/data/fernandez-busnadiego/vesicle_gt/v3"
PRED_ROOT = "./prediction"


def update_vesicle_gt(path, pred_path):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    fname = os.path.basename(path)
    out_path = os.path.join(OUTPUT_ROOT, fname)

    if os.path.exists(out_path):
        return

    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        gt = f["labels/vesicles"][:]
        mask = f["labels/mask"][:]

    with h5py.File(pred_path, "r") as f:
        # pred = f["/vesicles/segment_from_DA_cryo_v2_masked"][:]
        fg = f["/prediction_DA_cryo_v2_masked/foreground"][:]
        bd = f["/prediction_DA_cryo_v2_masked/boundaries"][:]

    print("Run watershed")
    ws_mask = np.logical_or(fg > 0.5, gt != 0)
    updated_vesicles = watershed(bd, markers=gt, mask=ws_mask)
    updated_vesicles[mask == 0] = 0
    print("done")

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(fg, visible=False)
    v.add_image(bd, visible=False)
    v.add_labels(gt, name="ground-truth")
    v.add_labels(updated_vesicles, name="updated-gt")
    v.add_labels(mask, visible=False, name="mask")

    @magicgui(call_button="Save Vesicles")
    def save_vesicles():
        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/vesicles", data=updated_vesicles, compression="gzip")
            f.create_dataset("labels/mask", data=mask, compression="gzip")

    v.window.add_dock_widget(save_vesicles)

    napari.run()


def main():
    paths = glob(os.path.join(INPUT_ROOT, "*.h5"))
    for path in paths:
        fname = os.path.basename(path)
        pred_path = os.path.join(PRED_ROOT, fname)
        assert os.path.exists(pred_path)
        update_vesicle_gt(path, pred_path)


if __name__ == "__main__":
    main()
