import os

import imageio.v3 as imageio
import napari

from elf.io import open_file
from synaptic_reconstruction.structures import segment_ribbon


def check_ribbon_segmentation(tomo, ribbon_pred, vesicles):

    ribbon_segmentation = segment_ribbon(ribbon_pred, vesicles, n_slices_exclude=18)

    v = napari.Viewer()
    v.add_image(tomo)
    v.add_image(ribbon_pred)
    v.add_labels(vesicles)
    v.add_labels(ribbon_segmentation)

    napari.run()

    ribbon_segmentation = v.layers["ribbon_segmentation"].data

    return ribbon_segmentation


def main():
    raw_root = "/home/pape/Work/data/moser/em-susi/04_wild_type_strong_stimulation/NichtAnnotiert"
    vesicle_seg_root = "/home/pape/Work/data/moser/em-susi/results/vesicles/v1/segmentations/NichtAnnotiert"
    seg_root = "/home/pape/Work/data/moser/em-susi/results/synaptic_structures/v4/segmentations/NichtAnnotiert"

    for root, dirs, files in os.walk(raw_root):
        dirs.sort()

        for ff in files:
            raw_path = os.path.join(root, ff)
            if not raw_path.endswith(".rec"):
                continue

            print("Checking", raw_path)
            fname = os.path.relpath(raw_path, raw_root)

            save_path_ribbon = raw_path.replace(".rec", "_ribbon.tif")
            save_path_vesicles = raw_path.replace(".rec", "_vesicles.tif")
            if os.path.exists(save_path_ribbon):
                print("Ribbon segmentation is already there, skip!")
                continue

            fname = fname.replace(".rec", ".h5")
            vesicle_path = os.path.join(vesicle_seg_root, fname)
            seg_path = os.path.join(seg_root, fname)

            assert os.path.exists(vesicle_path)
            assert os.path.exists(seg_path), seg_path

            with open_file(raw_path, "r") as f:
                tomo = f["data"][:]

            with open_file(vesicle_path, "r") as f:
                vesicles = f["seg"][:]

            with open_file(seg_path, "r") as f:
                ribbon_pred = f["seg"][0]

            ribbon_segmentation = check_ribbon_segmentation(tomo, ribbon_pred, vesicles)

            imageio.imwrite(save_path_ribbon, ribbon_segmentation, compression="zlib")
            imageio.imwrite(save_path_vesicles, vesicles, compression="zlib")


if __name__ == "__main__":
    main()
