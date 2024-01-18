import os

import imageio.v3 as imageio
import napari

from elf.io import open_file
from synaptic_reconstruction.structures import segment_presynaptic_density


def check_presyn_segmentation(tomo, presyn_pred, ribbon_segmentation):

    presyn_segmentation = segment_presynaptic_density(presyn_pred, ribbon_segmentation, n_slices_exclude=15)

    v = napari.Viewer()
    v.add_image(tomo)
    v.add_image(presyn_pred)
    v.add_labels(ribbon_segmentation)
    v.add_labels(presyn_segmentation)
    napari.run()

    presyn_segmentation = v.layers["presyn_segmentation"].data
    return presyn_segmentation


def main():
    raw_root = "/home/pape/Work/data/moser/em-susi/04_wild_type_strong_stimulation/NichtAnnotiert"
    seg_root = "/home/pape/Work/data/moser/em-susi/results/synaptic_structures/v2/segmentations/NichtAnnotiert"

    for root, dirs, files in os.walk(raw_root):
        dirs.sort()

        for ff in files:
            raw_path = os.path.join(root, ff)
            if not raw_path.endswith(".rec"):
                continue

            print("Checking", raw_path)
            fname = os.path.relpath(raw_path, raw_root)

            save_path_pd = raw_path.replace(".rec", "_pd.tif")
            if os.path.exists(save_path_pd):
                print("PD segmentation is already there, skip!")
                continue

            fname = fname.replace(".rec", ".h5")
            seg_path = os.path.join(seg_root, fname)
            ribbon_seg_path = raw_path.replace(".rec", "_ribbon.tif")

            assert os.path.exists(ribbon_seg_path)
            assert os.path.exists(seg_path)

            with open_file(raw_path, "r") as f:
                tomo = f["data"][:]

            with open_file(seg_path, "r") as f:
                presyn_pred = f["seg"][1]

            ribbon_seg = imageio.imread(ribbon_seg_path)
            pre_segmentation = check_presyn_segmentation(tomo, presyn_pred, ribbon_seg)

            imageio.imwrite(save_path_pd, pre_segmentation, compression="zlib")


if __name__ == "__main__":
    main()
