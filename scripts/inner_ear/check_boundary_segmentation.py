import os

import imageio.v3 as imageio
import napari

from elf.io import open_file
from synaptic_reconstruction.structures.boundaries import segment_boundary_next_to_pd


def check_boundary_segmentation(tomo, boundary_pred, pd_segmentation):

    boundary_segmentation = segment_boundary_next_to_pd(boundary_pred, pd_segmentation, n_slices_exclude=10)

    v = napari.Viewer()
    v.add_image(tomo)
    v.add_labels(boundary_pred)
    v.add_labels(boundary_segmentation)
    v.add_labels(pd_segmentation)
    napari.run()

    boundary_segmentation = v.layers["boundary_segmentation"].data

    return boundary_segmentation


def main():
    raw_root = "/home/pape/Work/data/moser/em-susi/04_wild_type_strong_stimulation/NichtAnnotiert"

    for root, dirs, files in os.walk(raw_root):
        dirs.sort()

        for ff in files:
            raw_path = os.path.join(root, ff)
            if not raw_path.endswith(".rec"):
                continue

            print("Checking", raw_path)
            save_path_boundary = raw_path.replace(".rec", "_boundary.tif")
            if os.path.exists(save_path_boundary):
                print("Boundary segmentation is already there, skip!")
                continue

            fname = os.path.basename(raw_path)
            pred_name = fname.replace(".rec", "_MemBrain_seg_v10_alpha.ckpt_segmented.mrc")
            boundary_pred_path = os.path.join("./predictions", pred_name)
            assert os.path.exists(boundary_pred_path), boundary_pred_path
            with open_file(boundary_pred_path, "r") as f:
                boundary_pred = f["data"][:]

            with open_file(raw_path, "r") as f:
                tomo = f["data"][:]

            save_path_pd = raw_path.replace(".rec", "_pd.tif")
            assert os.path.exists(save_path_pd)
            pd_segmentation = imageio.imread(save_path_pd)
            if pd_segmentation.sum() == 0:
                print("Did not find a presynapyic density, skipping!")
                boundary_segmentation = boundary_pred
            else:
                boundary_segmentation = check_boundary_segmentation(tomo, boundary_pred, pd_segmentation)

            imageio.imwrite(save_path_boundary, boundary_segmentation, compression="zlib")


if __name__ == "__main__":
    main()
