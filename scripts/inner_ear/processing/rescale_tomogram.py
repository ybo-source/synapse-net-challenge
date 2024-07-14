import os
from glob import glob

import h5py
import mrcfile
from skimage.transform import rescale

from synaptic_reconstruction.file_utils import get_data_path
from parse_table import get_data_root


def rescale_tomogram(folder, factor=2):
    tomo_path = get_data_path(folder)
    with mrcfile.open(tomo_path) as f:
        tomo = f.data[:]
        voxel_size = f.voxel_size.tolist()
    scale_factor = 1.0 / factor

    tomo = rescale(tomo, scale_factor, order=3, preserve_range=True).astype(tomo.dtype)
    voxel_size = [vx * factor for vx in voxel_size]
    mrcfile.write(tomo_path, tomo, voxel_size=voxel_size, overwrite=True)
    print("Rescaled to", tomo.shape, voxel_size)

    # rescale the segmentations
    segmentation_files = glob(os.path.join(folder, "automatisch", "v2", "*.h5"))
    for seg_file in segmentation_files:
        with h5py.File(seg_file, "r") as f:
            seg = f["segmentation"][:]

            if seg.shape[0] <= tomo.shape[0]:
                continue
            print("Rescaling", seg_file, "starting from", seg.shape)

            if "prediction" in f:
                pred = f["prediction"][:]
            else:
                pred = None

        seg = rescale(seg, scale_factor, order=0, anti_aliasing=False, preserve_range=True).astype(seg.dtype)
        assert seg.shape == tomo.shape, f"{seg.shape}, {tomo.shape}"
        if pred is not None:
            pred = rescale(pred, scale_factor, order=0, anti_aliasing=False, preserve_range=True).astype(pred.dtype)
            assert pred.shape == tomo.shape

        with h5py.File(seg_file, "w") as f:
            f.create_dataset("segmentation", data=seg, compression="gzip")
            if pred is not None:
                f.create_dataset("prediction", data=pred, compression="gzip")


def check_rescaling(folder):
    tomo_path = get_data_path(folder)
    with mrcfile.open(tomo_path) as f:
        tomo_shape = f.data.shape
    print("Tomo shape:", tomo_shape)

    # rescale the segmentations
    segmentation_files = glob(os.path.join(folder, "automatisch", "v2", "*.h5"))
    for seg_file in segmentation_files:
        with h5py.File(seg_file, "r") as f:
            seg_shape = f["segmentation"].shape
        print(seg_file)
        print(seg_shape)


def rescale_all():
    root = get_data_root()

    folder1 = os.path.join(root, "Electron-Microscopy-Susi/Analyse/WT control/Mouse 1/pillar/3")
    rescale_tomogram(folder1)

    folder2 = os.path.join(root, "Electron-Microscopy-Susi/Analyse/WT control/Mouse 1/pillar/4")
    rescale_tomogram(folder2)


def check_all():
    root = get_data_root()

    folder1 = os.path.join(root, "Electron-Microscopy-Susi/Analyse/WT control/Mouse 1/pillar/3")
    check_rescaling(folder1)

    folder2 = os.path.join(root, "Electron-Microscopy-Susi/Analyse/WT control/Mouse 1/pillar/4")
    check_rescaling(folder2)



def main():
    # rescale_all()
    check_all()


if __name__ == "__main__":
    main()
