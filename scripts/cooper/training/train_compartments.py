import os
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split

from skimage import img_as_ubyte
from skimage.segmentation import find_boundaries
from skimage.filters import gaussian, rank
from skimage.morphology import disk
from scipy.ndimage import binary_dilation

from synaptic_reconstruction.training import supervised_training

# TRAIN_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2"
TRAIN_ROOT = "/home/pape/Work/my_projects/synaptic-reconstruction/scripts/cooper/ground_truth/compartments/output/compartment_gt"  # noqa


def get_paths_2d():
    paths = sorted(glob(os.path.join(TRAIN_ROOT, "v1", "**", "*.h5"), recursive=True))
    return paths


def get_paths_3d():
    paths = sorted(glob(os.path.join(TRAIN_ROOT, "v2", "**", "*.h5"), recursive=True))
    return paths


def label_transform_2d(seg):
    boundaries = find_boundaries(seg).astype("float32")
    boundaries = gaussian(boundaries, sigma=1.0)
    boundaries = rank.autolevel(img_as_ubyte(boundaries), disk(8)).astype("float") / 255
    mask = binary_dilation(seg != 0, iterations=8)
    return np.stack([boundaries, mask])


def label_transform_3d(seg):
    output = np.zeros((2,) + seg.shape, dtype="float32")
    for z in range(seg.shape[0]):
        out = label_transform_2d(seg[z])
        output[:, z] = out
    return output


def train_compartments_2d_v1():
    paths = get_paths_2d()
    train_paths, val_paths = train_test_split(paths, test_size=0.15, random_state=42)

    patch_shape = (384, 384)
    batch_size = 4

    check = False

    save_root = "."
    supervised_training(
        name="compartment_model_2d/v1",
        train_paths=train_paths,
        val_paths=val_paths,
        raw_key="data",
        label_key="/labels/compartments",
        patch_shape=patch_shape, batch_size=batch_size,
        check=check,
        save_root=save_root,
        label_transform=label_transform_2d,
        mask_channel=True,
        out_channels=1,
        n_samples_train=400,
        n_samples_val=40,
        n_iterations=int(2e4),
    )


def train_compartments_3d_v1():
    paths = get_paths_3d()
    train_paths, val_paths = train_test_split(paths, test_size=0.15, random_state=42)

    patch_shape = (64, 384, 384)
    batch_size = 1

    check = True

    save_root = "."
    supervised_training(
        name="compartment_model_3d/v1",
        train_paths=train_paths,
        val_paths=val_paths,
        label_key="/labels/compartments",
        patch_shape=patch_shape, batch_size=batch_size,
        check=check,
        save_root=save_root,
        label_transform=label_transform_3d,
        mask_channel=True,
        n_samples_train=100,
        n_samples_val=10,
        n_iterations=int(2e4),
        out_channels=1,
    )


def main():
    train_compartments_2d_v1()
    # train_compartments_3d_v1()


if __name__ == "__main__":
    main()
