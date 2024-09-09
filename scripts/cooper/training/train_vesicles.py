import os
from glob import glob

from synaptic_reconstruction.training import supervised_training


def train_check():
    """Training on a subset of the old version of training data to
    check that the training implementation works.

    NOTE: patch_shape is chosen arbitrarily, please check what we have used for the
    earlier training.
    """

    root = "/scratch-grete/projects/nim00007/data/synaptic_reconstruction/train_data_cooper"
    train_paths = glob(os.path.join(root, "01_hoi_maus_2020_incomplete", "*.h5"))
    val_paths = glob(os.path.join(root, "02_hcc_nanogold", "*.h5"))

    supervised_training(
        name="check-vesicle-training",
        train_paths=train_paths,
        val_paths=val_paths,
        label_key="/labels/vesicles_postprocessed",
        patch_shape=(32, 128, 128),
        save_root=".",
    )


def main():
    train_check()


if __name__ == "__main__":
    main()
