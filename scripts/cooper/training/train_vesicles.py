import os
from glob import glob

from synaptic_reconstruction.training import supervised_training
from synaptic_reconstruction.training import semisupervised_training

def train_check():
    """Training on a subset of the old version of training data to
    check that the training implementation works.

    NOTE: patch_shape is chosen arbitrarily, please check what we have used for the
    earlier training.
    """

    root = "/scratch-grete/projects/nim00007/data/synaptic_reconstruction/train_data_cooper"
    train_paths = glob(os.path.join(root, "01_hoi_maus_2020_incomplete", "*.h5"))
    val_paths = glob(os.path.join(root, "02_hcc_nanogold", "*.h5"))

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    patch_shape = [48, 256, 256]
    batch_size = 4
    check = False

    supervised_training(
        name="vesicles-model-new_postprocessing",
        train_paths=train_paths,
        val_paths=val_paths,
        label_key="/labels/vesicles_postprocessed",
        patch_shape=patch_shape, batch_size=batch_size,
        n_samples_train=None, n_samples_val=25,
        check=check,
        save_root=".",
    )


def main():
    train_check()


if __name__ == "__main__":
    main()
