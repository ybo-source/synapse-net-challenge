import numpy as np

from synaptic_reconstruction.training import supervised_training
from torch_em.data.sampler import MinForegroundSampler


def train_actin_deepict():
    """Train a network for actin segmentation on the deepict dataset.
    """

    train_paths = [
        "/mnt/lustre-grete/usr/u12086/data/deepict/deepict_actin/00004.h5",
        "/mnt/lustre-grete/usr/u12086/data/deepict/deepict_actin/00012.h5",
    ]
    val_paths = [
        "/mnt/lustre-grete/usr/u12086/data/deepict/deepict_actin/00012.h5",
    ]

    train_rois = [np.s_[:, :, :], np.s_[:250, :, :]]
    val_rois = [np.s_[250:, :, :]]

    patch_shape = (64, 384, 384)
    sampler = MinForegroundSampler(min_fraction=0.025, p_reject=0.95)

    supervised_training(
        name="actin-deepict",
        label_key="/labels/actin",
        patch_shape=patch_shape,
        train_paths=train_paths,
        val_paths=val_paths,
        train_rois=train_rois,
        val_rois=val_rois,
        sampler=sampler,
        save_root=".",
    )


def main():
    train_actin_deepict()


if __name__ == "__main__":
    main()
