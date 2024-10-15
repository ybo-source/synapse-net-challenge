import os
from glob import glob

import numpy as np

from micro_sam.training import train_sam, default_sam_dataset
from sklearn.model_selection import train_test_split
from torch_em.data.sampler import MinInstanceSampler
from torch_em.segmentation import get_data_loader


def train_v1():
    data_path = "./segmentation.h5"

    with_segmentation_decoder = False
    patch_shape = [1, 462, 462]
    z_split = 400

    train_ds = default_sam_dataset(
        raw_paths=data_path, raw_key="raw_downscaled",
        label_paths=data_path, label_key="segmentation/compartments",
        patch_shape=patch_shape, with_segmentation_decoder=with_segmentation_decoder,
        sampler=MinInstanceSampler(2), rois=np.s_[z_split:, :, :],
        n_samples=200,
    )
    train_loader = get_data_loader(train_ds, shuffle=True, batch_size=2)

    val_ds = default_sam_dataset(
        raw_paths=data_path, raw_key="raw_downscaled",
        label_paths=data_path, label_key="segmentation/compartments",
        patch_shape=patch_shape, with_segmentation_decoder=with_segmentation_decoder,
        sampler=MinInstanceSampler(2), rois=np.s_[:z_split, :, :],
        is_train=False, n_samples=25,
    )
    val_loader = get_data_loader(val_ds, shuffle=True, batch_size=1)

    train_sam(
        name="compartment_model", model_type="vit_b",
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=100, n_objects_per_batch=10,
        with_segmentation_decoder=with_segmentation_decoder,
    )


def normalize_trafo(raw):
    raw = raw.astype("float32")
    raw -= raw.min()
    raw /= raw.max()
    raw *= 255
    return raw


def train_v2():
    data_root = "./output/postprocessed_annotations"
    paths = glob(os.path.join(data_root, "*.h5"))
    train_paths, val_paths = train_test_split(paths, test_size=0.1, random_state=42)

    with_segmentation_decoder = True
    patch_shape = (462, 462)

    train_ds = default_sam_dataset(
        raw_paths=train_paths, raw_key="data",
        label_paths=train_paths, label_key="labels/compartments",
        patch_shape=patch_shape, with_segmentation_decoder=with_segmentation_decoder,
        sampler=MinInstanceSampler(2), n_samples=250,
        raw_transform=normalize_trafo,
    )
    train_loader = get_data_loader(train_ds, shuffle=True, batch_size=2)

    val_ds = default_sam_dataset(
        raw_paths=val_paths, raw_key="data",
        label_paths=val_paths, label_key="labels/compartments",
        patch_shape=patch_shape, with_segmentation_decoder=with_segmentation_decoder,
        sampler=MinInstanceSampler(2),  is_train=False, n_samples=25,
        raw_transform=normalize_trafo,
    )
    val_loader = get_data_loader(val_ds, shuffle=True, batch_size=1)

    train_sam(
        name="compartment_model_v2", model_type="vit_b",
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=100, n_objects_per_batch=10,
        with_segmentation_decoder=with_segmentation_decoder,
    )


def main():
    train_v2()


if __name__ == "__main__":
    main()
