import numpy as np
from micro_sam.training import train_sam, default_sam_dataset
from torch_em.data.sampler import MinInstanceSampler
from torch_em.segmentation import get_data_loader

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
