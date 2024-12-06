from glob import glob

from micro_sam.training import train_sam, default_sam_dataset
from torch_em.segmentation import get_data_loader

images = sorted(glob("initial_annotations/images/*.tif"))
labels = sorted(glob("initial_annotations/labels/*.tif"))

train_images, train_labels = images[:-2], labels[:-2]
val_images, val_labels = images[-2:], labels[-2:]

min_size = 100


train_ds = default_sam_dataset(
    raw_paths=train_images, raw_key=None,
    label_paths=train_labels, label_key=None,
    patch_shape=[512, 512], with_segmentation_decoder=False,
    n_samples=200, min_size=min_size,
    max_sampling_attempts=5000,
)
train_loader = get_data_loader(train_ds, shuffle=True, num_workers=8, batch_size=2)

# from torch_em.util.debug import check_loader
# check_loader(train_loader, 5, instance_labels=True)

val_ds = default_sam_dataset(
    raw_paths=val_images, raw_key=None,
    label_paths=val_labels, label_key=None,
    patch_shape=[512, 512], with_segmentation_decoder=False,
    is_train=False, n_samples=25, min_size=min_size,
)
val_loader = get_data_loader(val_ds, shuffle=True, num_workers=4, batch_size=1)

train_sam(
    name="vesicle_model", model_type="vit_b_em_organelles",
    train_loader=train_loader, val_loader=val_loader,
    n_epochs=100, n_objects_per_batch=10,
    with_segmentation_decoder=False,
)
