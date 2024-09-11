from typing import Optional, Tuple

import torch
import torch_em
from torch_em.model import AnisotropicUNet, UNet2d


def get_3d_model(
    out_channels: int,
    scale_factors: Tuple[Tuple[int, int, int]] = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    initial_features: int = 32,
    final_activation: str = "Sigmoid",
) -> torch.nn.Module:
    """Get the U-Net model for 3D segmentation tasks.

    Args:
        out_channels: The number of output channels of the network.
        scale_factors: The downscaling factors for each level of the U-Net encoder.
        initial_features: The number of features in the first level of the U-Net.
            The number of features increases by a factor of two in each level.
        final_activation: The activation applied to the last output layer.
    Returns:
        The U-Net.
    """
    model = AnisotropicUNet(
        scale_factors=scale_factors,
        in_channels=1,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        final_activation=final_activation,
    )
    return model


def get_2d_model(
    out_channels: int,
    initial_features: int = 32,
    final_activation: str = "Sigmoid",
) -> torch.nn.Module:
    """Get the U-Net model for 2D segmentation tasks.

    Args:
        out_channels: The number of output channels of the network.
        initial_features: The number of features in the first level of the U-Net.
            The number of features increases by a factor of two in each level.
        final_activation: The activation applied to the last output layer.

    Returns:
        The U-Net.
    """
    model = UNet2d(
        in_channels=1,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        depth=4,
        final_activation=final_activation,
    )
    return model


def get_supervised_loader(
    data_paths: Tuple[str],
    raw_key: str,
    label_key: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples: Optional[int],
    add_boundary_transform: bool = True,
    label_dtype=torch.float32,
    rois: Optional[Tuple[Tuple[slice]]] = None,
    sampler: Optional[callable] = None,
) -> torch.utils.data.DataLoader:
    """Get a dataloader for supervised segmentation training.

    Args:
        data_paths: The filepaths to the hdf5 files containing the training data.
        raw_key: The key that holds the raw data inside of the hdf5.
        label_key: The key that holds the labels inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        batch_size: The batch size for training.
        n_samples: The number of samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        add_boundary_transform: Whether to add a boundary channel to the training data.
        label_dtype: The datatype of the labels returned by the dataloader.
        rois: Optional region of interests for training.
        sampler: Optional sampler for selecting blocks for training.
            By default a minimum instance sampler will be used.

    Returns:
        The PyTorch dataloader.
    """

    if add_boundary_transform:
        label_transform = torch_em.transform.BoundaryTransform(add_binary_target=True)
    else:
        label_transform = torch_em.transform.label.connected_components
    transform = torch_em.transform.Compose(
        torch_em.transform.PadIfNecessary(patch_shape), torch_em.transform.get_augmentations(3)
    )

    num_workers = 4 * batch_size

    if sampler is None:
        sampler = torch_em.data.sampler.MinInstanceSampler(min_num_instances=4)
    loader = torch_em.default_segmentation_loader(
        data_paths, raw_key,
        data_paths, label_key, sampler=sampler,
        batch_size=batch_size, patch_shape=patch_shape,
        is_seg_dataset=True, label_transform=label_transform, transform=transform,
        num_workers=num_workers, shuffle=True, n_samples=n_samples,
        label_dtype=label_dtype, rois=rois,
    )
    return loader


def supervised_training(
    name: str,
    train_paths: Tuple[str],
    val_paths: Tuple[str],
    label_key: str,
    patch_shape: Tuple[int, int, int],
    save_root: str,
    raw_key: str = "raw",
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e5),
    train_rois: Optional[Tuple[Tuple[slice]]] = None,
    val_rois: Optional[Tuple[Tuple[slice]]] = None,
    sampler: Optional[callable] = None,
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    check: bool = False,
):
    """Run supervised segmentation training.

    Args:
        name: The name for the checkpoint to be trained.
        train_paths: Filepaths to the hdf5 files for the training data.
        val_paths: Filepaths to the df5 files for the validation data.
        label_key: The key that holds the labels inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        save_root: Folder where the checkpoint will be saved.
        raw_key: The key that holds the raw data inside of the hdf5.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        train_rois: Optional region of interests for training.
        val_rois: Optional region of interests for validation.
        sampler: Optional sampler for selecting blocks for training.
            By default a minimum instance sampler will be used.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
        check: Whether to check the training and validation loaders instead of running training.
    """
    train_loader = get_supervised_loader(train_paths, raw_key, label_key, patch_shape, batch_size,
                                         n_samples=n_samples_train, rois=train_rois, sampler=sampler)
    val_loader = get_supervised_loader(val_paths, raw_key, label_key, patch_shape, batch_size,
                                       n_samples=n_samples_val, rois=val_rois, sampler=sampler)

    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=4)
        check_loader(val_loader, n_samples=4)
        return

    # TODO determine whether we train a 2D or 3D model.
    is_2d = False
    if is_2d:
        model = get_2d_model(out_channels=2)
    else:
        model = get_3d_model(out_channels=2)

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        save_root=save_root,
    )
    trainer.fit(n_iterations)
