import torch
import torch_em
from torch_em.model import AnisotropicUNet


def get_model(
    out_channels,
    scale_factors=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    initial_features=32,
    final_activation="Sigmoid"
):
    model = AnisotropicUNet(
        scale_factors=scale_factors,
        in_channels=1,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        final_activation=final_activation,
    )
    return model


def get_supervised_loader(
    data_paths,
    raw_key,
    label_key,
    patch_shape,
    batch_size, root,
    n_samples,
    add_boundary_transform=True,
    label_dtype=torch.float32,
):

    if add_boundary_transform:
        label_transform = torch_em.transform.BoundaryTransform(add_binary_target=True)
    else:
        label_transform = torch_em.transform.label.connected_components
    transform = torch_em.transform.Compose(
        torch_em.transform.PadIfNecessary(patch_shape), torch_em.transform.get_augmentations(3)
    )

    num_workers = 4 * batch_size

    sampler = torch_em.data.sampler.MinInstanceSampler(min_num_instances=4)
    loader = torch_em.default_segmentation_loader(
        data_paths, raw_key,
        data_paths, label_key, sampler=sampler,
        batch_size=batch_size, patch_shape=patch_shape,
        is_seg_dataset=True, label_transform=label_transform, transform=transform,
        num_workers=num_workers, shuffle=True, n_samples=n_samples,
        label_dtype=label_dtype,
    )
    return loader


# TODO enable 2d training
def supervised_training(
    name,
    root,
    train_paths,
    val_paths,
    label_key,
    patch_shape,
    raw_key="raw",
    batch_size=1,
    lr=1e-4,
    n_iterations=int(1e5),
    n_samples_train=None,
    n_samples_val=None,
    check=False,
):
    train_loader = get_supervised_loader(train_paths, raw_key, label_key, patch_shape, batch_size, root,
                                         n_samples=n_samples_train)
    val_loader = get_supervised_loader(val_paths, raw_key, label_key, patch_shape, batch_size, root,
                                       n_samples=n_samples_val)

    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=4)
        check_loader(val_loader, n_samples=4)
        return

    model = get_model(out_channels=2)

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        save_root=root,
    )
    trainer.fit(n_iterations)
