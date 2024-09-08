import numpy as np
import torch
import torch_em
import torch_em.self_training as self_training
from torchvision import transforms

from .supervised_training import get_model, get_supervised_loader


def weak_augmentations(p=0.75):
    norm = torch_em.transform.raw.standardize
    aug = transforms.Compose([
        norm,
        transforms.RandomApply([torch_em.transform.raw.GaussianBlur()], p=p),
        transforms.RandomApply([torch_em.transform.raw.AdditiveGaussianNoise(
            scale=(0, 0.15), clip_kwargs=False)], p=p
        ),
    ])
    return torch_em.transform.raw.get_raw_transform(normalizer=norm, augmentation1=aug)


def get_unsupervised_loader(
    paths, raw_key, patch_shape, batch_size, n_samples, exclude_top_and_bottom=False
):

    # We exclude the top and bottom slices where the tomogram reconstruction is bad.
    if exclude_top_and_bottom:
        roi = np.s_[5:-5, :, :]
    else:
        roi = None

    raw_transform = torch_em.transform.get_raw_transform()
    transform = torch_em.transform.get_augmentations(ndim=3)

    augmentations = (weak_augmentations(), weak_augmentations())
    datasets = [
        torch_em.data.RawDataset(path, raw_key, patch_shape, raw_transform, transform,
                                 augmentations=augmentations, roi=roi)
        for path in paths
    ]
    ds = torch.utils.data.ConcatDataset(datasets)

    num_workers = 4 * batch_size
    loader = torch_em.segmentation.get_data_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return loader


def semisupervised_training(
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

    unsupervised_train_loader = get_unsupervised_loader(train_paths, raw_key, patch_shape, batch_size,
                                                        n_samples=n_samples_train)
    unsupervised_val_loader = get_unsupervised_loader(val_paths, raw_key, patch_shape, batch_size,
                                                      n_samples=n_samples_val)

    # TODO check the semisup loader
    if check:
        # from torch_em.util.debug import check_loader
        # check_loader(train_loader, n_samples=4)
        # check_loader(val_loader, n_samples=4)
        return

    model = get_model(out_channels=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Self training functionality.
    pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=0.9)
    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        device=device,
        log_image_interval=100,
        compile_model=False,
        save_root=root,
    )
    trainer.fit(n_iterations)
