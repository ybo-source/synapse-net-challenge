import torch
import torch_em
import torch_em.self_training as self_training

from .semisupervised_training import get_unsupervised_loader
from .supervised_training import get_model, get_supervised_loader


def mean_teacher_adaptation(
    name,
    root,
    unsupervised_train_paths,
    unsupervised_val_paths,
    patch_shape,
    source_checkpoint=None,
    supervised_train_paths=None,
    supervised_val_paths=None,
    confidence_threshold=0.9,
    raw_key="raw",
    raw_key_supervised="raw",
    label_key=None,
    batch_size=1,
    lr=1e-4,
    n_iterations=int(1e4),
    n_samples_train=None,
    n_samples_val=None,
):
    assert (supervised_train_paths is None) == (supervised_val_paths is None)

    if source_checkpoint is None:
        # training from scratch only makes sense if we have supervised training data
        # that's why we have the assertion here.
        assert supervised_train_paths is not None
        print("Mean teacher training from scratch (AdaMT)")
        model = get_model(2)
        reinit_teacher = True
    else:
        print("Mean teacehr training initialized from source model:", source_checkpoint)
        model = torch_em.util.load_model(source_checkpoint)
        reinit_teacher = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # self training functionality
    pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=confidence_threshold)
    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()

    unsupervised_train_loader = get_unsupervised_loader(
        unsupervised_train_paths, raw_key, patch_shape, batch_size, n_samples=n_samples_train
    )
    unsupervised_val_loader = get_unsupervised_loader(
        unsupervised_val_paths, raw_key, patch_shape, batch_size, n_samples=n_samples_val
    )

    if supervised_train_paths is not None:
        assert label_key is not None
        supervised_train_loader = get_supervised_loader(
            supervised_train_paths, raw_key_supervised, label_key,
            patch_shape, batch_size, root, n_samples=n_samples_train,
        )
        supervised_val_loader = get_supervised_loader(
            supervised_val_paths, raw_key_supervised, label_key,
            patch_shape, batch_size, root, n_samples=n_samples_val,
        )
    else:
        supervised_train_loader = None
        supervised_val_loader = None

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=supervised_train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=supervised_val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        device=device,
        reinit_teacher=reinit_teacher,
        save_root=root,
    )
    trainer.fit(n_iterations)
