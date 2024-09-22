import os
from synaptic_reconstruction.training.domain_adaptation import mean_teacher_adaptation

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/fernandez-busnadiego/from_arsen/tomos_actin_18924"  # noqa


def actin_adaptation_v1():
    train_paths = [
        os.path.join(ROOT, "Lam12_ts_006_newstack_rec_deconv_bin4_250823.mrc"),
        os.path.join(ROOT, "Lam13_ts_003_dimi_resize.mrc")
    ]
    val_paths = [
        os.path.join(ROOT, "2023_08_10_lam9_ts_002_resize.mrc")
    ]
    patch_shape = (64, 384, 384)
    mean_teacher_adaptation(
        name="actin-adapted-v1",
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        raw_key="data",
        patch_shape=patch_shape,
        save_root=".",
        source_checkpoint="./checkpoints/actin-deepict",
        confidence_threshold=0.75,
    )


def main():
    actin_adaptation_v1()


if __name__ == "__main__":
    main()
