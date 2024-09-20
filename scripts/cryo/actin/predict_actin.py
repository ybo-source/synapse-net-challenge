import os
from glob import glob
from pathlib import Path

import h5py
import numpy as np
from elf.io import open_file
from synaptic_reconstruction.inference.actin import segment_actin


# Run prediction on the actin val volume.
def predict_actin_val():
    path = "/mnt/lustre-grete/usr/u12086/data/deepict/deepict_actin/00012.h5"

    # This is the validation ROI.
    roi = np.s_[250:, :, :]
    with h5py.File(path, "r") as f:
        raw = f["raw"][roi]

    model_path = "./checkpoints/actin-deepict"
    seg, pred = segment_actin(raw, model_path, verbose=True, return_predictions=True)

    with h5py.File("actin_pred.h5", "a") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("actin_seg", data=seg, compression="gzip")
        f.create_dataset("actin_pred", data=pred, compression="gzip")


def predict_actin_fb():
    root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/fernandez-busnadiego/from_arsen/tomos_actin_18924"  # noqa
    files = glob(os.path.join(root, "*.mrc"))

    model_path = "./checkpoints/actin-adapted-v1"

    for ff in files:
        print("Predict", ff)
        with open_file(ff, "r") as f:
            raw = f["data"][:]
        seg, pred = segment_actin(raw, model_path, verbose=True, return_predictions=True)

        out_path = f"{Path(ff).stem}.h5"
        with h5py.File(out_path, "a") as f:
            # f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("actin_seg", data=seg, compression="gzip")
            f.create_dataset("actin_pred", data=pred, compression="gzip")


def main():
    # predict_actin_val()
    predict_actin_fb()


if __name__ == "__main__":
    main()
