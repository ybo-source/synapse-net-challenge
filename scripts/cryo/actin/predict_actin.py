import h5py
import numpy as np
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


predict_actin_val()
