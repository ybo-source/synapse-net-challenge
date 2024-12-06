import os
from glob import glob

import h5py
from synapse_net.inference.mitochondria import segment_mitochondria
from tqdm import tqdm

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/04_full_reconstruction"  # noqa
MODEL_PATH = "/scratch-grete/projects/nim00007/models/exports_for_cooper/mito_model_s2.pt"  # noqa

# MODEL_PATH = "/scratch-grete/projects/nim00007/models/luca/mito/source_domain"


def run_seg(path):

    out_folder = "./mito_seg"
    ds, fname = os.path.split(path)
    ds = os.path.basename(ds)

    os.makedirs(os.path.join(out_folder, ds), exist_ok=True)
    out_path = os.path.join(out_folder, ds, fname)
    if os.path.exists(out_path):
        return

    with h5py.File(path, "r") as f:
        raw = f["raw"][:]

    scale = (0.5, 0.5, 0.5)
    seg = segment_mitochondria(raw, model_path=MODEL_PATH, scale=scale, verbose=False)
    with h5py.File(out_path, "a") as f:
        f.create_dataset("labels/mitochondria", data=seg, compression="gzip")


def run_seg_and_pred(path):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]

    scale = (0.5, 0.5, 0.5)
    seg, pred = segment_mitochondria(
        raw, model_path=MODEL_PATH, scale=scale, verbose=False, return_predictions=True
    )

    out_folder = "./mito_pred"
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, os.path.basename(path))

    with h5py.File(out_path, "a") as f:
        f.create_dataset("raw", data=raw[::2, ::2, ::2])
        f.create_dataset("labels/mitochondria", data=seg, compression="gzip")
        f.create_dataset("pred", data=pred, compression="gzip")


def main():
    paths = sorted(glob(os.path.join(ROOT, "**/*.h5"), recursive=True))
    for path in tqdm(paths):
        run_seg(path)
        # run_seg_and_pred(path)


main()
