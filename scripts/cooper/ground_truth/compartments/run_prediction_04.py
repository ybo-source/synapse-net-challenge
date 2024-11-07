import os
from glob import glob

import h5py
from tqdm import tqdm

from synaptic_reconstruction.inference.utils import _Scaler
from synaptic_reconstruction.inference.compartments import segment_compartments

INPUT_ROOT = ""
MODEL_PATH = ""
OUTPUT = "./predictions"


def segment_volume(input_path, model_path):
    with h5py.File(input_path, "r") as f:
        raw = f["raw"][:]

    scale = (0.25, 0.25, 0.25)
    scaler = _Scaler(scale)
    raw = scaler.scale_input(raw)

    seg = segment_compartments(raw, model_path, verbose=False)

    return raw, seg


def main():
    inputs = sorted(glob(os.path.join(INPUT_ROOT, "**/*.h5"), recursive=True))
    for input_path in tqdm(inputs):
        ds_name, fname = os.path.split(input_path)
        ds_name = os.path.split(ds_name)[1]
        output_folder = os.path.join(OUTPUT, ds_name)
        output_path = os.path.join(output_folder, fname)

        if os.path.exists(output_path):
            continue

        raw, seg = segment_volume(input_path, MODEL_PATH)
        os.makedirs(output_folder, exist_ok=True)
        with h5py.File(output_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/compartments", data=seg, compression="gzip")


if __name__ == "__main__":
    main()
