import os
import sys
from glob import glob

import h5py

sys.path.append("../processing")
from parse_table import get_data_root  # noqa


def main():
    input_folder = os.path.join(get_data_root(), "Electron-Microscopy-Susi", "Analyse", "for_annotation")
    output_folder = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/vesicle_gt"
    os.makedirs(output_folder, exist_ok=True)

    files = glob(os.path.join(input_folder, "*.h5"))
    for ff in files:
        fname = os.path.basename(ff)
        output_file = os.path.join(output_folder, fname)

        with h5py.File(ff, "r") as f:
            raw = f["raw"][:]
            vesicles = f["corrected/vesicles"][:]

        with h5py.File(output_file, "a") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/vesicles", data=vesicles, compression="gzip")


if __name__ == "__main__":
    main()
