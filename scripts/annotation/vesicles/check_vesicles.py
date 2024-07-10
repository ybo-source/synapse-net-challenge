import argparse
import os

from glob import glob
from natsort import natsorted

import h5py
import napari

from common import get_root


def check_annotations(version):
    data_root = get_root(version)
    correction_folder = f"./corrections/v{version}"

    files = natsorted(glob(os.path.join(data_root, "**/*.h5")))

    for path in files:
        with h5py.File(path, "r") as f:
            raw = f["raw"][:]
            seg = f["vesicles/segment_from_boundaries_indv"][:]

        correction_path = os.path.join(correction_folder, os.path.relpath(path, data_root))
        corrected_seg = None
        if os.path.exists(correction_path):
            with h5py.File(correction_path, "r") as f:
                corrected_seg = f["vesicles/corrected"][:]

        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(seg)
        if corrected_seg is not None:
            v.add_labels(corrected_seg)
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", default=1, type=int)
    args = parser.parse_args()
    check_annotations(args.version)


if __name__ == "__main__":
    main()
