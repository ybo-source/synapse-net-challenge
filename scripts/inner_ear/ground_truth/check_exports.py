import argparse
import os
from glob import glob

import h5py
import napari


def check_file(path):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        segmentations = {}
        for name, ds in f["labels"].items():
            segmentations[name] = ds[:]

    v = napari.Viewer()
    v.add_image(raw)
    for name, seg in segmentations.items():
        v.add_labels(seg, name=name)
    v.title = path
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    folders = sorted(glob(os.path.join(args.root, "*")))

    for folder in folders:
        files = sorted(glob(os.path.join(folder, "*.h5")))
        for ff in files:
            check_file(ff)


if __name__ == "__main__":
    main()
