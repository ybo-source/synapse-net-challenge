import argparse
import os
from glob import glob

import h5py
import napari
import numpy as np
from tqdm import tqdm


def check_folder(folder, restrict_to_bb):
    files = glob(os.path.join(folder, "*.h5"))

    for ff in tqdm(files):
        with h5py.File(ff, "r") as f:
            vesicles = f["labels/vesicles"][:]
            if restrict_to_bb:
                bb = np.where(vesicles != 0)
                bb = tuple(slice(
                    int(b.min()), int(b.max())
                ) for b in bb)
                vesicles = vesicles[bb]
            else:
                bb = np.s_[:]
            data = f["raw"][bb]
        assert data.shape == vesicles.shape

        v = napari.Viewer()
        v.add_image(data)
        v.add_labels(vesicles)
        v.title = os.path.basename(ff)
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--restrict_to_bb", "-r", action="store_true")
    args = parser.parse_args()

    check_folder(args.folder, args.restrict_to_bb)


if __name__ == "__main__":
    main()
