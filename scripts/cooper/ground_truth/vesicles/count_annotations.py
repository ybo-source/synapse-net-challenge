import os
from glob import glob

import h5py
import numpy as np

from tqdm import tqdm


ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2"
SKIP_PREFIX = ("06", "08", "09")


def main():
    n_tomograms = {}
    n_vesicles_imod = {}
    n_vesicles_auto = {}
    n_vesicles_total = {}

    datasets = sorted(glob(os.path.join(ROOT, "*")))

    for ds in tqdm(datasets):
        ds_name = os.path.basename(ds)
        if ds_name.startswith(SKIP_PREFIX):
            continue
        tomograms = glob(os.path.join(ds, "*.h5"))

        n_ves_imod, n_ves_auto = 0, 0
        for tomo in tomograms:
            with h5py.File(tomo, "r") as f:
                ves_imod = f["/labels/vesicles/imod"][:]
                ves_auto = f["/labels/vesicles/additional_vesicles"][:]
            n_ves_imod += (len(np.unique(ves_imod)) - 1)
            n_ves_auto += (len(np.unique(ves_auto)) - 1)

        n_tomograms[ds_name] = len(tomograms)
        n_vesicles_imod[ds_name] = n_ves_imod
        n_vesicles_auto[ds_name] = n_ves_auto
        n_vesicles_total[ds_name] = n_ves_imod + n_ves_auto

    print("Total number of tomograms:")
    print(sum(n_tomograms.values()))

    print("Total number of vesicles:")
    print(sum(n_vesicles_total.values()))

    # TODO analyze the number of vesicles from IMOD and auto annotation further for the methods tile


if __name__ == "__main__":
    main()
