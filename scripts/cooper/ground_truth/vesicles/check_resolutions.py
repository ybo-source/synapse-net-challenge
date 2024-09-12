import os
from glob import glob

import mrcfile
import numpy as np


ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer"  # noqa


def read_resolution(data_path):
    with mrcfile.open(data_path, "r") as f:
        resolution = f.voxel_size.tolist()
    resolution = tuple(np.round(res / 10, 3) for res in resolution)
    return resolution


def check_resolution(folder):
    files = glob(os.path.join(folder, "**/*.mrc"), recursive=True)
    files += glob(os.path.join(folder, "**/*.rec"), recursive=True)
    resolution_list = []
    for path in files:
        res = read_resolution(path)
        resolution_list.append(res)

    resolutions = list(set(resolution_list))
    if len(resolutions) == 1:
        return resolutions[0], None

    counts = []
    for res in resolutions:
        count = len([resl == res for resl in resolution_list])
        counts.append(count)
    return resolutions, counts


def main():
    folders = sorted(glob(os.path.join(ROOT, "*")))
    for folder in folders:
        name = os.path.basename(folder)
        print(name)
        resolutions, counts = check_resolution(folder)
        if counts is None:
            print(resolutions)
        else:
            print(resolutions)


if __name__ == "__main__":
    main()
