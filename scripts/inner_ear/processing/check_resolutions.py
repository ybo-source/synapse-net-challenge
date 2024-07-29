import os

import mrcfile
import numpy as np

from tqdm import tqdm
from synaptic_reconstruction.file_utils import get_data_path
from parse_table import parse_table, get_data_root


def read_resolution(folder):
    path = get_data_path(folder)

    with mrcfile.open(path, "r") as f:
        resolution = f.voxel_size.tolist()
    resolution = tuple(np.round(res / 10, 3) for res in resolution)
    return resolution


def check_resolutions(table):
    resolutions_new = set()
    resolutions_old = set()

    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        micro = row["EM alt vs. Neu"]
        if micro == "beides":
            resolution = read_resolution(folder)
            if len(resolutions_old) > 0 and resolution not in resolutions_old:
                print("Old:", folder, resolution)
            resolutions_old.add(resolution)

            folder_new = os.path.join(folder, "Tomo neues EM")
            if not os.path.exists(folder_new):
                folder_new = os.path.join(folder, "neues EM")
            resolution = read_resolution(folder_new)
            if len(resolutions_new) > 0 and resolution not in resolutions_new:
                print("New:", folder, resolution)
            resolutions_new.add(resolution)

        elif micro == "alt":
            resolution = read_resolution(folder)
            if len(resolutions_old) > 0 and resolution not in resolutions_old:
                print("Old:", folder, resolution)
            resolutions_old.add(resolution)

        elif micro == "neu":
            resolution = read_resolution(folder)
            if len(resolutions_new) > 0 and resolution not in resolutions_new:
                print("New:", folder, resolution)
            resolutions_new.add(resolution)

    print("Resolutions old:")
    print(resolutions_old)
    print("Resolutions New:")
    print(resolutions_new)


def main():
    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)
    check_resolutions(table)


if __name__ == "__main__":
    main()
