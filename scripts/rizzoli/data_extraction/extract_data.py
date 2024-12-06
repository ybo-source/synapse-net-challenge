import os
from glob import glob

import h5py
import napari

from common import read_volume, read_labels


def extract_folder(input_folder, name, output_path, view):

    print("Extracing:", name)
    stack = read_volume(input_folder)
    vesicles = read_labels(input_folder, stack.shape, ["vesicles", "labeled_vesicles"])
    membrane = read_labels(input_folder, stack.shape, "membrane", fill_mask=True)

    if view:
        v = napari.Viewer()
        v.add_image(stack)
        v.add_labels(vesicles)
        v.add_labels(membrane)
        v.title = name
        napari.run()

    if output_path is not None:
        with h5py.File(output_path, "a") as f:
            f.create_dataset("raw", data=stack, compression="gzip")
            f.create_dataset("labels/membrane", data=membrane, compression="gzip")
            f.create_dataset("labels/vesicles", data=vesicles, compression="gzip")


def extract_all_data(root, output_root=None, view=False):

    skip_names = ["block10U3A_one_copy"]

    folders = sorted(glob(os.path.join(root, "block*")))
    for folder in folders:
        root_name = os.path.basename(folder)
        sub_folders = sorted(glob(os.path.join(folder, "*")))
        for sub_folder in sub_folders:
            sub_name = os.path.basename(sub_folder)
            name = f"{root_name}_{sub_name}"

            if name in skip_names:
                continue

            if output_root is None:
                output_path = None
            else:
                os.makedirs(output_root, exist_ok=True)
                output_path = os.path.join(output_root, f"{name}.h5")
            extract_folder(sub_folder, name, output_path, view)


def main():
    # folder = "/home/pape/Work/data/silvio/frog-em/block10U3A/one"
    root = "/home/pape/Work/data/silvio/frog-em"
    output_root = "/home/pape/Work/data/silvio/frog-em/extracted"
    extract_all_data(root, output_root, view=False)


if __name__ == "__main__":
    main()
