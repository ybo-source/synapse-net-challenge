import os

import h5py
import numpy as np
import pandas as pd
import napari

from skimage.measure import label

from tqdm import tqdm

val_table = "/home/pape/Desktop/sfb1286/mboc_synapse/qualitative-stem-eval.xlsx"
val_table = pd.read_excel(val_table)


def _get_n_azs(path):
    access = np.s_[::2, ::2, ::2]
    with h5py.File(path, "r") as f:
        az = f["labels/active_zone"][access]
    az = label(az)
    ids, sizes = np.unique(az, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]
    n_azs = np.sum(sizes > 10000)
    return n_azs, n_azs


def eval_az():
    azs_found = []
    azs_total = []

    # for the "all" tomograms load the prediction, measure number components,
    # size filter and count these as found and as total
    for i, row in tqdm(val_table.iterrows(), total=len(val_table)):
        az_found = row["AZ Found"]
        if az_found == "all":
            path = os.path.join("04_full_reconstruction", row.dataset, row.tomogram)
            assert os.path.exists(path)
            az_found, az_total = _get_n_azs(path)
        else:
            az_total = row["AZ Total"]

        azs_found.append(az_found)
        azs_total.append(az_total)

    n_found = np.sum(azs_found)
    n_azs = np.sum(azs_total)

    print("AZ Evaluation:")
    print("Number of correctly identified AZs:", n_found, "/", n_azs, f"({float(n_found)/n_azs}%)")


# measure in how many pieces each compartment was split
def eval_compartments():
    pieces_per_compartment = []
    for i, row in val_table.iterrows():
        for comp in [
            "Compartment 1",
            "Compartment 2",
            "Compartment 3",
            "Compartment 4",
        ]:
            n_pieces = row[comp]
            if isinstance(n_pieces, str):
                n_pieces = len(n_pieces.split(","))
            elif np.isnan(n_pieces):
                continue
            else:
                assert isinstance(n_pieces, (float, int))
                n_pieces = 1
            pieces_per_compartment.append(n_pieces)

    avg = np.mean(pieces_per_compartment)
    std = np.std(pieces_per_compartment)
    max_ = np.max(pieces_per_compartment)
    print("Compartment Evaluation:")
    print("Avergage pieces per compartment:", avg, "+-", std)
    print("Max pieces per compartment:", max_)
    print("Number of compartments:", len(pieces_per_compartment))


def eval_mitos():
    mito_correct = []
    mito_split = []
    mito_merged = []
    mito_total = []
    wrong_object = []

    mito_table = val_table.fillna(0)
    # measure % of mito correct, mito split and mito merged
    for i, row in mito_table.iterrows():
        mito_correct.append(row["Mito Correct"])
        mito_split.append(row["Mito Split"])
        mito_merged.append(row["Mito Merged"])
        mito_total.append(row["Mito Total"])
        wrong_object.append(row["Wrong Object"])

    n_mitos = np.sum(mito_total)
    n_correct = np.sum(mito_correct)
    print("Mito Evaluation:")
    print("Number of correctly identified mitos:", n_correct, "/", n_mitos, f"({float(n_correct)/n_mitos}%)")
    print("Number of merged mitos:", np.sum(mito_merged))
    print("Number of split mitos:", np.sum(mito_split))
    print("Number of wrongly identified objects:", np.sum(wrong_object))


def check_mitos():
    scale = 3
    access = np.s_[::scale, ::scale, ::scale]

    root = "./04_full_reconstruction"
    for i, row in tqdm(val_table.iterrows(), total=len(val_table)):
        ds, fname = row.dataset, row.tomogram
        path = os.path.join(root, ds, fname)
        with h5py.File(path, "r") as f:
            raw = f["raw"][access]
            mitos = f["labels/mitochondria"][access]

        # ids, sizes = np.unique(mitos, return_counts=True)
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(mitos)
        napari.run()


def main():
    # check_mitos()

    # eval_mitos()
    # print()
    eval_compartments()
    # print()
    # eval_az()


main()
