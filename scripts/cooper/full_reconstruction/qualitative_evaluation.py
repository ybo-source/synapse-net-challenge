import os

import h5py
import numpy as np
import pandas as pd
import napari

from tqdm import tqdm

val_table = "/home/pape/Desktop/sfb1286/mboc_synapse/qualitative-stem-eval.xlsx"
val_table = pd.read_excel(val_table)


def eval_az():
    az_found = []
    az_total = []

    # TODO for the "all" tomograms load the prediction, measure number components,
    # size filter and count these as found and as total
    for i, row in val_table.iterrows():
        pass


# TODO measure in how many pieces each compartment was split
def eval_compartments():
    pass


def eval_mitos():
    mito_correct = []
    mito_split = []
    mito_merged = []
    mito_total = []

    # TODO measure % of mito correct, mito split and mito merged
    for i, row in val_table.iterrows():
        pass


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
    check_mitos()


main()
