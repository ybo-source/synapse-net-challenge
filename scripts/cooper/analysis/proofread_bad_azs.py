import os
from glob import glob

import napari
import numpy as np
import pandas as pd
import h5py

from magicgui import magicgui
from tqdm import tqdm
from synapse_net.morphology import skeletonize_object
from synapse_net.ground_truth.shape_refinement import edge_filter


def proofread_az(raw_path, seg_path):
    assert os.path.exists(raw_path), raw_path
    assert os.path.exists(seg_path), seg_path

    with h5py.File(seg_path, "r") as f:
        if "thin_az_corrected" in f:
            return
        seg = f["/az_thin_proofread"][:]
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:]

    hmap = edge_filter(raw, sigma=1.0, method="sato", per_slice=True, n_threads=8)
    membrane_mask = hmap > 0.5

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg, colormap={1: "orange"}, opacity=1)
    v.add_labels(np.zeros_like(seg), name="canvas")
    v.add_labels(membrane_mask, visible=False)

    @magicgui(call_button="skeletonize")
    def skeletonize():
        data = v.layers["canvas"].data
        data = np.logical_and(data, membrane_mask)
        data = skeletonize_object(data)
        new_mask = data != 0
        v.layers["seg"].data[new_mask] = data[new_mask]

    @magicgui(call_button="save")
    def save():
        seg = v.layers["seg"].data

        with h5py.File(seg_path, "a") as f:
            f.create_dataset("thin_az_corrected", data=seg, compression="gzip")

    v.window.add_dock_widget(skeletonize)
    v.window.add_dock_widget(save)

    napari.run()


def main():
    ratings = pd.read_excel("quality_ratings/az_quality_clean_FM.xlsx")

    paths = sorted(glob("proofread_az/**/*.h5", recursive=True))
    for path in tqdm(paths):

        ds, fname = os.path.split(path)
        ds = os.path.split(ds)[1]
        fname = os.path.splitext(fname)[0]

        try:
            rating = ratings[
                (ratings["Dataset"] == ds) & (ratings["Tomogram"] == fname)
            ]["Rating"].values[0]
        except IndexError:
            breakpoint()
        if rating == "Good":
            continue

        print(rating)
        print(ds, fname)

        raw_path = os.path.join("imig_data", ds, f"{fname}.h5")
        proofread_az(raw_path, path)


if __name__ == "__main__":
    main()
