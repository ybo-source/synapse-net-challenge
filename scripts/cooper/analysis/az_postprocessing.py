import os
from glob import glob

import h5py
import napari
import numpy as np

from magicgui import magicgui
from scipy.ndimage import binary_dilation, binary_opening
from skimage.measure import label


def postprocess_az(thin_az_seg):
    # seg = binary_dilation(thin_az_seg, iterations=1)
    # seg = binary_opening(seg)
    seg = label(thin_az_seg)

    ids, sizes = np.unique(seg, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]
    seg = seg == ids[np.argmax(sizes)].astype("uint8")
    return seg


def process_az(raw_path, az_path):
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:]

    with h5py.File(az_path, "r") as f:
        seg = f["thin_az"][:]

    seg_pp = postprocess_az(seg)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg, opacity=1, visible=True)
    segl = v.add_labels(seg_pp, opacity=1)
    segl.new_colormap()
    v.title = raw_path
    napari.run()


def check_all_postprocessed():
    raw_paths = sorted(glob(os.path.join("imig_data/**/*.h5"), recursive=True))
    seg_paths = sorted(glob(os.path.join("az_segmentation/**/*.h5"), recursive=True))
    assert len(raw_paths) == len(seg_paths)
    for raw_path, seg_path in zip(raw_paths, seg_paths):
        process_az(raw_path, seg_path)


def proofread_file(raw_path, az_path, out_root):
    ds, fname = os.path.split(raw_path)
    ds = os.path.basename(ds)

    out_folder = os.path.join(out_root, ds)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, fname)

    if os.path.exists(out_path):
        return

    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:]

    with h5py.File(az_path, "r") as f:
        seg = f["thin_az"][:]

    seg_pp = postprocess_az(seg)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg, opacity=1, visible=True, name="original")
    segl = v.add_labels(seg_pp, opacity=1, name="postprocessed")
    segl.new_colormap()

    v.title = raw_path

    @magicgui(call_button="Postprocess")
    def postprocess():
        seg = v.layers["postprocessed"].data
        seg = postprocess_az(seg)
        v.layers["postprocessed"].data = seg

    @magicgui(call_button="Save")
    def save():
        seg = v.layers["postprocessed"].data
        with h5py.File(out_path, "a") as f:
            f.create_dataset("az_thin_proofread", data=seg, compression="gzip")
        print("Save done!")

    v.window.add_dock_widget(postprocess)
    v.window.add_dock_widget(save)

    napari.run()


def proofread_az(out_folder):
    raw_paths = sorted(glob(os.path.join("imig_data/**/*.h5"), recursive=True))
    seg_paths = sorted(glob(os.path.join("az_segmentation/**/*.h5"), recursive=True))
    assert len(raw_paths) == len(seg_paths)
    os.makedirs(out_folder, exist_ok=True)
    for i, (raw_path, seg_path) in enumerate(zip(raw_paths, seg_paths)):
        print(i, "/", len(seg_paths))
        proofread_file(raw_path, seg_path, out_folder)


def main():
    # check_all_postprocessed()
    # process_az(
    #     "./imig_data/Munc13DKO/A_M13DKO_060212_DKO1.1_crop.h5",
    #     "./az_segmentation/Munc13DKO/A_M13DKO_060212_DKO1.1_crop.h5"
    # )
    proofread_az("./proofread_az")


if __name__ == "__main__":
    main()
