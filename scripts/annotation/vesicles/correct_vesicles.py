import argparse
import os

from copy import deepcopy
from glob import glob
from natsort import natsorted

import h5py
import napari
import numpy as np

from magicgui import magicgui
from napari.utils.notifications import show_info
from skimage.measure import label

from common import get_root


def run_correction(input_path, output_path, fname):
    continue_correction = True
    next_id = 1

    with h5py.File(input_path, "r") as f:
        raw = f["raw"][:]
        segmentation = f["vesicles/segment_from_boundaries_indv"][:]

    segmentation = label(segmentation)
    n_labels = int(np.max(segmentation)) + 1

    v = napari.Viewer()

    v.add_image(raw)
    v.add_labels(segmentation)
    v.layers["segmentation"].show_selected_label = True

    v.title = f"Tomo: {fname}, {n_labels} vesicles"

    @magicgui(call_button="Next Vesicle [n]")
    def next_vesicle(v: napari.Viewer):
        nonlocal next_id

        # Check if we are at the end.
        if next_id == n_labels:
            show_info("All segmented vesicles inspected.")
            return

        layer = v.layers["segmentation"]
        mask = layer.data == next_id

        if mask.sum() == 0:
            next_id = next_id + 1
            next_vesicle(v)
            return

        layer.selected_label = next_id
        plane = int(np.round(np.mean(np.where(mask)[0]), 0))

        # Set the viewer to the plane.
        v.dims.current_step = [plane, 0, 0]

        next_id = next_id + 1

    @magicgui(call_button="Last Vesicle")
    def last_vesicle(v: napari.Viewer):
        nonlocal next_id

        # Check if we are at the end.
        if next_id == 1:
            show_info("Cannot go further back.")
            return

        next_id = next_id - 2

        layer = v.layers["segmentation"]
        mask = layer.data == next_id

        if mask.sum() == 0:
            return

        layer.selected_label = next_id
        plane = int(np.round(np.mean(np.where(mask)[0]), 0))

        # Set the viewer to the plane.
        v.dims.current_step = [plane, 0, 0]

        next_id = next_id + 1

    @magicgui(call_button="Save Correction")
    def save_correction(v: napari.Viewer):
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        labels = v.layers["segmentation"].data
        labels = label(labels)
        with h5py.File(output_path, "a") as f:
            ds = f.require_dataset(
                "vesicles/corrected", shape=labels.shape, dtype=labels.dtype, compression="gzip"
            )
            ds[:] = labels
        show_info(f"Saved segmentation to {output_path}.")

    @magicgui(call_button="Paint New Vesicle [p]")
    def paint_new_vesicle(v: napari.Viewer):
        layer = v.layers["segmentation"]
        paint_label = int(layer.data.max()) + 1
        layer.selected_label = paint_label
        layer.mode = "paint"

    @magicgui(call_button="Toggle [t]")
    def toggle(v: napari.Viewer):
        layer = v.layers["segmentation"]
        vis = deepcopy(layer.visible)

        seg_mode = deepcopy(layer.mode)
        layer.visible = not vis

        # toggle on: restore the previous mode
        layer.mode = seg_mode

    @magicgui(call_button="Stop Correction")
    def stop_correction(v: napari.Viewer):
        nonlocal continue_correction
        show_info("Stop correction.")
        continue_correction = False

    v.window.add_dock_widget(next_vesicle)
    v.window.add_dock_widget(last_vesicle)
    v.window.add_dock_widget(toggle)
    v.window.add_dock_widget(paint_new_vesicle)
    v.window.add_dock_widget(save_correction)
    v.window.add_dock_widget(stop_correction)

    v.bind_key("n", lambda _:  next_vesicle(v))
    v.bind_key("t", lambda _:  toggle(v))
    v.bind_key("p", lambda _:  paint_new_vesicle(v))

    napari.run()

    return continue_correction


def correct_vesicles(version, output_root):
    data_root = get_root(version)

    files = natsorted(glob(os.path.join(data_root, "**/*.h5")))

    for path in files:
        fname = os.path.relpath(path, data_root)

        output_path = os.path.join(output_root, fname)
        if os.path.exists(output_path):
            continue

        if not run_correction(path, output_path, fname):
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", default=1, type=int)
    args = parser.parse_args()

    output_root = f"./corrections/v{args.version}"
    correct_vesicles(args.version, output_root)


if __name__ == "__main__":
    main()
