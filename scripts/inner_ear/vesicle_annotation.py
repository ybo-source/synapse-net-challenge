import argparse
import os
import sys
from pathlib import Path
from glob import glob

import h5py
import imageio.v3 as imageio
import numpy as np
import napari
import pandas as pd

from elf.io import open_file
from magicgui import magicgui
from napari.utils.notifications import show_info
from skimage.measure import label
from skimage.segmentation import relabel_sequential
from skimage.transform import resize

sys.path.append("processing")
from parse_table import get_data_root  # noqa

ROOT = os.path.join(get_data_root(), "Electron-Microscopy-Susi")
OUTPUT_ROOT = os.path.join(ROOT, "Analyse", "for_annotation")


def _size_filter(segmentation, min_size):
    ids, sizes = np.unique(segmentation, return_counts=True)
    filter_ids = ids[sizes < min_size]
    segmentation[np.isin(segmentation, filter_ids)] = 0
    return segmentation


def create_data_for_vesicle_annotation(view=True):
    from synapse_net.ground_truth import find_additional_objects

    table = pd.read_excel(os.path.join(ROOT, "Validierungs-Tabelle-v3.xlsx"))

    # Select only tomograms from the old microscope.
    mask = (table["EM alt vs. Neu"] == "alt") | (table["EM alt vs. Neu"] == "beides")
    table = table[mask]

    # Select all tomogrms that are fully corrected.
    mask = (table["Kommentar 08.09.24"] == "passt") | (table["Kommentar 08.09.24"] == "Passt")
    table = table[mask]

    # Divide by different conditions.
    table["condition"] = table["Bedingung"] + "/" + "Mouse " + table["Maus"].astype(str) + "/" + table["Ribbon-Orientierung"]  # noqa
    conditions = pd.unique(table["condition"])

    halo = (64, 384, 384)

    # Select the first 3 tomograms per condition.
    n_per_condition = 3
    for condition in conditions:
        print("Select tomograms from condition:", condition)
        count = 0
        sub_table = table[table["condition"] == condition]
        for _, row in sub_table.iterrows():
            folder = os.path.join(ROOT, "Analyse", row.condition, str(row["OwnCloud-Unterordner"]))
            assert os.path.exists(folder), folder
            files = glob(os.path.join(folder, "*.rec")) + glob(os.path.join(folder, "*.mrc"))
            assert len(files) == 1, folder
            tomo_file = files[0]

            corrected_vesicle_path = os.path.join(folder, "korrektur", "vesicle_pools.tif")
            if not os.path.exists(corrected_vesicle_path):
                continue

            seg_paths = glob(os.path.join(folder, "automatisch", "v2", "*.h5"))
            vesicle_path = [path for path in seg_paths if path.endswith("_vesicles.h5")]
            assert len(vesicle_path) == 1
            vesicle_path = vesicle_path[0]

            with open_file(tomo_file, "r") as f:
                shape = f["data"].shape

            corrected_vesicles = imageio.imread(corrected_vesicle_path)
            corrected_vesicles = resize(
                corrected_vesicles, shape, order=0, anti_aliasing=False, preserve_range=True
            ).astype(corrected_vesicles.dtype)
            assert corrected_vesicles.shape == shape

            fg_coords = np.where(corrected_vesicles != 0)
            center = [int(np.mean(coord)) for coord in fg_coords]
            bb = tuple(slice(
                max(0, ce - ha), min(sh, ce + ha)
            ) for ce, ha, sh in zip(center, halo, shape))

            corrected_vesicles = corrected_vesicles[bb]
            with open_file(tomo_file, "r") as f:
                raw = f["data"][bb]
            assert corrected_vesicles.shape == raw.shape

            with open_file(vesicle_path, "r") as f:
                assert f["/segmentation"].shape == shape
                vesicles = f["/segmentation"][bb]

            # Apply size filter to the vesicle segmentations.
            corrected_vesicles = _size_filter(corrected_vesicles, min_size=250)
            vesicles = _size_filter(vesicles, min_size=250)

            # Match segmentations
            additional_vesicles = find_additional_objects(corrected_vesicles, vesicles)

            # Create the segmentations for annotation.
            for_annotation = corrected_vesicles.copy()
            for_annotation = relabel_sequential(for_annotation)[0]
            mask = (for_annotation > 0).astype("uint8")
            offset = int(for_annotation.max())
            new_mask = additional_vesicles != 0
            for_annotation[new_mask] = (additional_vesicles[new_mask] + offset)

            if view:
                v = napari.Viewer()
                v.add_image(raw)
                v.add_labels(for_annotation)
                v.add_labels(mask)
                v.add_labels(corrected_vesicles, visible=False)
                v.add_labels(vesicles, visible=False)
                v.add_labels(additional_vesicles, visible=False)
                napari.run()
            else:
                # Store output
                os.makedirs(OUTPUT_ROOT, exist_ok=True)
                fname = f"{row.condition.replace(' ', '-').replace('/', '_')}_{row['OwnCloud-Unterordner']}.h5"
                output_path = os.path.join(OUTPUT_ROOT, fname)
                print("Writing output to", output_path)
                with h5py.File(output_path, "a") as f:
                    f.create_dataset("raw", data=raw, compression="gzip")
                    f.create_dataset("labels/vesicles", data=for_annotation, compression="gzip")
                    f.create_dataset("labels/correction_mask", data=mask, compression="gzip")

            count += 1
            if count == n_per_condition:
                break


def correct_volume(path, show_corrected):
    fname = Path(path).stem

    initial_vesicles = None
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]

        if "corrected" in f:
            if show_corrected:
                vesicles = f["corrected/vesicles"][:]
                correction_mask = f["corrected/correction_mask"][:]

                initial_vesicles = f["labels/vesicles"][:]
            else:
                print("Skipping annotations for", fname, "because corrections are already saved.")
                return True
        else:
            vesicles = f["labels/vesicles"][:]
            correction_mask = f["labels/correction_mask"][:]

    continue_correction = True

    v = napari.Viewer()

    v.add_image(raw)
    v.add_labels(vesicles)
    v.add_labels(correction_mask)
    if initial_vesicles is not None:
        v.add_labels(initial_vesicles, visible=False)
    v.title = fname

    @magicgui(call_button="Save Correction")
    def save_correction(v: napari.Viewer):

        labels = v.layers["vesicles"].data
        labels = label(labels)
        mask = v.layers["correction_mask"].data

        with h5py.File(path, "a") as f:
            ds = f.require_dataset(
                "corrected/vesicles", shape=labels.shape, dtype=labels.dtype, compression="gzip"
            )
            ds[:] = labels

            ds = f.require_dataset(
                "corrected/correction_mask", shape=mask.shape, dtype=mask.dtype, compression="gzip"
            )
            ds[:] = mask

        show_info(f"Saved segmentation to {path}.")

    @magicgui(call_button="Stop Correction")
    def stop_correction(v: napari.Viewer):
        nonlocal continue_correction
        show_info("Stop correction.")
        continue_correction = False

    @magicgui(call_button="Mark Vesicle [m]")
    def mark_vesicle_as_corrected(v: napari.Viewer):
        labels = v.layers["vesicles"].data
        mask = v.layers["correction_mask"].data
        vesicle_id = v.layers["vesicles"].selected_label
        mask[labels == vesicle_id] = 1
        v.layers["correction_mask"].data = mask
        v.layers["correction_mask"].refresh()

    v.window.add_dock_widget(mark_vesicle_as_corrected)
    v.window.add_dock_widget(save_correction)
    v.window.add_dock_widget(stop_correction)

    v.bind_key("m", lambda _:  mark_vesicle_as_corrected(v))

    napari.run()

    return continue_correction


def _run_correction(show_corrected):
    files = sorted(glob(os.path.join(OUTPUT_ROOT, "*.h5")))
    for path in files:
        continue_correction = correct_volume(path, show_corrected)
        if not continue_correction:
            break


def run_correction():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_corrected", "-s", action="store_true")
    args = parser.parse_args()
    _run_correction(args.show_corrected)


def main():
    # create_data_for_vesicle_annotation(view=False)
    run_correction()


if __name__ == "__main__":
    main()
