import os
import sys
from glob import glob

import imageio.v3 as imageio
import napari
import numpy as np
import pandas

from elf.io import open_file
from synapse_net.file_utils import get_data_path
from synapse_net.tools.distance_measurement import _downsample
from tqdm import tqdm

sys.path.append("../processing")

from parse_table import parse_table, get_data_root  # noqa
from parse_table import _match_correction_folder, _match_correction_file  # noqa


def _load_segmentation(correction_file, seg_file, tomo_shape):
    if os.path.exists(correction_file):
        seg = imageio.imread(correction_file)
    else:
        with open_file(seg_file, "r") as f:
            seg = f["segmentation"][:]
    if seg.dtype == np.dtype("uint64"):
        seg = seg.astype("uint32")

    seg = _downsample(seg, is_seg=True, target_shape=tomo_shape, scale=None)
    return seg


def _read_segmentations(folder, tomo_shape):
    segmentation_version = 2

    correction_folder = _match_correction_folder(folder)
    seg_folder = os.path.join(folder, "automatisch", f"v{segmentation_version}")
    pattern = os.path.join(seg_folder, "*.h5")
    seg_files = glob(pattern)
    assert len(seg_files) > 0, pattern

    segmentations = {}
    for seg_file in seg_files:
        seg_name = seg_file.split("_")[-1].rstrip(".h5")
        correction_file = _match_correction_file(correction_folder, seg_name)

        if seg_name == "vesicles":
            vesicle_pool_path = os.path.join(correction_folder, "vesicle_pools.tif")
            if os.path.exists(vesicle_pool_path):
                correction_file = vesicle_pool_path

        seg = _load_segmentation(correction_file, seg_file, tomo_shape)
        segmentations[seg_name] = seg

    return segmentations


def _crop_and_upscale(tomo, folder):
    segmentations = _read_segmentations(folder, tomo.shape)

    halo = (16, 194, 194)

    bb_min = np.full(3, np.inf)
    bb_max = np.zeros(3)

    for seg in segmentations.values():
        fg_coords = np.where(seg != 0)
        this_bb_min = np.array([fg.min() for fg in fg_coords])
        this_bb_max = np.array([fg.max() for fg in fg_coords]) + 1

        bb_min = np.minimum(bb_min, this_bb_min)
        bb_max = np.maximum(bb_max, this_bb_max)

    bb = tuple(
        slice(
            max(0, int(bmin - ha)), min(sh, int(bmax + ha))
        )
        for bmin, bmax, sh, ha in zip(bb_min, bb_max, tomo.shape, halo)
    )

    tomo = tomo[bb]
    tomo = _downsample(tomo, is_seg=False, scale=0.5)

    for name, seg in segmentations.items():
        seg = seg[bb]
        seg = _downsample(seg, is_seg=True, scale=None, target_shape=tomo.shape)
        segmentations[name] = seg

    return tomo, segmentations


def extract_folder(folder, output_path, rescale):
    if os.path.exists(output_path):
        return

    # Read the tomogram and all segmentation results.
    tomo_path = get_data_path(folder)
    with open_file(tomo_path, "r") as f:
        tomo = f["data"][:]

    # Rescale everything to double size if this data comes from the new microscope.
    if rescale:
        print("Run crop and upscale for", folder)
        tomo, segmentations = _crop_and_upscale(tomo, folder)
    else:
        # Read the segmentations.
        segmentations = _read_segmentations(folder, tomo.shape)

    expected_names = ["vesicles", "ribbon", "PD", "membrane"]
    assert len(segmentations) == len(expected_names)
    for name in expected_names:
        assert name in segmentations, name

    if output_path is None:
        # Visualize the results.
        v = napari.Viewer()
        v.add_image(tomo)
        for name, seg in segmentations.items():
            v.add_labels(seg, name=name)
        napari.run()

    else:
        # Save the results.
        print("Exporting segmentation to", output_path)
        with open_file(output_path, "a") as f:
            f.create_dataset("raw", data=tomo, compression="gzip")
            for name, seg in segmentations.items():
                f.create_dataset(f"labels/{name}", data=seg, compression="gzip")


def extract_automatic_structures(table, val_table, output_root):
    n = 0
    table["condition"] = table["Bedingung"] + "/" + "Mouse " + table["Maus"].astype(str) + "/" + table["Ribbon-Orientierung"]  # noqa

    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        row_selection = (val_table.Bedingung == row.Bedingung) &\
            (val_table.Maus == row.Maus) &\
            (val_table["Ribbon-Orientierung"] == row["Ribbon-Orientierung"]) &\
            (val_table["OwnCloud-Unterordner"] == row["OwnCloud-Unterordner"])

        assert "Fertig 3.0?" in val_table.columns
        skip_vals = val_table[row_selection]["Fertig 3.0?"].values
        is_skip = (
            (skip_vals == "skip") |
            (skip_vals == "Anzeigefehler") |
            (skip_vals == "Ausschluss") |
            (skip_vals == "Keine PD")
        ).all()
        if is_skip:
            print("The tomogram", folder)
            print("is skipped due to", skip_vals)
            continue

        n += 1
        if output_root is None:
            output_path = None
        else:
            folder_name = f"{row.condition.replace(' ', '-').replace('/', '_')}"
            output_folder = os.path.join(output_root, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            fname = f"{int(row['OwnCloud-Unterordner'])}.h5"
            output_path = os.path.join(output_folder, fname)

        micro = row["EM alt vs. Neu"]
        if micro == "alt":
            extract_folder(folder, output_path, rescale=False)

        elif micro == "neu":
            extract_folder(folder, output_path, rescale=True)

        elif micro == "beides":
            extract_folder(folder, output_path, rescale=False)

            if output_root is not None:
                output_path = output_path[:-3] + "_new.h5"

            folder_new = os.path.join(folder, "Tomo neues EM")
            if not os.path.exists(folder_new):
                folder_new = os.path.join(folder, "neues EM")
            assert os.path.exists(folder_new), folder_new
            extract_folder(folder_new, output_path, rescale=True)

    print("Number of exported tomograms:", n)


def main():
    data_root = get_data_root()
    output_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/inner_ear_data"

    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    val_table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Validierungs-Tabelle-v3.xlsx")
    val_table = pandas.read_excel(val_table_path)

    extract_automatic_structures(table, val_table, output_root)


if __name__ == "__main__":
    main()
