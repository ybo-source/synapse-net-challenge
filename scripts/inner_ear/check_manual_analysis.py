import os
import sys
from glob import glob

import imageio.v3 as imageio
import napari
import numpy as np

from synapse_net.file_utils import get_data_path

from elf.io import open_file
from tqdm import tqdm

from napari_skimage_regionprops import add_table
from synapse_net.tools.distance_measurement import _downsample
from check_results import get_distance_visualization, create_vesicle_pools, _update_colors

sys.path.append("processing")


def _load_segmentation(seg_file, binning, tomo):
    seg = imageio.imread(seg_file)
    if seg.dtype == np.dtype("uint64"):
        seg = seg.astype("uint32")
    seg = _downsample(seg, is_seg=True, scale=binning, target_shape=tomo.shape)
    return seg


def _get_pool_colors(seg):
    colors = {
        1: (0, 0.33, 0),
        2: (1, 0.549, 0),
        3: (1, 1, 0),
    }
    return colors


def visualize_folder(folder, binning):
    result_folder = os.path.join(folder, "manuell")
    result_path = os.path.join(result_folder, "measurements.xlsx")
    if not os.path.exists(result_path):
        return

    raw_path = get_data_path(folder)
    with open_file(raw_path, "r") as f:
        tomo = f["data"][:]
    tomo = _downsample(tomo, scale=binning)

    seg_files = glob(os.path.join(result_folder, "*.tif"))
    segmentations = {}
    for seg_file in seg_files:
        if "refined_vesicles" in seg_file:
            seg_name = "refined_vesicles"
        else:
            seg_name = os.path.basename(seg_file).split("_")[-1].rstrip(".tif").lower()
        seg = _load_segmentation(seg_file, binning, tomo)
        if seg_name == "vesikel":
            segmentations["vesicles"] = seg
        elif "ribbon" in seg_name:
            segmentations["ribbon"] = seg
        else:
            segmentations[seg_name] = seg

    segmentations["vesicle_pools"], vesicle_ids, colors = create_vesicle_pools(
        segmentations["vesicles"], result_path
    )
    colors["pools"] = _get_pool_colors(segmentations["pools"])
    colors = _update_colors(colors)

    distance_folder = os.path.join(result_folder, "distances")
    distance_files = {
        name: os.path.join(distance_folder, f"{name}.npz") for name in ["ribbon", "PD", "membrane"]
    }

    tomo, segmentations, distance_lines, mem_props = get_distance_visualization(
        tomo, segmentations, distance_files, vesicle_ids, scale=binning, return_mem_props=True
    )
    mem_props["distance"] = np.round(mem_props["distance"], 2)

    v = napari.Viewer()
    v.add_image(tomo)
    for name, seg in segmentations.items():
        try:
            v.add_labels(seg, name=name, color=colors.get(name, None))
        except TypeError:
            v.add_labels(seg, name=name, colormap=colors.get(name, None))

    for name, lines in distance_lines.items():
        layer = v.add_shapes(
            lines, shape_type="line", name=name, visible=False, properties=mem_props,
        )
        if name == "membrane_distances":
            add_table(layer, v)

    v.title = folder
    napari.run()


def isint(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


def _get_bin_factor(is_old, binning):
    if isint(binning):
        return int(binning)
    elif binning == "none":
        return None
    else:
        assert binning == "auto"
        return 2 if is_old else 3


def visualize_all_data(data_root, table, binning="auto", show_micro=None):
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        # if skip_iteration is not None and i < skip_iteration:
        #     continue

        micro = row["EM alt vs. Neu"]
        if micro == "alt" and show_micro in (None, "both", "alt", "old"):
            binning_ = _get_bin_factor(True, binning)
            visualize_folder(folder, binning_)

        elif micro == "neu" and show_micro in (None, "both", "neu", "new"):
            binning_ = _get_bin_factor(False, binning)
            visualize_folder(folder, binning_)

        elif micro == "beides":
            if show_micro in (None, "both", "alt", "old"):
                binning_ = _get_bin_factor(True, binning)
                visualize_folder(folder, binning_)

            if show_micro in (None, "both", "neu", "new"):
                folder_new = os.path.join(folder, "Tomo neues EM")
                if not os.path.exists(folder_new):
                    folder_new = os.path.join(folder, "neues EM")
                assert os.path.exists(folder_new), folder_new
                binning_ = _get_bin_factor(False, binning)
                visualize_folder(folder_new, binning_)


def main():
    import argparse
    from parse_table import parse_table, get_data_root

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--binning", default="auto")
    parser.add_argument("-m", "--micro")
    args = parser.parse_args()

    binning = args.binning
    assert (binning in ("none", "auto") or isint(binning))

    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")

    table = parse_table(table_path, data_root)

    visualize_all_data(data_root, table, binning=binning, show_micro=args.micro)


if __name__ == "__main__":
    main()
