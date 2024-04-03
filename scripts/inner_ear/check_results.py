import os
import sys
from glob import glob

import h5py
import napari
import numpy as np
import pandas

from synaptic_reconstruction.distance_measurements import create_object_distance_lines
from synaptic_reconstruction.file_utils import get_data_path
from synaptic_reconstruction.tools.distance_measurement import _downsample

from elf.io import open_file
from tqdm import tqdm

sys.path.append("processing")


def get_distance_visualization(tomo, segmentations, distance_paths, vesicle_ids, scale=2):
    tomo = _downsample(tomo, scale=scale)
    segmentations = {k: _downsample(v, is_seg=True, scale=scale) for k, v in segmentations.items()}

    ribbon_lines, _ = create_object_distance_lines(distance_paths["ribbon"], seg_ids=vesicle_ids, scale=scale)
    pd_lines, _ = create_object_distance_lines(distance_paths["PD"], seg_ids=vesicle_ids, scale=scale)
    membrane_lines, _ = create_object_distance_lines(distance_paths["membrane"], seg_ids=vesicle_ids, scale=scale)

    distance_lines = {
        "ribbon_distances": ribbon_lines,
        "pd_distances": pd_lines,
        "membrane_distances": membrane_lines
    }
    return tomo, segmentations, distance_lines


def create_vesicle_pools(vesicles, result_path):
    vesicle_pools = np.zeros_like(vesicles)

    assignment_result = pandas.read_excel(result_path)
    vesicle_ids = assignment_result["id"].values
    pool_assignments = assignment_result["pool"].values
    pool_assignments = {vid: pool for vid, pool in zip(vesicle_ids, pool_assignments)}

    for pool_id, pool_name in enumerate(("RA-V", "MP-V", "Docked-V"), 1):
        ves_ids_pool = [vid for vid, pname in pool_assignments.items() if pname == pool_name]
        vesicle_pools[np.isin(vesicles, ves_ids_pool)] = pool_id

    vesicle_ids = np.sort(vesicle_ids)
    return vesicle_pools, vesicle_ids


def visualize_folder(folder, segmentation_version, visualize_distances):
    raw_path = get_data_path(folder)

    if segmentation_version is None:
        with open_file(raw_path, "r") as f:
            tomo = f["data"][:]

        v = napari.Viewer()
        v.add_image(tomo)
        v.title = folder
        napari.run()

    else:
        seg_folder = os.path.join(folder, "automatisch", f"v{segmentation_version}")
        seg_files = glob(os.path.join(seg_folder, "*.h5"))
        if len(seg_files) == 0:
            print("No segmentations for", folder, "skipping!")

        with open_file(raw_path, "r") as f:
            tomo = f["data"][:]

        segmentations = {}
        for seg_file in seg_files:
            seg_name = seg_file.split("_")[-1].rstrip(".h5")
            with h5py.File(seg_file, "r") as f:
                seg = f["segmentation"][:] if "segmentation" in f else f["prediction"][:]
                # if "prediction" in f:
                #     seg = f["prediction"][:]
                # else:
                #     seg = f["segmentation"][:]
            segmentations[seg_name] = seg

        result_path = os.path.join(seg_folder, "measurements.xlsx")
        if os.path.exists(result_path):
            segmentations["vesicle_pools"], vesicle_ids = create_vesicle_pools(
                segmentations["vesicles"], result_path
            )
        else:
            vesicle_ids = None

        distance_folder = os.path.join(seg_folder, "distances")
        distance_files = {
            name: os.path.join(distance_folder, f"{name}.npz") for name in ["ribbon", "PD", "membrane"]
        }
        have_distances = all(os.path.exists(path) for path in distance_files.values())

        if have_distances and visualize_distances:
            assert vesicle_ids is not None
            tomo, segmentations, distance_lines = get_distance_visualization(
                tomo, segmentations, distance_files, vesicle_ids,
            )
        else:
            distance_lines = {}

        v = napari.Viewer()
        v.add_image(tomo)
        for name, seg in segmentations.items():
            v.add_labels(seg, name=name)
        for name, lines in distance_lines.items():
            v.add_shapes(lines, shape_type="line", name=name, visible=False)
        v.title = folder
        napari.run()


def visualize_all_data(
    data_root, table,
    segmentation_version=None, check_micro=None,
    visualize_distances=False, skip_iteration=None,
):
    assert check_micro in ["new", "old", "both", None]

    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        if skip_iteration is not None and i < skip_iteration:
            continue

        micro = row["EM alt vs. Neu"]
        if micro == "alt" and check_micro in ("old", "both", None):
            visualize_folder(folder, segmentation_version, visualize_distances)

        elif micro == "neu" and ("new", "both", None):
            visualize_folder(folder, segmentation_version, visualize_distances)

        elif micro == "beides":
            if check_micro in ("old", "both", None):
                visualize_folder(folder, segmentation_version, visualize_distances)
            if check_micro in ("new", "both", None):
                folder_new = os.path.join(folder, "Tomo neues EM")
                if not os.path.exists(folder_new):
                    folder_new = os.path.join(folder, "neues EM")
                assert os.path.exists(folder_new), folder_new
                visualize_folder(folder_new, segmentation_version, visualize_distances)


def main():
    import argparse
    from parse_table import parse_table

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iteration", default=None, type=int)
    parser.add_argument("-m", "--microscope", default=None)
    parser.add_argument("-d", "--visualize_distances", action="store_false")
    args = parser.parse_args()
    assert args.microscope in (None, "both", "old", "new")

    data_root = "/home/pape/Work/data/moser/em-synapses"
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    segmentation_version = 2

    visualize_all_data(
        data_root, table,
        segmentation_version=segmentation_version, check_micro=args.microscope,
        visualize_distances=args.visualize_distances, skip_iteration=args.iteration,
    )


# Tomos With Artifacts:
# Analyse/WT strong stim/Mouse 1/modiolar/14/Emb71M1aGridA3sec3mod14.rec
# Analyse/WT strong stim/Mouse 1/modiolar/18/Emb71M1aGridA1sec1mod3.rec
# Analyse/WT strong stim/Mouse 1/modiolar/8/Emb71M1aGridA1sec1mod2.rec

# New Tomos for Annotation:
# WT Strong stim/Mouse 1/ modiolar/1
# -> ribbon segmentation + PD
# WT Contriol/Mouse 2/ modiolar/5
# -> membrane segmentation
if __name__ == "__main__":
    main()
