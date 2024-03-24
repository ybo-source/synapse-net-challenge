import os
import sys
from glob import glob

import h5py
import napari

from synaptic_reconstruction.file_utils import get_data_path

from elf.io import open_file
from tqdm import tqdm

sys.path.append("processing")


def visualize_folder(folder, segmentation_version):
    raw_path = get_data_path(folder)

    if segmentation_version is None:
        with open_file(raw_path, "r") as f:
            tomo = f["data"][:]

        v = napari.Viewer()
        v.add_image(tomo)
        v.title = folder
        napari.run()

    else:
        seg_folder = os.path.join(folder, "automatisch", "v1")
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

        v = napari.Viewer()
        v.add_image(tomo)
        for name, seg in segmentations.items():
            v.add_labels(seg, name=name)
        v.title = folder
        napari.run()


def visualize_all_data(data_root, table, segmentation_version=None, check_micro=None):
    assert check_micro in ["new", "old", "both", None]

    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        micro = row["EM alt vs. Neu"]
        if micro == "alt" and check_micro in ("old", "both", None):
            visualize_folder(folder, segmentation_version)

        elif micro == "neu" and ("new", "both", None):
            visualize_folder(folder, segmentation_version)

        elif micro == "beides":
            if check_micro in ("old", "both", None):
                visualize_folder(folder, segmentation_version)
            if check_micro in ("new", "both", None):
                folder_new = os.path.join(folder, "Tomo neues EM")
                if not os.path.exists(folder_new):
                    folder_new = os.path.join(folder, "neues EM")
                assert os.path.exists(folder_new), folder_new
                visualize_folder(folder_new, segmentation_version)


# TODO distance visualization
def main():
    from parse_table import parse_table

    data_root = "/home/pape/Work/data/moser/em-synapses"

    table_path = "./processing/Übersicht.xlsx"
    table = parse_table(table_path, data_root)
    segmentation_version = 1

    visualize_all_data(
        data_root, table, segmentation_version=segmentation_version, check_micro="new"
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
