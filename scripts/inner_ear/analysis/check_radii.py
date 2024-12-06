import os
import sys
from pathlib import Path

import imageio
import napari
import pandas as pd
from elf.io import open_file

from tqdm import tqdm
from synapse_net.file_utils import get_data_path

sys.path.append("../processing")


def visualize_folder(folder):
    result_folder = os.path.join(folder, "manuell")
    if not os.path.exists(result_folder):
        result_folder = os.path.join(folder, "Manuell")
    if not os.path.exists(result_folder):
        return

    raw_path = get_data_path(folder)
    with open_file(raw_path, "r") as f:
        tomo = f["data"][:]

    fname = Path(raw_path).stem
    auto_path = os.path.join(folder, "automatisch", "v2", f"{fname}_vesicles.h5")
    man_path = os.path.join(result_folder, "Vesikel.tif")

    with open_file(auto_path, "r") as f:
        automatic = f["segmentation"][:]
    manual = imageio.imread(man_path)

    v = napari.Viewer()
    v.add_image(tomo)
    v.add_labels(manual)
    v.add_labels(automatic)
    v.title = folder
    napari.run()


def visualize_all_radii():
    from parse_table import parse_table, get_data_root

    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")

    table = parse_table(table_path, data_root)

    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue
        visualize_folder(folder)


def check_diameter_results():
    diam_auto = pd.read_excel("./results/vesicle_diameters_tomos_with_manual_annotations.xlsx", sheet_name="Proofread")
    diam_man = pd.read_excel("./results/vesicle_diameters_tomos_with_manual_annotations.xlsx", sheet_name="Manual")

    print("Summary for the manual measurements:")
    print(diam_man["diameter [nm]"].mean(), "+-", diam_man["diameter [nm]"].std())

    print("Summary for the auto measurements:")
    print(diam_auto["diameter [nm]"].mean(), "+-", diam_auto["diameter [nm]"].std())

    print("Unique values")
    print("Manual:", len(pd.unique(diam_man["diameter [nm]"])), "/", len(diam_man))
    print("Automatic:", len(pd.unique(diam_auto["diameter [nm]"])), "/", len(diam_auto))

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, sharex=True)
    sns.histplot(data=diam_man, x="diameter [nm]", bins=16, kde=False, ax=axes[0])
    sns.histplot(data=diam_auto, x="diameter [nm]", bins=16, kde=False, ax=axes[1])
    plt.show()


def test_export():
    from synapse_net.imod.to_imod import write_segmentation_to_imod_as_points
    from subprocess import run

    mrc_path = "/home/pape/Work/data/moser/em-synapses/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/pillar/1/Emb71M1aGridA3sec3pil12.rec"  # noqa
    seg_path = "/home/pape/Work/data/moser/em-synapses/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/pillar/1/automatisch/v2/Emb71M1aGridA3sec3pil12_vesicles.h5"  # noqa
    out_path = "exported_vesicles.mod"

    with open_file(seg_path, "r") as f:
        seg = f["segmentation"][:]

    write_segmentation_to_imod_as_points(
        mrc_path, seg, out_path, min_radius=10, radius_factor=0.85
    )
    run(["imod", mrc_path, out_path])


def main():
    # test_export()
    check_diameter_results()
    # visualize_all_radii()


if __name__ == "__main__":
    main()
