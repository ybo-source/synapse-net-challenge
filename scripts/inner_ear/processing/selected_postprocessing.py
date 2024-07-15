import os

from parse_table import parse_table, get_data_root
from run_structure_postprocessing import postprocess_folder


def selected_postprocessing(data_root, tomograms, version, force, use_corrected_pd):
    for tomo in tomograms:
        folder = os.path.join(data_root, tomo)


def main():
    data_root = get_data_root()
    
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    version = 2
    process_new_microscope = True
    force = True

    tomograms = [
        "WT control/Mouse 1/modiolar/6",
        "WT control/Mouse 1/pillar/6",
        "WT control/Mouse 2/modiolar/7",
    ]

    selected_postprocessing(data_root, tomograms, version, force=force, use_corrected_pd=True)


if __name__ == "__main__":
    main()
