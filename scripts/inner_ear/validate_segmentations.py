import os
import sys
from pathlib import Path

import pandas

from synapse_net.file_utils import get_data_path

from elf.io import open_file
from tqdm import tqdm

sys.path.append("processing")


def validate_folder(folder, segmentation_version):
    raw_path = get_data_path(folder)
    seg_folder = os.path.join(folder, "automatisch", "v1")

    fname = Path(raw_path).stem
    segmentation_names = ["vesicles", "ribbon", "PD", "membrane"]
    segmentation_status = {}

    for name in segmentation_names:
        seg_path = os.path.join(seg_folder, f"{fname}_{name}.h5")
        if not os.path.exists(seg_path):
            segmentation_status[name] = ["Nicht Segmentiert"]
            continue
        with open_file(seg_path, "r") as f:
            seg = f["segmentation"][:]
        if seg.sum() == 0:
            segmentation_status[name] = ["Leere Segmentierung"]
        else:
            segmentation_status[name] = [""]

    segmentation_status = pandas.DataFrame(segmentation_status)
    return segmentation_status


def validate_segmentations(data_root, table, segmentation_version):
    new_rows = []
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        # if i > 4:
        #     break

        micro = row["EM alt vs. Neu"]
        if micro == "alt":
            segmentation_status = validate_folder(folder, segmentation_version)

        elif micro == "neu":
            segmentation_status = validate_folder(folder, segmentation_version)

        elif micro == "beides":
            segmentation_status = validate_folder(folder, segmentation_version)
            # folder_new = os.path.join(folder, "Tomo neues EM")
            # if not os.path.exists(folder_new):
            #     folder_new = os.path.join(folder, "neues EM")
            # assert os.path.exists(folder_new), folder_new
            # validate_folder(folder_new, segmentation_version)

        new_row = {
            "Bedingung": [row["Bedingung"]], "Maus": [row["Maus"]], "PD vorhanden?": [row["PD vorhanden? "]],
            "Dateiname": [row["Dateiname"]],
        }
        new_row.update(segmentation_status)
        new_row["Kommentar"] = [""]
        new_rows.append(pandas.DataFrame(new_row))

    output_path = "Validierung_v1.xlsx"
    segmentation_val = pandas.concat(new_rows)
    segmentation_val.to_excel(output_path, index=False)


# TODO start from existing val table
def main():
    from parse_table import parse_table

    data_root = "/home/pape/Work/data/moser/em-synapses"

    table_path = "./processing/Übersicht.xlsx"
    table = parse_table(table_path, data_root)
    segmentation_version = 1

    validate_segmentations(data_root, table, segmentation_version)


if __name__ == "__main__":
    main()
