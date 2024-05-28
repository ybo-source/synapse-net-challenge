import os
import warnings

import numpy as np
import pandas as pd


def get_data_root():
    root = os.path.join(os.path.expanduser("~"), "data")
    hostname = os.uname()[1]
    if hostname == "pc-kreshuk11":
        root = "/home/pape/Work/data/moser/em-synapses"
    elif hostname == "glogin9":
        root = "/scratch-emmy/usr/nimcpape/data/moser"
    elif "ggpu" in hostname:
        root = "/scratch-emmy/usr/nimcpape/data/moser"
    return root


def parse_table(table_path, data_root, require_local_path=True):
    table = pd.read_excel(table_path)
    local_paths = []

    prefix = "Electron-Microscopy-Susi/Analyse"
    for i, row in table.iterrows():
        # Skip empty rows.
        if row.Bedingung == "":
            local_paths.append("")
            continue
        if isinstance(row.Bedingung, float) and np.isnan(row.Bedingung):
            local_paths.append("")
            continue
        if isinstance(row["OwnCloud-Unterordner"], float) and np.isnan(row["OwnCloud-Unterordner"]):
            local_paths.append("")
            continue

        try:
            path = os.path.join(
                data_root, prefix, row.Bedingung, f"Mouse {int(row.Maus)}",
                row["Ribbon-Orientierung"].lower().rstrip("?"), str(int(row["OwnCloud-Unterordner"]))
            )
        except ValueError as e:
            print(i, ":", row)
            raise e

        if not os.path.exists(path):
            if require_local_path:
                raise RuntimeError(f"Cannot find {path}")
            else:
                warnings.warn(f"Cannot find {path}")
        local_paths.append(path)

    assert len(local_paths) == len(table)
    table["Local Path"] = local_paths
    table = table.dropna(how="all")
    return table


def _match_correction_folder(folder):
    possible_names = ["Korrektur", "korrektur", "korektur", "Korektur"]
    for name in possible_names:
        correction_folder = os.path.join(folder, name)
        if os.path.exists(correction_folder):
            return correction_folder
    return folder


def _match_correction_file(correction_folder, seg_name):
    possible_names = [seg_name, seg_name.lower()]
    for name in possible_names:
        correction_file = os.path.join(correction_folder, f"{name}.tif")
        if os.path.exists(correction_file):
            return correction_file
    return correction_file


def main():
    table_path = "./Übersicht.xlsx"
    data_root = "/scratch-emmy/usr/nimcpape/data/moser"
    parse_table(table_path, data_root)


if __name__ == "__main__":
    main()
