# import os
import sys

import pandas as pd

sys.path.append("../processing")

from parse_table import get_data_root  # noqa


def get_manual_assignments():
    result_path = "../results/20240917_1/fully_manual_analysis_results.xlsx"
    results = pd.read_excel(result_path)
    return results


def get_semi_automatic_assignments(tomograms):
    result_path = "../results/20240917_1/automatic_analysis_results.xlsx"
    results = pd.read_excel(result_path)
    results = results[results["tomogram"].isin(tomograms)]
    return results


def get_automatic_assignments(tomograms):
    result_path = "../results/fully_automatic_analysis_results.xlsx"
    results = pd.read_excel(result_path)
    results = results[results["tomogram"].isin(tomograms)]
    return results


def get_measurements_with_annotation():
    manual_assignments = get_manual_assignments()
    manual_tomograms = pd.unique(manual_assignments["tomogram"])
    semi_automatic_assignments = get_semi_automatic_assignments(manual_tomograms)

    tomograms = pd.unique(semi_automatic_assignments["tomogram"])
    manual_assignments = manual_assignments[manual_assignments["tomogram"].isin(tomograms)]
    assert len(pd.unique(manual_assignments["tomogram"])) == len(pd.unique(semi_automatic_assignments["tomogram"]))

    automatic_assignments = get_automatic_assignments(tomograms)
    filtered_tomograms = pd.unique(manual_assignments["tomogram"])
    assert len(filtered_tomograms) == len(pd.unique(automatic_assignments["tomogram"]))

    print("Tomograms with manual annotations:", len(filtered_tomograms))
    return manual_assignments, semi_automatic_assignments, automatic_assignments


def get_all_measurements():
    # data_root = get_data_root()
    # val_table = os.path.join(data_root, "Electron-Microscopy-Susi", "Validierungs-Tabelle-v3.xlsx")

    val_table = "/home/pape/Desktop/sfb1286/mboc_synapse/misc/Validierungs-Tabelle-v3-passt.xlsx"
    val_table = pd.read_excel(val_table)

    val_table = val_table[val_table["Kommentar 22.11.24"] == "passt"]
    n_tomos = len(val_table)
    print("All tomograms:", n_tomos)
    assert n_tomos > 0
    tomo_names = []
    for _, row in val_table.iterrows():
        name = "/".join([
            row.Bedingung, f"Mouse {int(row.Maus)}",
            row["Ribbon-Orientierung"].lower().rstrip("?"),
            str(int(row["OwnCloud-Unterordner"]))]
        )
        tomo_names.append(name)

    semi_automatic_assignments = get_semi_automatic_assignments(tomo_names)
    filtered_tomo_names = pd.unique(semi_automatic_assignments["tomogram"]).tolist()

    automatic_assignments = get_automatic_assignments(tomo_names)
    assert len(filtered_tomo_names) == len(pd.unique(automatic_assignments["tomogram"]))

    return semi_automatic_assignments, automatic_assignments


def main():
    get_measurements_with_annotation()
    get_all_measurements()


if __name__ == "__main__":
    main()
