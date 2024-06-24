import os
import argparse
import sys

import pandas as pd

sys.path.append("processing")


def combine_results(results, output_path, sheet_name):
    big_table = []
    for tomo_name, result in results.items():
        res = pd.read_excel(result, sheet_name=sheet_name)
        res.insert(0, "tomogram", [tomo_name] * len(res))
        big_table.append(res)
    big_table = pd.concat(big_table)

    if os.path.exists(output_path):
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="a") as writer:
            big_table.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        big_table.to_excel(output_path, sheet_name=sheet_name, index=False)


def combine_manual_results(table, data_root):
    results = {}
    for _, row in table.iterrows():
        folder = row["Local Path"]
        if folder == "":
            continue

        tomo_name = os.path.relpath(folder, os.path.join(data_root, "Electron-Microscopy-Susi/Analyse"))
        res_path = os.path.join(folder, "manuell", "measurements.xlsx")
        if not os.path.exists(res_path):
            res_path = os.path.join(folder, "Manuell", "measurements.xlsx")
        if os.path.exists(res_path):
            results[tomo_name] = res_path

    output_path = "./manual_analysis_results.xlsx"
    combine_results(results, output_path, sheet_name="vesicles")
    combine_results(results, output_path, sheet_name="morphology")


def combine_automatic_results(table, data_root):
    val_table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Validierungs-Tabelle-v3.xlsx")
    val_table = pd.read_excel(val_table_path)

    results = {}
    for _, row in table.iterrows():
        folder = row["Local Path"]
        if folder == "":
            continue

        row_selection = (val_table.Bedingung == row.Bedingung) &\
            (val_table.Maus == row.Maus) &\
            (val_table["Ribbon-Orientierung"] == row["Ribbon-Orientierung"]) &\
            (val_table["OwnCloud-Unterordner"] == row["OwnCloud-Unterordner"])
        complete_vals = val_table[row_selection]["Fertig!"].values
        is_complete = (complete_vals == "ja").all()
        if not is_complete:
            continue

        tomo_name = os.path.relpath(folder, os.path.join(data_root, "Electron-Microscopy-Susi/Analyse"))
        res_path = os.path.join(folder, "korrektur", "measurements.xlsx")
        if not os.path.exists(res_path):
            res_path = os.path.join(folder, "Korrektur", "measurements.xlsx")
        assert os.path.exists(res_path), res_path
        results[tomo_name] = res_path

    output_path = "./automatic_analysis_results.xlsx"
    combine_results(results, output_path, sheet_name="vesicles")
    combine_results(results, output_path, sheet_name="morphology")


def main():
    from parse_table import parse_table, get_data_root

    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", "-m", action="store_false")

    args = parser.parse_args()
    data_root = get_data_root()

    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    if args.manual:
        combine_manual_results(table, data_root)
    else:
        combine_automatic_results(table, data_root)


if __name__ == "__main__":
    main()
