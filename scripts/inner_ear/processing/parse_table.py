import os
import warnings

import numpy as np
import pandas as pd


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

        try:
            path = os.path.join(
                data_root, prefix, row.Bedingung, f"Mouse {int(row.Maus)}",
                row["Ribbon-Orientierung"].lower(), str(int(row["OwnCloud-Unterordner"]))
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


def main():
    table_path = "./Übersicht.xlsx"
    data_root = "/scratch-emmy/usr/nimcpape/data/moser"
    parse_table(table_path, data_root)


if __name__ == "__main__":
    main()
