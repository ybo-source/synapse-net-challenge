import os
from shutil import rmtree


def clear_folder(folder):
    from parse_table import _match_correction_folder
    correction_folder = _match_correction_folder(folder)

    if not os.path.exists(correction_folder):
        return

    ves_pool_path = os.path.join(correction_folder, "vesicle_pools.tif")
    if not os.path.exists(ves_pool_path):
        return

    print("Clearing the vesicle pools for", correction_folder)

    try:
        rmtree(os.path.join(correction_folder, "distances"))
    except Exception:
        pass

    try:
        os.remove(os.path.join(correction_folder, "measurements.xlsx"))
    except Exception:
        pass


def clear_vesicle_pools(data_root, table):

    for i, row in table.iterrows():
        folder = row["Local Path"]
        if folder == "":
            continue

        micro = row["EM alt vs. Neu"]
        if micro == "alt":
            clear_folder(folder)

        elif micro == "neu":
            clear_folder(folder)

        elif micro == "beides":
            clear_folder(folder)
            folder_new = os.path.join(folder, "Tomo neues EM")
            clear_folder(folder_new)


def main():
    from parse_table import parse_table, get_data_root

    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")

    table = parse_table(table_path, data_root)

    clear_vesicle_pools(data_root, table)


if __name__ == "__main__":
    main()
