import os

from parse_table import parse_table, get_data_root
from run_structure_postprocessing import postprocess_folder


def selected_postprocessing(data_root, table, tomograms, version, force, use_corrected_pd):
    for tomo in tomograms:
        folder = os.path.join(data_root, tomo)
        assert os.path.exists(folder), folder
        row = table[table["Local Path"] == folder]

        n_ribbons = int(row["Anzahl Ribbons"].values[0])
        n_pds = row["Anzahl PDs"].values[0]
        try:
            n_pds = int(n_pds)
        except Exception:
            n_pds = 1
        is_new = row["EM alt vs. Neu"].values[0] != "alt"

        postprocess_folder(
            folder, version=version, n_ribbons=n_ribbons, n_pds=n_pds, is_new=is_new, force=force,
            use_corrected_pd=use_corrected_pd,
        )


def main():
    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    version = 2
    force = True

    tomograms = [
        "Electron-Microscopy-Susi/Analyse/WT control/Mouse 1/modiolar/6",
        "Electron-Microscopy-Susi/Analyse/WT control/Mouse 1/pillar/6",
        "Electron-Microscopy-Susi/Analyse/WT control/Mouse 2/modiolar/7",
    ]

    selected_postprocessing(data_root, table, tomograms, version, force=force, use_corrected_pd=True)


if __name__ == "__main__":
    main()
