import argparse

import pandas as pd


def add_summary_stats(table_path):
    vesicle_table = pd.read_excel(table_path, sheet_name="vesicles")
    morpho_table = pd.read_excel(table_path, sheet_name="morphology")

    tomograms = pd.unique(vesicle_table.tomogram)

    n_ravs, n_mpvs, n_dockeds = [], [], []
    ves_per_surfs = []
    for tomo in tomograms:
        tomo_table = vesicle_table[vesicle_table.tomogram == tomo]

        n_rav = (tomo_table.pool == "RA-V").sum()
        n_mpv = (tomo_table.pool == "MP-V").sum()
        n_docked = (tomo_table.pool == "Docked-V").sum()

        n_ves = n_rav + n_mpv + n_docked
        tomo_table = morpho_table[morpho_table.tomogram == tomo]
        ribbon_surface = tomo_table[tomo_table.structure == "ribbon"]["surface [nm^2]"].values[0]
        ves_per_surface = n_ves / ribbon_surface

        n_ravs.append(n_rav)
        n_mpvs.append(n_mpv)
        n_dockeds.append(n_docked)
        ves_per_surfs.append(ves_per_surface)

    summary = pd.DataFrame({
        "tomogram": tomograms,
        "N_RA-V": n_ravs,
        "N_MP-V": n_mpvs,
        "N_Docked-V": n_dockeds,
        "Vesicles / Surface [1 / nm^2]": ves_per_surfs,
    })

    with pd.ExcelWriter(table_path, engine="openpyxl", mode="a") as writer:
        summary.to_excel(writer, sheet_name="vesicle_statistics", index=False)


# compute summary statistics:
# - vesicles / surface
# - number of vesicles per pool
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("table")
    args = parser.parse_args()

    add_summary_stats(args.table)


if __name__ == "__main__":
    main()
