import argparse

import pandas as pd


def add_summary_stats(table_path):
    vesicle_table = pd.read_excel(table_path, sheet_name="vesicles")
    morpho_table = pd.read_excel(table_path, sheet_name="morphology")

    tomograms = pd.unique(vesicle_table.tomogram)

    boundary_dists = {"All": [], "RA-V": [], "MP-V": [], "Docked-V": []}
    pd_dists = {"All": [], "RA-V": [], "MP-V": [], "Docked-V": []}
    ribbon_dists = {"All": [], "RA-V": [], "MP-V": [], "Docked-V": []}
    radii = {"All": [], "RA-V": [], "MP-V": [], "Docked-V": []}

    n_ravs, n_mpvs, n_dockeds = [], [], []
    ves_per_surfs = []
    for tomo in tomograms:
        tomo_table = vesicle_table[vesicle_table.tomogram == tomo]

        rav_mask = tomo_table.pool == "RA-V"
        mpv_mask = tomo_table.pool == "MP-V"
        docked_mask = tomo_table.pool == "Docked-V"

        masks = {"All": tomo_table.pool != "", "RA-V": rav_mask, "MP-V": mpv_mask, "Docked-V": docked_mask}

        for pool, mask in masks.items():
            pool_table = tomo_table[mask]
            radii[pool].append(pool_table["radius [nm]"].mean())
            ribbon_dists[pool].append(pool_table["ribbon_distance [nm]"].mean())
            pd_dists[pool].append(pool_table["pd_distance [nm]"].mean())
            boundary_dists[pool].append(pool_table["boundary_distance [nm]"].mean())

        n_rav = rav_mask.sum()
        n_mpv = mpv_mask.sum()
        n_docked = docked_mask.sum()

        n_ves = n_rav + n_mpv + n_docked
        tomo_table = morpho_table[morpho_table.tomogram == tomo]
        ribbon_surface = tomo_table[tomo_table.structure == "ribbon"]["surface [nm^2]"].values[0]
        ves_per_surface = n_ves / ribbon_surface

        n_ravs.append(n_rav)
        n_mpvs.append(n_mpv)
        n_dockeds.append(n_docked)
        ves_per_surfs.append(ves_per_surface)

    summary = {
        "tomogram": tomograms,
        "N_RA-V": n_ravs,
        "N_MP-V": n_mpvs,
        "N_Docked-V": n_dockeds,
        "Vesicles / Surface [1 / nm^2]": ves_per_surfs,
    }
    summary.update({f"{pool}: radius [nm]": dists for pool, dists in radii.items()})
    summary.update({f"{pool}: ribbon_distance [nm]": dists for pool, dists in ribbon_dists.items()})
    summary.update({f"{pool}: pd_distance [nm]": dists for pool, dists in pd_dists.items()})
    summary.update({f"{pool}: boundary_distance [nm]": dists for pool, dists in boundary_dists.items()})
    summary = pd.DataFrame(summary)

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
