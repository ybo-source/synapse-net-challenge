import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import get_all_measurements, get_measurements_with_annotation


def _plot_all(distances):
    pools = pd.unique(distances["pool"])
    dist_cols = ["ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]

    fig, axes = plt.subplots(3, 3)

    # multiple = "stack"
    multiple = "layer"

    structures = ["Ribbon", "PD", "Boundary"]
    for i, pool in enumerate(pools):
        pool_distances = distances[distances["pool"] == pool]
        for j, dist_col in enumerate(dist_cols):
            ax = axes[i, j]
            ax.set_title(f"{pool} to {structures[j]}")
            sns.histplot(
                data=pool_distances, x=dist_col, hue="approach", multiple=multiple, kde=False, ax=ax
            )
            ax.set_xlabel("distance [nm]")

    fig.tight_layout()
    plt.show()


# TODO rename the method names.
# We only care about the following distances:
# - MP-V -> PD, AZ (Boundary)
# - Docked-V -> PD
# - RA-V -> Ribbon
def _plot_selected(distances, save_path=None):
    fig, axes = plt.subplots(2, 2)
    multiple = "layer"

    if save_path is not None and os.path.exists(save_path):
        os.remove(save_path)

    def _plot(pool_name, distance_col, structure_name, ax):

        this_distances = distances[distances["pool"] == pool_name][["approach", distance_col]]

        ax.set_title(f"{pool_name} to {structure_name}")
        sns.histplot(
            data=this_distances, x=distance_col, hue="approach", multiple=multiple, kde=False, ax=ax
        )
        ax.set_xlabel("distance [nm]")

        if save_path is not None:
            approaches = pd.unique(this_distances["approach"])
            dist_values = [
                this_distances[this_distances["approach"] == approach][distance_col].values.tolist()
                for approach in approaches
            ]
            max_len = max([len(vals) for vals in dist_values])
            save_distances = {
                approach: dists + [np.nan] * (max_len - len(dists))
                for approach, dists in zip(approaches, dist_values)
            }
            save_distances = pd.DataFrame(save_distances)

            sheet_name = f"{pool_name}_{structure_name}"
            if os.path.exists(save_path):
                with pd.ExcelWriter(save_path, engine="openpyxl", mode="a") as writer:
                    save_distances.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                save_distances.to_excel(save_path, index=False, sheet_name=sheet_name)

    _plot("MP-V", "pd_distance [nm]", "PD", axes[0, 0])
    _plot("MP-V", "boundary_distance [nm]", "AZ Membrane", axes[0, 1])
    _plot("Docked-V", "pd_distance [nm]", "PD", axes[1, 0])
    _plot("RA-V", "ribbon_distance [nm]", "Ribbon", axes[1, 1])

    fig.tight_layout()
    plt.show()


def for_tomos_with_annotation(plot_all=True):
    manual_assignments, semi_automatic_assignments, automatic_assignments = get_measurements_with_annotation()

    manual_distances = manual_assignments[
        ["pool", "ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]
    ]
    manual_distances["approach"] = ["manual"] * len(manual_distances)

    semi_automatic_distances = semi_automatic_assignments[
        ["pool", "ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]
    ]
    semi_automatic_distances["approach"] = ["semi_automatic"] * len(semi_automatic_distances)

    automatic_distances = automatic_assignments[
        ["pool", "ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]
    ]
    automatic_distances["approach"] = ["automatic"] * len(automatic_distances)

    distances = pd.concat([manual_distances, semi_automatic_distances, automatic_distances])
    if plot_all:
        distances.to_excel("./results/distances_with_manual_annotations.xlsx", index=False)
        _plot_all(distances)
    else:
        _plot_selected(distances, save_path="./results/selected_distances_manual_annotations.xlsx")


def for_all_tomos(plot_all=True):
    semi_automatic_assignments, automatic_assignments = get_all_measurements()

    semi_automatic_distances = semi_automatic_assignments[
        ["pool", "ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]
    ]
    semi_automatic_distances["approach"] = ["semi_automatic"] * len(semi_automatic_distances)

    automatic_distances = automatic_assignments[
        ["pool", "ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]
    ]
    automatic_distances["approach"] = ["automatic"] * len(automatic_distances)

    distances = pd.concat([semi_automatic_distances, automatic_distances])
    if plot_all:
        distances.to_excel("./results/distances_all_tomograms.xlsx", index=False)
        _plot_all(distances)
    else:
        _plot_selected(distances, save_path="./results/selected_distances_all_tomograms.xlsx")


def main():
    plot_all = False
    for_tomos_with_annotation(plot_all=plot_all)
    for_all_tomos(plot_all=plot_all)


if __name__ == "__main__":
    main()
