import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import get_all_measurements, get_measurements_with_annotation


def for_tomos_with_annotation():
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
    distances.to_excel("./results/distances_with_manual_annotations.xlsx", index=False)

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


def for_all_tomos():
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
    distances.to_excel("./results/distances_all_tomograms.xlsx", index=False)

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


def main():
    for_tomos_with_annotation()
    for_all_tomos()


if __name__ == "__main__":
    main()
