import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import get_all_measurements, get_measurements_with_annotation


def plot_pools(data, errors):
    data_for_plot = pd.melt(data, id_vars="Pool", var_name="Method", value_name="Measurement")

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data_for_plot, x="Pool", y="Measurement", hue="Method")

    # FIXME
    # error_for_plot = pd.melt(errors, id_vars="Pool", var_name="Method", value_name="Error")
    # # Add error bars manually
    # for i, bar in enumerate(plt.gca().patches):
    #     # Get Standard Deviation for the current bar
    #     err = error_for_plot.iloc[i % len(error_for_plot)]["Error"]
    #     bar_x = bar.get_x() + bar.get_width() / 2
    #     bar_y = bar.get_height()
    #     plt.errorbar(bar_x, bar_y, yerr=err, fmt="none", c="black", capsize=4)

    # Customize the chart
    plt.title("Different measurements for vesicles per pool")
    plt.xlabel("Vesicle Pools")
    plt.ylabel("Vesicles per Tomogram")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Approaches")

    # Show the plot
    plt.tight_layout()
    plt.show()


# TODO use the actual results without vesicle post-processing.
def for_tomos_with_annotation():
    manual_assignments, automatic_assignments = get_measurements_with_annotation()

    manual_counts = manual_assignments.groupby(["tomogram", "pool"]).size().unstack(fill_value=0)
    automatic_counts = automatic_assignments.groupby(["tomogram", "pool"]).size().unstack(fill_value=0)

    manual_stats = manual_counts.agg(["mean", "std"]).transpose().reset_index()
    automatic_stats = automatic_counts.agg(["mean", "std"]).transpose().reset_index()

    data = pd.DataFrame({
        "Pool": manual_stats["pool"],
        "Manual": manual_stats["mean"],
        "Semi-automatic": automatic_stats["mean"],
        "Automatic": automatic_stats["mean"],
    })
    errors = pd.DataFrame({
        "Pool": manual_stats["pool"],
        "Manual": manual_stats["std"],
        "Semi-automatic": automatic_stats["std"],
        "Automatic": automatic_stats["std"],
    })

    plot_pools(data, errors)

    output_path = "./vesicle_pools_small.xlsx"
    data.to_excel(output_path, index=False, sheet_name="Average")
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a") as writer:
        errors.to_excel(writer, sheet_name="StandardDeviation", index=False)


# TODO use the actual results without vesicle post-processing.
def for_all_tomos():

    automatic_assignments = get_all_measurements()
    # TODO double check why this number is so different! (64 vs. 81)
    # tomos = pd.unique(automatic_assignments["tomogram"])
    # print(len(tomos), n_tomos)
    # assert len(tomos) == n_tomos

    automatic_counts = automatic_assignments.groupby(["tomogram", "pool"]).size().unstack(fill_value=0)
    automatic_stats = automatic_counts.agg(["mean", "std"]).transpose().reset_index()

    data = pd.DataFrame({
        "Pool": automatic_stats["pool"],
        "Semi-automatic": automatic_stats["mean"],
        "Automatic": automatic_stats["mean"],
    })
    errors = pd.DataFrame({
        "Pool": automatic_stats["pool"],
        "Semi-automatic": automatic_stats["std"],
        "Automatic": automatic_stats["std"],
    })

    plot_pools(data, errors)

    output_path = "./vesicle_pools_large.xlsx"
    data.to_excel(output_path, index=False, sheet_name="Average")
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a") as writer:
        errors.to_excel(writer, sheet_name="StandardDeviation", index=False)


def main():
    # for_tomos_with_annotation()
    for_all_tomos()


if __name__ == "__main__":
    main()
