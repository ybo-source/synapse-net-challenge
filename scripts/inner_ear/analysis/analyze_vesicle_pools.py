import sys

import numpy as np
import pandas as pd

sys.path.append("..")
sys.path.append("../processing")

from combine_measurements import combine_manual_results, combine_automatic_results  # noqa
# from compare_pool_assignments import create_manual_assignment
from parse_table import parse_table, get_data_root  # noqa


def get_manual_assignments():
    result_path = "../results/20240917_1/fully_manual_analysis_results.xlsx"
    results = pd.read_excel(result_path)
    return results


def get_automatic_assignments(tomograms):
    result_path = "../results/20240917_1/automatic_analysis_results.xlsx"
    results = pd.read_excel(result_path)
    results = results[results["tomogram"].isin(tomograms)]
    return results


def plot_confusion_matrix(manual_assignments, automatic_assignments):
    pass


def for_tomos_with_annotation():
    manual_assignments = get_manual_assignments()
    manual_tomograms = pd.unique(manual_assignments["tomogram"])
    automatic_assignments = get_automatic_assignments(manual_tomograms)

    tomograms = pd.unique(automatic_assignments["tomogram"])
    manual_assignments = manual_assignments[manual_assignments["tomogram"].isin(tomograms)]
    assert len(pd.unique(manual_assignments["tomogram"])) == len(pd.unique(automatic_assignments["tomogram"]))

    n_tomograms = len(tomograms)
    pool_names, manual_pool_counts = np.unique(manual_assignments["pool"].values, return_counts=True)
    _, automatic_pool_counts = np.unique(automatic_assignments["pool"].values, return_counts=True)

    manual_pool_counts = manual_pool_counts.astype("float32")
    manual_pool_counts /= n_tomograms
    automatic_pool_counts = automatic_pool_counts.astype("float32")
    automatic_pool_counts /= n_tomograms

    print(pool_names)
    print(manual_pool_counts)
    print(automatic_pool_counts)

    # TODO plot as a bar chart
    # TODO save excel
    # TODO add 'more automatic' results

    breakpoint()


# TODO
def for_all_tomos():
    pass


def main():
    # data_root = get_data_root()
    # table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    # table = parse_table(table_path, data_root)
    for_tomos_with_annotation()


if __name__ == "__main__":
    main()
