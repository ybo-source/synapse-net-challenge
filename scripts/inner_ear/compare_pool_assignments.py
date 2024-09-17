import os
import json
from glob import glob

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def _create_manual_assignments(folder, assignment_path):
    vesicle_path, pool_path = None, None
    files = glob(os.path.join(folder, "*.tif"))

    for ff in files:
        if "manual_pools" in ff:
            pool_path = ff
        if "Vesikel" in ff and "Vesikel_pools" not in ff:
            vesicle_path = ff
    assert pool_path is not None, folder
    assert vesicle_path is not None, folder

    vesicles = imageio.imread(vesicle_path)
    pools = imageio.imread(pool_path)
    assert vesicles.shape == pools.shape

    ids, indices = np.unique(vesicles, return_index=True)
    ids, indices = ids[1:], indices[1:]
    indices = np.unravel_index(indices, vesicles.shape)

    pool_assignments = pools[indices]
    assert len(pool_assignments) == len(ids)
    assert 0 not in pool_assignments

    pool_assignments = dict(zip(ids.tolist(), pool_assignments.tolist()))

    with open(assignment_path, "w") as f:
        json.dump(pool_assignments, f)


def create_manual_assignment(root, tomograms, force):
    data_root = os.path.join(root, "Electron-Microscopy-Susi", "Analyse")

    for tomo in tqdm(tomograms, desc="Create manual pool assignments"):
        folder = os.path.join(data_root, tomo, "manuell")
        assignment_path = os.path.join(folder, "manual_pool_assignments.json")
        if os.path.exists(assignment_path) and not force:
            continue
        _create_manual_assignments(folder, assignment_path)


def compare_assignments(root, tomograms, result_table, output_path=None):
    data_root = os.path.join(root, "Electron-Microscopy-Susi", "Analyse")

    translation = {"RA-V": 1, "MP-V": 2, "Docked-V": 3, "unassigned": 4}
    manual_pools = []
    distance_pools = []

    for tomo in tomograms:

        assignment_path = os.path.join(data_root, tomo, "manuell", "manual_pool_assignments.json")
        with open(assignment_path) as f:
            manual_assignments = json.load(f)
            manual_assignments = {int(k): v for k, v in manual_assignments.items()}
            manual_assignments = dict(sorted(manual_assignments.items()))

        res_table = result_table[result_table["tomogram"] == tomo]
        distance_assignments = dict(zip(
            res_table["id"].values.tolist(),
            res_table["pool"].values.tolist(),
        ))
        distance_assignments = dict(sorted(distance_assignments.items()))
        distance_assignments = {k: translation[v] for k, v in distance_assignments.items()}

        assert len(manual_assignments) == len(distance_assignments), \
            f"{len(manual_assignments)}, {len(distance_assignments)}"
        assert all(k1 == k2 for k1, k2 in zip(manual_assignments.keys(), distance_assignments.keys()))

        manual_pools.extend(list(manual_assignments.values()))
        distance_pools.extend(list(distance_assignments.values()))

    accuracy = accuracy_score(manual_pools, distance_pools)

    plt_labels = list(translation.values())
    ConfusionMatrixDisplay.from_predictions(manual_pools, distance_pools, display_labels=plt_labels)
    if output_path is None:
        print("Accuracy:", accuracy)
        plt.show()
    else:
        plt.title(f"Assignment accuacy = {np.round(accuracy, 4)}")
        plt.savefig(output_path)


def update_measurements(root, tomograms, result_path, output_path="./fully_manual_analysis_results.xlsx"):
    data_root = os.path.join(root, "Electron-Microscopy-Susi", "Analyse")

    vesicle_table = pd.read_excel(result_path)
    translation = {1: "RA-V", 2: "MP-V", 3: "Docked-V"}

    updated_results = []
    for tomo in tqdm(tomograms, desc="Update full manual results"):

        assignment_path = os.path.join(data_root, tomo, "manuell", "manual_pool_assignments.json")
        with open(assignment_path) as f:
            manual_assignments = json.load(f)
            manual_assignments = {int(k): translation[v] for k, v in manual_assignments.items()}

        res_table = vesicle_table[vesicle_table["tomogram"] == tomo].copy()
        res_table.pool = [manual_assignments[idd] for idd in res_table.id]
        updated_results.append(res_table)

    updated_results = pd.concat(updated_results)
    updated_results.to_excel(output_path, index=False, sheet_name="vesicles")

    morpho_table = pd.read_excel(result_path, sheet_name="morphology")
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a") as writer:
        morpho_table.to_excel(writer, sheet_name="morphology", index=False)


def main():
    result_path = "manual_analysis_results.xlsx"
    result_table = pd.read_excel(result_path)
    tomograms = pd.unique(result_table["tomogram"])

    # print("Number of tomograms:", len(tomograms))
    # for tomo in tomograms:
    #     print(tomo)
    # return

    data_root = "/home/pape/Work/data/moser/em-synapses"
    create_manual_assignment(data_root, tomograms, force=False)
    compare_assignments(data_root, tomograms, result_table)
    update_measurements(data_root, tomograms, result_path)


if __name__ == "__main__":
    main()
