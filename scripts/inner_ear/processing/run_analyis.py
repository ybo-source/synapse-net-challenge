import os
from pathlib import Path

import mrcfile
import numpy as np
import pandas

from synaptic_reconstruction.file_utils import get_data_path
from synaptic_reconstruction.distance_measurements import (
    measure_segmentation_to_object_distances,
    filter_blocked_segmentation_to_object_distances,
)
from synaptic_reconstruction.morphology import compute_radii, compute_object_morphology
from synaptic_reconstruction.inference.postprocessing import filter_border_vesicles

from elf.io import open_file
from tqdm import tqdm
from parse_table import parse_table


def compute_distances(segmentation_paths, save_folder, resolution, force):
    os.makedirs(save_folder, exist_ok=True)

    vesicle_path = segmentation_paths["vesicles"]
    vesicles = None

    def _require_vesicles():
        if vesicles is None:
            with open_file(vesicle_path, "r") as f:
                return f["segmentation"][:]
        else:
            return vesicles

    # Compute the distance of the vesicles to the ribbon.
    ribbon_save = os.path.join(save_folder, "ribbon.npz")
    if force or not os.path.exists(ribbon_save):
        vesicles = _require_vesicles()
        with open_file(segmentation_paths["ribbon"], "r") as f:
            ribbon = f["segmentation"][:]
        if ribbon.sum() == 0:
            print("The ribbon segmentation at", segmentation_paths["ribbon"], "is empty. Skipping analysis.")
            return None, True
        measure_segmentation_to_object_distances(vesicles, ribbon, save_path=ribbon_save, resolution=resolution)

    # Compute the distance of the vesicles to the PD.
    pd_save = os.path.join(save_folder, "PD.npz")
    if force or not os.path.exists(pd_save):
        vesicles = _require_vesicles()
        with open_file(segmentation_paths["PD"], "r") as f:
            pd = f["segmentation"][:]
        if pd.sum() == 0:
            print("The PD segmentation at", segmentation_paths["PD"], "is empty. Skipping analysis.")
            return None, True
        measure_segmentation_to_object_distances(vesicles, pd, save_path=pd_save, resolution=resolution)

    # Compute the distance of the vesicle to the membrane.
    membrane_save = os.path.join(save_folder, "membrane.npz")
    if force or not os.path.exists(membrane_save):
        vesicles = _require_vesicles()
        with open_file(segmentation_paths["membrane"], "r") as f:
            membrane = f["segmentation"][:]
        measure_segmentation_to_object_distances(
            vesicles, membrane, save_path=membrane_save, resolution=resolution
        )

    distance_paths = {"ribbon": ribbon_save, "PD": pd_save, "membrane": membrane_save}
    return distance_paths, False


def assign_vesicles_to_pools(vesicles, distance_paths):

    def load_dist(measurement_path, seg_ids=None):
        auto_dists = np.load(measurement_path)
        distances, this_seg_ids = auto_dists["distances"], auto_dists["seg_ids"]
        object_ids = auto_dists.get("object_ids", np.zeros_like(seg_ids))
        if seg_ids is not None:
            assert np.array_equal(seg_ids, this_seg_ids)
        return distances, this_seg_ids, object_ids

    ribbon_distances, seg_ids, ribbon_ids = load_dist(distance_paths["ribbon"])
    pd_distances, _, pd_ids = load_dist(distance_paths["PD"], seg_ids=seg_ids)
    bd_distances, _, _ = load_dist(distance_paths["membrane"], seg_ids=seg_ids)
    assert len(seg_ids) == len(ribbon_distances) == len(pd_distances) == len(bd_distances)

    # Find the vesicles that are ribbon associated (RA-V).
    # Criterion: vesicles are closer than 80 nm to ribbon and they are in the first row
    # (i.e. not blocked by another vesicle).
    rav_ribbon_distance = 80  # nm
    rav_ids = seg_ids[ribbon_distances < rav_ribbon_distance]
    # Filter out the blocked vesicles.
    rav_ids = filter_blocked_segmentation_to_object_distances(
        vesicles, distance_paths["ribbon"], seg_ids=rav_ids, line_dilation=4, verbose=True,
    )
    rav_ids = filter_border_vesicles(vesicles, seg_ids=rav_ids)
    # n_blocked = len(rav_ids_all) - len(rav_ids)
    # print(n_blocked, "ribbon associated vesicles were blocked by the ribbon.")

    # Find the vesicles that are membrane proximal (MP-V).
    # Criterion: vesicles are closer than 50 nm to the membrane and closer than 100 nm to the PD.
    mpv_pd_distance = 100  # nm
    mpv_bd_distance = 50  # nm
    mpv_ids = seg_ids[np.logical_and(pd_distances < mpv_pd_distance, bd_distances < mpv_bd_distance)]
    mpv_ids = filter_border_vesicles(vesicles, seg_ids=mpv_ids)

    # Find the vesicles that are membrane docked (Docked-V).
    # Criterion: vesicles are closer than 2 nm to the membrane and closer than 100 nm to the PD.
    docked_pd_distance = 100  # nm
    docked_bd_distance = 2  # nm
    docked_ids = seg_ids[np.logical_and(pd_distances < docked_pd_distance, bd_distances < docked_bd_distance)]
    docked_ids = filter_border_vesicles(vesicles, seg_ids=docked_ids)

    # Keep only the vesicle ids that are in one of the three categories.
    vesicle_ids = np.unique(np.concatenate([rav_ids, mpv_ids, docked_ids]))

    # Create a dictionary to map vesicle ids to their corresponding pool.
    # (RA-V get's over-written by MP-V, which is correct).
    pool_assignments = {vid: "RA-V" for vid in rav_ids}
    pool_assignments.update({vid: "MP-V" for vid in mpv_ids})
    pool_assignments.update({vid: "Docked-V" for vid in docked_ids})

    id_mask = np.isin(seg_ids, vesicle_ids)
    assert id_mask.sum() == len(vesicle_ids)
    distances = {
        "ribbon": pandas.DataFrame({
            "id": vesicle_ids,
            "distance": ribbon_distances[id_mask],
            "ribbon_id": ribbon_ids[id_mask],
        }),
        "PD": pandas.DataFrame({
            "id": vesicle_ids,
            "distance": pd_distances[id_mask],
            "pd_id": pd_ids[id_mask],
        }),
        "membrane": pandas.DataFrame({
            "id": vesicle_ids,
            "distance": bd_distances[id_mask],
        })
    }

    return vesicle_ids, pool_assignments, distances


def to_excel(
    ves_assignments,
    ribbon_distances,
    pd_distances,
    boundary_distances,
    morphology_measurements,
    result_path
):
    # Merge all vesicle features into one table.
    table = ves_assignments.merge(ribbon_distances, on="id")
    table = table.rename(columns={"distance": "ribbon_distance [nm]"})

    table = table.merge(pd_distances, on="id")
    table = table.rename(columns={"distance": "pd_distance [nm]"})

    table = table.merge(boundary_distances, on="id")
    table = table.rename(columns={"distance": "boundary_distance [nm]"})

    table = table.sort_values(by="pool")

    table.to_excel(result_path, sheet_name="vesicles", index=False)
    with pandas.ExcelWriter(result_path, engine="openpyxl", mode="a") as writer:
        morphology_measurements.to_excel(writer, sheet_name="morphology", index=False)


def compute_morphology(ribbon, pd, resolution):
    ribbon_morph = compute_object_morphology(ribbon, "ribbon", resolution=resolution)
    pd_morph = compute_object_morphology(pd, "presynaptic-density", resolution=resolution)
    measurements = pandas.concat([ribbon_morph, pd_morph])
    return measurements


def analyze_distances(segmentation_paths, distance_paths, resolution, result_path):
    with open_file(segmentation_paths["vesicles"], "r") as f:
        vesicles = f["segmentation"][:]
    with open_file(segmentation_paths["ribbon"], "r") as f:
        ribbon = f["segmentation"][:]
    with open_file(segmentation_paths["PD"], "r") as f:
        pd = f["segmentation"][:]

    vesicle_ids, pool_assignments, distances = assign_vesicles_to_pools(vesicles, distance_paths)
    vesicle_radii = compute_radii(vesicles, resolution, ids=vesicle_ids)
    morphology_measurements = compute_morphology(ribbon, pd, resolution)

    ves_assignments = pandas.DataFrame.from_dict(
        {
            "id": list(pool_assignments.keys()),
            "pool": list(pool_assignments.values()),
            "radius [nm]": vesicle_radii,
        }
    )

    to_excel(
        ves_assignments, distances["ribbon"], distances["PD"], distances["membrane"],
        morphology_measurements=morphology_measurements,
        result_path=result_path,
    )


# TODO adapt to segmentation without PD
def analyze_folder(folder, version, n_ribbons, force):
    data_path = get_data_path(folder)
    output_folder = os.path.join(folder, "automatisch", f"v{version}")

    fname = Path(data_path).stem

    segmentation_names = ["vesicles", "ribbon", "PD", "membrane"]
    segmentation_paths = {name: os.path.join(output_folder, f"{fname}_{name}.h5") for name in segmentation_names}

    if not all(os.path.exists(path) for path in segmentation_paths.values()):
        print("Not all required segmentations were found")
        return

    # Get the resolution (in Angstrom) and convert it to nanometer
    with mrcfile.open(data_path, "r") as f:
        resolution = f.voxel_size.tolist()
    resolution = [res / 10 for res in resolution]

    out_distance_folder = os.path.join(output_folder, "distances")
    distance_paths, skip = compute_distances(segmentation_paths, out_distance_folder, resolution, force)
    if skip:
        return

    result_path = os.path.join(output_folder, "measurements.xlsx")
    if force or not os.path.exists(result_path):
        analyze_distances(segmentation_paths, distance_paths, resolution, result_path)


def run_analysis(table, version, force=False):
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        # We have to handle the segmentation without ribbon separately.
        if row["PD vorhanden? "] == "nein":
            continue

        n_pds = row["Anzahl PDs"]
        if n_pds == "unklar":
            continue
        n_pds = int(n_pds)
        n_ribbons = int(row["Anzahl Ribbons"])
        if (n_ribbons == 2 and n_pds == 1):
            print(f"The tomogram {folder} has {n_ribbons} ribbons and {n_pds} PDs.")
            print("The structure post-processing for this case is not yet implemented and will be skipped.")
            continue

        micro = row["EM alt vs. Neu"]
        if micro == "beides":
            analyze_folder(folder, version, n_ribbons, force=force)

            folder_new = os.path.join(folder, "Tomo neues EM")
            if not os.path.exists(folder_new):
                folder_new = os.path.join(folder, "neues EM")
            assert os.path.exists(folder_new), folder_new
            analyze_folder(folder_new, version, n_ribbons, force=force)

        elif micro == "alt":
            analyze_folder(folder, version, n_ribbons, force=force)

        elif micro == "neu":
            analyze_folder(folder, version, n_ribbons, force=force)


def main():
    # data_root = "/scratch-emmy/usr/nimcpape/data/moser"
    # data_root = "/home/pape/Work/data/moser/em-synapses"
    data_root = "/home/sophia/data"
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    version = 2
    force = False

    run_analysis(table, version, force=force)


if __name__ == "__main__":
    main()
