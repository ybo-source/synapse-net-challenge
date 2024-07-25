import os
from pathlib import Path

import mrcfile
import pandas
from tqdm import tqdm
from elf.io import open_file

from synaptic_reconstruction.morphology import compute_radii
from synaptic_reconstruction.file_utils import get_data_path
from run_analyis import _load_segmentation, compute_morphology, to_excel
from parse_table import parse_table, get_data_root, _match_correction_folder, _match_correction_file


def update_excel(result_path, vesicle_ids, vesicle_radii, morphology_measurements):
    vesicle_table = pandas.read_excel(result_path)

    pool_assignments = dict(zip(vesicle_table["id"].values, vesicle_table["pool"].values))
    ves_assignments = pandas.DataFrame.from_dict(
        {
            "id": vesicle_ids,
            "pool": [pool_assignments[vid] for vid in vesicle_ids],
            "radius [nm]": [vesicle_radii[vid] for vid in vesicle_ids],
        }
    )

    distances = {
        "ribbon": pandas.DataFrame({
            "id": vesicle_table["id"].values,
            "distance": vesicle_table["ribbon_distance [nm]"].values,
            "ribbon_id": vesicle_table["ribbon_id"].values,
        }),
        "PD": pandas.DataFrame({
            "id": vesicle_table["id"].values,
            "distance": vesicle_table["pd_distance [nm]"].values,
            "pd_id": vesicle_table["pd_id"].values,
        }),
        "membrane": pandas.DataFrame({
            "id": vesicle_table["id"].values,
            "distance": vesicle_table["boundary_distance [nm]"].values,
        })
    }

    to_excel(
        ves_assignments, distances["ribbon"], distances["PD"], distances["membrane"],
        morphology_measurements=morphology_measurements,
        result_path=result_path,
    )


def compute_tomo_morphology(segmentation_paths, resolution, result_path, tomo_shape, keep_unassigned=False):
    vesicles = _load_segmentation(segmentation_paths["vesicles"], tomo_shape)
    ribbon = _load_segmentation(segmentation_paths["ribbon"], tomo_shape)
    pd = _load_segmentation(segmentation_paths["PD"], tomo_shape)

    morphology_measurements = compute_morphology(ribbon, pd, resolution=resolution)
    vesicle_ids = pandas.read_excel(result_path)["id"].values
    vesicle_ids, vesicle_radii = compute_radii(vesicles, resolution, ids=vesicle_ids)

    update_excel(result_path, vesicle_ids, vesicle_radii, morphology_measurements)


def _get_seg_paths_automatic(folder, data_path, version, force):
    output_folder = os.path.join(folder, "automatisch", f"v{version}")

    fname = Path(data_path).stem

    segmentation_names = ["vesicles", "ribbon", "PD", "membrane"]
    segmentation_paths = {name: os.path.join(output_folder, f"{fname}_{name}.h5") for name in segmentation_names}

    if not all(os.path.exists(path) for path in segmentation_paths.values()):
        print("Not all required segmentations were found")
        return

    correction_folder = _match_correction_folder(folder)
    if os.path.exists(correction_folder):
        output_folder = correction_folder
        result_path = os.path.join(output_folder, "measurements.xlsx")
        if os.path.exists(result_path) and not force:
            return

        for seg_name in segmentation_names:
            seg_path = _match_correction_file(correction_folder, seg_name)
            if os.path.exists(seg_path):
                segmentation_paths[seg_name] = seg_path

    segmentation_paths["output_folder"] = output_folder
    return segmentation_paths


def _get_seg_paths_manual(folder, data_path):
    output_folder = os.path.join(folder, "manuell")

    segmentation_names = {"vesicles": "Vesikel", "ribbon": "Ribbon", "PD": "PD", "membrane": "Membrane"}
    segmentation_paths = {name: os.path.join(output_folder, f"{nname}.tif")
                          for name, nname in segmentation_names.items()}

    missing_segmentations = [name for name, path in segmentation_paths.items() if not os.path.exists(path)]
    if missing_segmentations:
        return

    segmentation_paths["output_folder"] = output_folder
    return segmentation_paths


def compute_morphology_folder(folder, version, force, analyze_manual_annotations):
    data_path = get_data_path(folder)

    if analyze_manual_annotations:
        segmentation_paths = _get_seg_paths_manual(folder, data_path)
    else:
        segmentation_paths = _get_seg_paths_automatic(folder, data_path, version, force)
    if segmentation_paths is None:
        return
    output_folder = segmentation_paths.pop("output_folder")

    # Get the resolution (in Angstrom) and convert it to nanometer
    with mrcfile.open(data_path, "r") as f:
        resolution = f.voxel_size.tolist()
    resolution = [res / 10 for res in resolution]

    # Get the tomogram shape.
    with open_file(data_path, "r") as f:
        tomo_shape = f["data"].shape

    result_path = os.path.join(output_folder, "measurements.xlsx")
    if force or not os.path.exists(result_path):
        compute_tomo_morphology(segmentation_paths, resolution, result_path, tomo_shape)


def run_morphology_computation(table, version, force, analyze_manual_annotations):
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        # We have to handle the segmentation without ribbon separately.
        if row["PD vorhanden? "] == "nein":
            continue

        n_pds = row["Anzahl PDs"]
        if n_pds == "unklar":
            n_pds = 1
            # continue
        n_pds = int(n_pds)
        n_ribbons = int(row["Anzahl Ribbons"])
        if (n_ribbons == 2 and n_pds == 1):
            print(f"The tomogram {folder} has {n_ribbons} ribbons and {n_pds} PDs.")
            print("The structure post-processing for this case is not yet implemented and will be skipped.")
            continue

        compute_morphology_folder(
            folder, version, force=force, analyze_manual_annotations=analyze_manual_annotations,
        )
        micro = row["EM alt vs. Neu"]
        if micro == "beides":

            folder_new = os.path.join(folder, "Tomo neues EM")
            if not os.path.exists(folder_new):
                folder_new = os.path.join(folder, "neues EM")
            assert os.path.exists(folder_new), folder_new
            compute_morphology_folder(
                folder, version, force=force, analyze_manual_annotations=analyze_manual_annotations,
            )


# TODO also support updates for manual tomograms
def main():
    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    version = 2
    force = True
    analyze_manual_annotations = True

    run_morphology_computation(table, version, force=force, analyze_manual_annotations=analyze_manual_annotations)


if __name__ == "__main__":
    main()
