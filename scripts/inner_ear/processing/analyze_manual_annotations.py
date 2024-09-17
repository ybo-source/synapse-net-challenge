import os

import mrcfile
from elf.io import open_file
from synaptic_reconstruction.file_utils import get_data_path
from tqdm import tqdm

from parse_table import parse_table, get_data_root
from run_analyis import compute_distances, analyze_distances


def analyze_folder(folder, n_ribbons, force, use_refined_vesicles):
    data_path = get_data_path(folder)
    output_folder = os.path.join(folder, "manuell")

    ves_name = "refined_vesicles" if use_refined_vesicles else "Vesikel"
    segmentation_names = {"vesicles": ves_name, "ribbon": "Ribbon", "PD": "PD", "membrane": "Membrane"}
    segmentation_paths = {name: os.path.join(output_folder, f"{nname}.tif")
                          for name, nname in segmentation_names.items()}

    missing_segmentations = [name for name, path in segmentation_paths.items() if not os.path.exists(path)]
    if missing_segmentations:
        print(f"Not all required segmentations were found in {folder}, missing {missing_segmentations}")
        return

    result_path = os.path.join(output_folder, "measurements.xlsx")
    if os.path.exists(result_path) and not force:
        return

    # Get the resolution (in Angstrom) and convert it to nanometer
    with mrcfile.open(data_path, "r") as f:
        resolution = f.voxel_size.tolist()
    resolution = [res / 10 for res in resolution]

    # Get the tomogram shape.
    with open_file(data_path, "r") as f:
        tomo_shape = f["data"].shape

    out_distance_folder = os.path.join(output_folder, "distances")
    distance_paths, skip = compute_distances(segmentation_paths, out_distance_folder, resolution, force, tomo_shape)
    if skip:
        return

    if force or not os.path.exists(result_path):
        analyze_distances(
            segmentation_paths, distance_paths, resolution, result_path, tomo_shape, keep_unassigned=True,
            apply_extra_filters=False
        )


def run_analysis(table, force=False, use_refined_vesicles=True):
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
            analyze_folder(folder, n_ribbons, force=force, use_refined_vesicles=use_refined_vesicles)

            folder_new = os.path.join(folder, "Tomo neues EM")
            if not os.path.exists(folder_new):
                folder_new = os.path.join(folder, "neues EM")
            assert os.path.exists(folder_new), folder_new
            analyze_folder(folder_new, n_ribbons, force=force, use_refined_vesicles=use_refined_vesicles)

        elif micro == "alt":
            analyze_folder(folder, n_ribbons, force=force, use_refined_vesicles=use_refined_vesicles)

        elif micro == "neu":
            analyze_folder(folder, n_ribbons, force=force, use_refined_vesicles=use_refined_vesicles)


def main():
    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    force = True
    run_analysis(table, force=force)


if __name__ == "__main__":
    main()
