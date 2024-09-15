import os

import imageio.v3 as imageio
from elf.io import open_file
from synaptic_reconstruction.ground_truth.shape_refinement import edge_filter, refine_vesicle_shapes
from synaptic_reconstruction.file_utils import get_data_path

from parse_table import get_data_root, parse_table
from tqdm import tqdm


def process_folder(folder):
    output_path = os.path.join(folder, "manuell", "refined_vesicles.tif")
    if os.path.exists(output_path):
        return

    data_path = get_data_path(folder)
    vesicle_path = os.path.join(folder, "manuell", "Vesikel.tif")
    if not os.path.exists(vesicle_path):
        vesicle_path = os.path.join(folder, "manuell", "vesikel.tif")
    assert os.path.exists(vesicle_path)

    with open_file(data_path, "r") as f:
        tomo = f["data"][:]

    vesicles = imageio.imread(vesicle_path)

    edges = edge_filter(tomo, sigma=3.5, method="sobel")
    refined_vesicles, seeds = refine_vesicle_shapes(vesicles, edges, fit_to_outer_boundary=False)

    imageio.imwrite(output_path, refined_vesicles)


def refine_vesicle_segmentations(data_root, table):
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue
        assert os.path.exists(folder), folder

        annotation = row["Manuelle Annotierung"].strip().lower()
        assert annotation in ("ja", "teilweise", "nein"), annotation
        have_manual = annotation in ("ja", "teilweise")

        if have_manual:
            process_folder(folder)

    extra_folder = os.path.join(
        data_root,
        "Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/modiolar/1/Tomo neues EM"
    )
    process_folder(extra_folder)


def main():
    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)
    refine_vesicle_segmentations(data_root, table)


if __name__ == "__main__":
    main()
