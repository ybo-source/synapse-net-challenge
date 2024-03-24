import os
from functools import partial
from pathlib import Path

import synaptic_reconstruction.inference.postprocessing as postprocessing
from synaptic_reconstruction.file_utils import get_data_path

from elf.io import open_file
from tqdm import tqdm
from parse_table import parse_table


POSTPROCESSING = {
    "old": {
        "ribbon": partial(postprocessing.segment_ribbon, n_slices_exclude=20),
        "PD": partial(postprocessing.segment_presynaptic_density, n_slices_exclude=20),
        "membrane": partial(postprocessing.segment_membrane_next_to_object, n_slices_exclude=20),
    },
    "new": {
        "ribbon": partial(postprocessing.segment_ribbon, n_slices_exclude=20, max_vesicle_distance=30),
        "PD": partial(postprocessing.segment_presynaptic_density, n_slices_exclude=20, max_distance_to_ribbon=22.5),
        "membrane": partial(postprocessing.segment_membrane_next_to_object, n_slices_exclude=20, radius=37.5),
    }
}


# TODO adapt to segmentation without PD
def postprocess_folder(folder, version, n_ribbons, is_new, force):
    output_folder = os.path.join(folder, "automatisch", f"v{version}")
    data_path = get_data_path(folder)

    structure_names = ["ribbon", "PD", "membrane"]
    pp_dict = POSTPROCESSING["new" if is_new else "old"]

    segmentations = {}

    for name in structure_names:
        pp = pp_dict[name]

        segmentation_path = os.path.join(output_folder, Path(data_path).stem + f"_{name}.h5")

        with open_file(segmentation_path, "r") as f:
            if "segmentation" in f and not force:
                continue

            assert "prediction" in f
            prediction = f["prediction"][:]

        if name == "ribbon":
            vesicle_seg_path = os.path.join(output_folder, Path(data_path).stem + "_vesicles.h5")
            assert os.path.exists(vesicle_seg_path)
            with open_file(vesicle_seg_path, "r") as f:
                vesicle_segmentation = f["segmentation"][:]
            segmentations[name] = pp(prediction, vesicle_segmentation=vesicle_segmentation, n_ribbons=n_ribbons)

        elif name == "PD":
            segmentations[name] = pp(prediction, ribbon_segmentation=segmentations["ribbon"])

        elif name == "membrane":
            ribbon_and_pd = segmentations["ribbon"] + segmentations["PD"]
            segmentations[name] = pp(prediction, object_segmentation=ribbon_and_pd, n_fragments=n_ribbons)

        seg = segmentations[name]
        with open_file(segmentation_path, "a") as f:
            ds = f.require_dataset(name="segmentation", shape=seg.shape, dtype=seg.dtype, compression="gzip")
            ds[:] = seg


def run_structure_postprocessing(table, version, process_new_microscope, force=False):
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        # We have to handle the segmentation without ribbon separately.
        if row["PD vorhanden? "] == "nein":
            continue

        n_ribbons = row["Anzahl Ribbons"]
        assert isinstance(n_ribbons, int)

        micro = row["EM alt vs. Neu"]
        if micro == "beides":
            postprocess_folder(folder, version, n_ribbons, is_new=False, force=force)
            if process_new_microscope:
                folder_new = os.path.join(folder, "Tomo neues EM")
                if not os.path.exists(folder_new):
                    folder_new = os.path.join(folder, "neues EM")
                assert os.path.exists(folder_new), folder_new
                postprocess_folder(folder_new, version, n_ribbons, is_new=True, force=force)

        elif micro == "alt":
            postprocess_folder(folder, version, n_ribbons, is_new=False, force=force)

        elif micro == "neu" and process_new_microscope:
            postprocess_folder(folder, version, n_ribbons, is_new=True, force=force)


def main():
    table_path = "./Übersicht.xlsx"
    data_root = "/scratch-emmy/usr/nimcpape/data/moser"
    # data_root = "/home/pape/Work/data/moser/em-synapses"
    table = parse_table(table_path, data_root)

    version = 1
    process_new_microscope = True
    force = False

    run_structure_postprocessing(table, version, process_new_microscope, force=force)


if __name__ == "__main__":
    main()
