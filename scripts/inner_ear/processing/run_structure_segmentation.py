import os
import warnings
from functools import partial
from pathlib import Path

import imageio.v3 as imageio
from tqdm import tqdm
from elf.io import open_file

import synaptic_reconstruction.inference.postprocessing as postprocessing
from synaptic_reconstruction.file_utils import get_data_path
from synaptic_reconstruction.inference import segment_structures
from parse_table import parse_table

VERSIONS = {
    1: {
        "model": "/scratch-grete/projects/nim00007/data/synaptic_reconstruction/models/moser/structures/supervised-v4.zip",
    },
}

POSTPROCESSING = {
    "old": {
        "ribbon": partial(postprocessing.segment_ribbon, n_slices_exclude=20),
        "PD": partial(postprocessing.segment_presynaptic_density, n_slices_exclude=20),
        "membrane": None
    },
    "new": {}
}


# TODO adapt to segmentation without PD
def segment_folder(
    model_path, folder, version, is_new, postprocess, force
):
    if is_new:
        raise NotImplementedError

    output_folder = os.path.join(folder, "automatisch", f"v{version}")
    os.makedirs(output_folder, exist_ok=True)

    data_path = get_data_path(folder)

    structure_names = ["ribbon", "PD", "membrane"]

    output_paths = {
        name: os.path.join(output_folder, Path(data_path).stem + f"_{name}.tif")
        for name in structure_names
    }
    if all(os.path.exists(path) for path in output_paths.values()) and not force:
        return

    print("Segmenting structures for", data_path)
    with open_file(data_path, "r") as f:
        data = f["data"][:]

    # Suppress warnings from bioimageio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = segment_structures(
            data, model_path, structure_names,
            threshold=0.5, verbose=False,
        )

    if postprocess:
        segmentation = {}
        pp_dict = POSTPROCESSING["new" if is_new else "old"]
        for name in structure_names:
            pp = pp_dict[name]
            if pp is None:
                segmentation[name] = prediction[name]
                continue
            if name == "ribbon":
                vesicle_seg_path = os.path.join(output_folder, Path(data_path).stem + "_vesicles.tif")
                assert os.path.exists(vesicle_seg_path)
                vesicle_segmentation = imageio.imread(vesicle_seg_path)
                segmentation[name] = pp(prediction[name], vesicle_segmentation=vesicle_segmentation)
            elif name == "PD":
                segmentation[name] = pp(prediction[name], ribbon_segmentation=segmentation["ribbon"])
            else:
                segmentation[name] = pp(prediction[name])

    else:
        segmentation = prediction

    assert list(segmentation.keys()) == structure_names
    for name, seg in segmentation.items():
        imageio.imwrite(output_paths[name], seg.astype("uint8"), compression="zlib")


def run_structure_segmentation(table, version, process_new_microscope, force=False):
    model_path = VERSIONS[version]["model"]

    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        # We have to handle the segmentation without ribbon separately.
        if row["PD vorhanden? "] == "nein":
            continue

        micro = row["EM alt vs. Neu"]
        if micro == "beides":
            segment_folder(
                model_path, folder, version, is_new=False, postprocess=True, force=force
            )
            if process_new_microscope:
                folder_new = os.path.join("Tomo neues EM")
                segment_folder(
                    model_path, folder_new, version, is_new=True, postprocess=True, force=force
                )
        elif micro == "alt":
            segment_folder(
                model_path, folder, version, is_new=False, postprocess=True, force=force
            )
        elif micro == "neu" and process_new_microscope:
            segment_folder(
                model_path, folder, version, is_new=True, postprocess=True, force=force
            )


def main():
    table_path = "./Übersicht.xlsx"
    data_root = "/scratch-emmy/usr/nimcpape/data/moser"
    table = parse_table(table_path, data_root)

    version = 1
    process_new_microscope = False
    force = False

    run_structure_segmentation(table, version, process_new_microscope, force=force)


if __name__ == "__main__":
    main()
