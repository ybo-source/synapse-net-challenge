import argparse
import os
from pathlib import Path

from synaptic_reconstruction.imod.export import get_label_names, export_segmentation


def extract_mask_from_imod(input_path, mod_file, name, output_folder, interpolation):
    label_names = get_label_names(mod_file)
    name_to_id = {v: k for k, v in label_names.items()}
    if name not in name_to_id:
        raise ValueError(
            f"Could not find the name {name} in {mod_file}. The available names are {list(name_to_id.keys())}."
        )
    object_id = name_to_id[name]

    os.makedirs(output_folder, exist_ok=True)
    fname = Path(input_path).stem

    output_path = os.path.join(output_folder, f"{fname}_mask.tif")
    export_segmentation(
        mod_file, input_path, object_id=object_id, output_path=output_path, depth_interpolation=interpolation
    )


def main():
    parser = argparse.ArgumentParser(description="Extract a mask from imod annotations.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file containing the tomogram data."
    )
    parser.add_argument(
        "--mod_file", "-f", required=True,
        help="The filepath to the mod file with the object to extract."
    )
    parser.add_argument(
        "--name", "-n", required=True, help="The name of the object to extract."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the extracted mask will be saved."
    )
    parser.add_argument(
        "--interpolation", type=int, default=10,
        help="The number of slices to interpolate over. This option can be used to"
        "close gaps between masks that only contain annotations in a subset of slices."
    )
    args = parser.parse_args()

    extract_mask_from_imod(
        args.input_path, args.mod_file, args.name, args.output_path, args.interpolation
    )


if __name__ == "__main__":
    main()
