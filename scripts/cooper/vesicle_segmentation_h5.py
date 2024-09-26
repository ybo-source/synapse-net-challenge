import argparse
import h5py
import os
from pathlib import Path

from tqdm import tqdm

from synaptic_reconstruction.inference.vesicles import segment_vesicles
from synaptic_reconstruction.inference.util import parse_tiling

def _require_output_folders(output_folder):
    seg_output = os.path.join(output_folder, "segmentations")
    os.makedirs(seg_output, exist_ok=True)
    return seg_output

def get_volume(input_path):
    with h5py.File(input_path) as seg_file:
        input_volume = seg_file["raw"][:]
    return input_volume

def run_vesicle_segmentation(input_path, output_path, model_path, tile_shape, halo, include_boundary, key_label):
    tiling = parse_tiling(tile_shape, halo)
    input = get_volume(input_path)
    segmentation = segment_vesicles(input_volume=input, model_path=model_path, verbose=False, tiling=tiling, exclude_boundary=not include_boundary)

    seg_output = _require_output_folders(output_path)
    file_name = Path(input_path).stem
    seg_path = os.path.join(seg_output, f"{file_name}.h5")

    #check
    os.makedirs(Path(seg_path).parent, exist_ok=True)

    with h5py.File(seg_path, "a") as f:
        f.create_dataset("raw", data=input, compression="gzip")
        f.create_dataset(f"vesicles/{key_label}", data=segmentation, compression="gzip")


def segment_folder(args):
    input_files = []
    for root, dirs, files in os.walk(args.input_path):
        input_files.extend([
            os.path.join(root, name) for name in files if name.endswith(".h5")
        ])
    print(input_files)
    pbar = tqdm(input_files, desc="Run segmentation")
    for input_path in pbar:
        run_vesicle_segmentation(input_path, args.output_path, args.model_path, args.tile_shape, args.halo, args.include_boundary, args.key_label)

def main():
    parser = argparse.ArgumentParser(description="Segment vesicles in EM tomograms.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True, help="The filepath to the vesicle model."
    )
    parser.add_argument(
        "--tile_shape", type=int, nargs=3,
        help="The tile shape for prediction. Lower the tile shape if GPU memory is insufficient."
    )
    parser.add_argument(
        "--halo", type=int, nargs=3,
        help="The halo for prediction. Increase the halo to minimize boundary artifacts."
    )
    parser.add_argument(
        "--include_boundary", action="store_true",
        help="Include vesicles that touch the top / bottom of the tomogram. By default these are excluded."
    )
    parser.add_argument(
        "--key_label", default = "segment_from_combined_vesicles",
        help="Give the key name for saving the segmentation in h5."
    )
    args = parser.parse_args()

    input_ = args.input_path
    if os.path.isdir(input_):
        segment_folder(args)
    else:
        run_vesicle_segmentation(input_, args.output_path, args.model_path, args.tile_shape, args.halo, args.include_boundary, args.key_label)


if __name__ == "__main__":
    main()