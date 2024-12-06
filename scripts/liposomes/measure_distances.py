import argparse
import os
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import pandas as pd
from elf.io import open_file

from synapse_net.distance_measurements import measure_pairwise_object_distances
from synapse_net.tools.distance_measurement import measure_distances, _downsample, create_distance_lines

# ON WORKSTATION
TOMO_ROOT = "/home/user2/data/corrected_tomos_mrc"
SEG_ROOT = "/home/user2/data/results"

# ON MY LAPTOP
TOMO_ROOT = "/home/pape/Work/data/moser/lipids-julia/corrected_tomos_mrc"
SEG_ROOT = "/home/pape/Work/data/moser/lipids-julia/results"


def get_data(name, version):
    tomo_path = os.path.join(TOMO_ROOT, f"{name}.mrc_10.00Apx_corrected.mrc")
    with open_file(tomo_path, "r") as f:
        tomo = f["data"][:]

    seg_path = os.path.join(SEG_ROOT, f"v{version}", f"{name}.mrc_10.00Apx_corrected.tif")
    seg = imageio.imread(seg_path).astype("uint32")

    distance_folder = os.path.join(SEG_ROOT, f"v{version}", "distances")
    os.makedirs(distance_folder, exist_ok=True)
    distance_path = os.path.join(distance_folder, f"{name}.mrc_10.00Apx_corrected.npz")

    # NOTE: the resolution in the mrcfile is not correct so we hard-code it to the correct value
    resolution = 0.253 * 4

    return tomo, seg, distance_path, resolution


def compute_all_distances(version):
    tomos = sorted(glob(os.path.join(TOMO_ROOT, "*.mrc")))
    tomo_names = [os.path.basename(tomo)[:4] for tomo in tomos]
    for name in tomo_names:
        print("Precompute distances for tomogram", name)
        _, seg, distance_path, resolution = get_data(name, version)
        measure_pairwise_object_distances(
            seg, "boundary", n_threads=12, resolution=resolution, save_path=distance_path,
        )


def export_distance_measurements(version, export_path):
    seg_folder = os.path.join(SEG_ROOT, f"v{version}")
    seg_paths = sorted(glob(os.path.join(seg_folder, "*.tif")))

    distance_folder = os.path.join(SEG_ROOT, f"v{version}", "distances")
    distance_paths = sorted(glob(os.path.join(distance_folder, "*.npz")))
    assert len(seg_paths) == len(distance_paths), f"{len(seg_paths)}, {len(distance_paths)}"

    def measure_nn_distances(distance_path, seg_path):
        segmentation = imageio.imread(seg_path)
        segmentation = _downsample(segmentation, scale=2, is_seg=True)
        # pairs = keep_direct_distances(segmentation, distance_path, scale=2)
        pairs = None
        _, distances = create_distance_lines(
            distance_path, n_neighbors=1, scale=2, pairs=pairs, remove_duplicates=False
        )
        return pd.DataFrame(distances)

    os.makedirs(export_path, exist_ok=True)
    for seg_path, distance_path in zip(seg_paths, distance_paths):
        fname = Path(distance_path).stem
        out_path = os.path.join(export_path, f"{fname}.xlsx")
        distances = measure_nn_distances(distance_path, seg_path)
        distances.to_excel(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the tomogram, e.g. 'TS01'.")
    parser.add_argument("-p", "--precompute_all_distances", action="store_true", help="Precompute all distances.")
    parser.add_argument("-v", "--version", type=int, default=2, help="Version of segmentation results.")
    parser.add_argument("-e", "--export_path", help="Export location for all distance measurements.")
    args = parser.parse_args()

    if args.precompute_all_distances:
        compute_all_distances(args.version)
    elif args.export_path is not None:
        export_distance_measurements(args.version, args.export_path)
    else:
        tomo, seg, distance_path, resolution = get_data(args.name, args.version)
        measure_distances(tomo, seg, distance_path, resolution=resolution)


if __name__ == "__main__":
    main()
