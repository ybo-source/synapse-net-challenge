import argparse
import os
from glob import glob

import imageio.v3 as imageio
from elf.io import open_file

from synaptic_reconstruction.distance_measurements import measure_pairwise_object_distances
from synaptic_reconstruction.tools.distance_measurement import measure_distances

TOMO_ROOT = "/home/user2/data/corrected_tomos_mrc"
SEG_ROOT = "/home/user2/data/results"


def get_data(name, version):
    tomo_path = os.path.join(TOMO_ROOT, f"{name}.mrc_10.00Apx_corrected.mrc")
    with open_file(tomo_path, "r") as f:
        tomo = f["data"][:]

    seg_path = os.path.join(SEG_ROOT, f"v{version}", f"{name}.mrc_10.00Apx_corrected.tif")
    seg = imageio.imread(seg_path).astype("uint32")

    distance_folder = os.path.join(SEG_ROOT, f"v{version}", "distances")
    os.makedirs(distance_folder, exist_ok=True)
    distance_path = os.path.join(distance_folder, f"{name}.mrc_10.00Apx_corrected.pkl")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the tomogram, e.g. 'TS01'.")
    parser.add_argument("-p", "--precompute_all_distances", action="store_true", help="Precompute all distances.")
    parser.add_argument("-v", "--version", type=int, default=2, help="Version of segmentation results.")
    args = parser.parse_args()

    if args.precompute_all_distances:
        compute_all_distances(args.version)
    else:
        tomo, seg, distance_path, resolution = get_data(args.name, args.version)
        measure_distances(tomo, seg, distance_path, resolution=resolution)


if __name__ == "__main__":
    main()
