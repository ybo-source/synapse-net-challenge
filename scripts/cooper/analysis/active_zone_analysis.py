import os
from glob import glob

import h5py
import numpy as np

from scipy.ndimage import binary_closing
from skimage.measure import label
from synaptic_reconstruction.ground_truth.shape_refinement import edge_filter
from tqdm import tqdm

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/20241102_TOMO_DATA_Imig2014/final_Imig2014_seg_manComp"  # noqa

OUTPUT_AZ = "./boundary_az"


def filter_az(path):
    # Check if we have the output already.
    ds, fname = os.path.split(path)
    ds = os.path.basename(ds)
    out_path = os.path.join(OUTPUT_AZ, ds, fname)
    os.makedirs(os.path.join(OUTPUT_AZ, ds), exist_ok=True)
    if os.path.exists(out_path):
        return

    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        az = f["AZ/segment_from_AZmodel_v3"][:]
        vesicles = f["/vesicles/segment_from_combined_vesicles"][:]

    # Compute the sato filter of the raw data, smooth it afterwards.
    # This will highlight dark ridge-like structures, and so
    # will yield high values for the plasma membrane.
    hmap = edge_filter(raw, sigma=1.0, method="sato", per_slice=True, n_threads=8)

    # Filter the active zone by combining a bunch of things:
    # 1. Find a mask with high values in the ridge filter.
    threshold_hmap = 0.5
    az_filtered = hmap > threshold_hmap
    # 2. Intersect it with the active zone predictions.
    az_filtered = np.logical_and(az_filtered, az)
    # 3. Intersect it with the negative vesicle mask.
    az_filtered = np.logical_and(az_filtered, vesicles == 0)

    # Postprocessing of the filtered active zone:
    # 1. Apply connected components and only keep the largest component.
    az_filtered = label(az_filtered)
    ids, sizes = np.unique(az_filtered, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]
    az_filtered = (az_filtered == ids[np.argmax(sizes)]).astype("uint8")
    # 2. Apply binary closing.
    az_filtered = np.logical_or(az_filtered, binary_closing(az_filtered, iterations=4)).astype("uint8")

    # Save the result.
    with h5py.File(out_path, "a") as f:
        f.create_dataset("filtered_az", data=az_filtered, compression="gzip")


def main():
    files = sorted(glob(os.path.join(ROOT, "**/*.h5"), recursive=True))
    for ff in tqdm(files):
        filter_az(ff)


if __name__ == "__main__":
    main()
