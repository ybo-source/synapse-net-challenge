import os
from glob import glob

import napari
import numpy as np
import h5py

from skimage.measure import regionprops
from skimage.morphology import remove_small_holes
from tqdm import tqdm


def fill_and_filter_vesicles(vesicles):
    ids, sizes = np.unique(vesicles, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]

    # import matplotlib.pyplot as plt
    # n, bins, patches = plt.hist(sizes, bins=32)
    # print(bins[:5])
    # plt.show()

    min_size = 2500
    vesicles_pp = vesicles.copy()
    filter_ids = ids[sizes < min_size]
    vesicles_pp[np.isin(vesicles, filter_ids)] = 0

    props = regionprops(vesicles_pp)
    for prop in props:
        bb = prop.bbox
        bb = np.s_[
            bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]
        ]
        mask = vesicles_pp[bb] == prop.label
        mask = remove_small_holes(mask, area_threshold=1000)
        vesicles_pp[bb][mask] = prop.label

    return vesicles_pp


# Filter out the vesicles so that only the ones overlapping with the max compartment are taken.
def process_tomogram(path, out_path):
    with h5py.File(out_path, "r") as f:
        if "vesicles" in f:
            return

    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        compartments = f["compartments/segment_from_3Dmodel_v2"][:]
        vesicles = f["vesicles/segment_from_combined_vesicles"][:]

    # Fill out small holes in vesicles and then apply a size filter.
    vesicles_pp = fill_and_filter_vesicles(vesicles)

    def n_vesicles(mask, ves):
        return len(np.unique(ves[mask])) - 1

    # Find the segment with most vesicles.
    props = regionprops(compartments, intensity_image=vesicles_pp, extra_properties=[n_vesicles])
    compartment_ids = [prop.label for prop in props]
    vesicle_counts = [prop.n_vesicles for prop in props]
    if len(compartment_ids) == 0:
        mask = np.ones(compartments.shape, dtype="bool")
    else:
        mask = (compartments == compartment_ids[np.argmax(vesicle_counts)]).astype("uint8")

    # Filter all vesicles that are not in the mask.
    props = regionprops(vesicles_pp, mask)
    filter_ids = [prop.label for prop in props if prop.max_intensity == 0]

    name = os.path.basename(path)
    print(name)

    no_filter = ["C_M13DKO_080212_CTRL6.7B_crop.h5", "E_M13DKO_080212_DKO1.2_crop.h5",
                 "G_M13DKO_080212_CTRL6.7B_crop.h5", "A_SNAP25_120812_CTRL2.3_14_crop.h5",
                 "A_SNAP25_12082_KO2.1_6_crop.h5", "B_SNAP25_120812_CTRL2.3_14_crop.h5",
                 "B_SNAP25_12082_CTRL2.3_5_crop.h5", "D_SNAP25_120812_CTRL2.3_14_crop.h5",
                 "G_SNAP25_12.08.12_KO1.1_3_crop.h5"]
    # Don't filter for wrong masks (visual inspection)
    if name not in no_filter:
        vesicles_pp[np.isin(vesicles_pp, filter_ids)] = 0

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(compartments, visible=False)
    v.add_labels(vesicles, visible=False)
    v.add_labels(vesicles_pp)
    v.add_labels(mask)
    v.title = name
    napari.run()

    with h5py.File(out_path, "a") as f:
        f.create_dataset("vesicles", data=vesicles_pp, compression="gzip")
        f.create_dataset("mask", data=mask, compression="gzip")


def main():
    files = sorted(glob("imig_data/**/*.h5", recursive=True))
    out_files = sorted(glob("proofread_az/**/*.h5", recursive=True))

    # for path, out_path in zip(files, out_files):
    for path, out_path in tqdm(zip(files, out_files), total=len(files)):
        process_tomogram(path, out_path)


if __name__ == "__main__":
    main()
