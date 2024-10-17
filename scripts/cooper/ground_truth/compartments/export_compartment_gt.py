import os
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import h5py
import numpy as np

from scipy.ndimage import binary_erosion, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes
from skimage.segmentation import watershed
from synaptic_reconstruction.ground_truth.shape_refinement import edge_filter
from tqdm import tqdm


def process_compartment_gt(im_path, ann_path, output_root, view=True, snap_to_bd=False):
    output_path = os.path.join(output_root, os.path.basename(im_path))
    if os.path.exists(output_path):
        return

    seg = imageio.imread(ann_path)

    with h5py.File(im_path, "r") as f:
        tomo = f["data"][:]

    if snap_to_bd:
        hmap = edge_filter(tomo, sigma=3.0, per_slice=True)
    else:
        hmap = None

    seg_pp = label(seg)
    props = regionprops(seg_pp)

    # for dilation / eroision
    structure_element = np.ones((3, 3))  # 3x3 structure for XY plane
    structure_3d = np.zeros((1, 3, 3))  # Only applied in the XY plane
    structure_3d[0] = structure_element

    # Apply the post-processing for each segment.
    min_size = 500
    for prop in props:
        # 1. size filter
        if prop.area < min_size:
            seg_pp[seg_pp == prop.label] = 0
            continue

        # 2. get the box and mask for the current object
        bb = tuple(slice(start, stop) for start, stop in zip(prop.bbox[:3], prop.bbox[3:]))
        mask = seg_pp[bb] == prop.label

        # 3. filling smal holes and closing closing
        mask = remove_small_holes(mask, area_threshold=500)
        mask = np.logical_or(binary_closing(mask, iterations=4), mask)
        mask = np.logical_or(binary_closing(mask, iterations=8, structure=structure_3d), mask)

        # 4. snap to boundary
        if snap_to_bd:
            seeds = binary_erosion(mask, structure=structure_3d, iterations=3).astype("uint8")
            bg_seeds = binary_erosion(~mask, structure=structure_3d, iterations=3)
            seeds[bg_seeds] = 2
            mask = watershed(hmap[bb], markers=seeds) == 1

        # 5. write back
        seg_pp[bb][mask] = prop.label

    if view:
        import napari

        v = napari.Viewer()
        v.add_image(tomo)
        if hmap is not None:
            v.add_image(hmap)
        v.add_labels(seg, visible=False)
        v.add_labels(seg_pp)
        napari.run()
        return

    # Cut some border pixels to avoid artifacts.
    bb = np.s_[4:-4, 16:-16, 16:-16]
    tomo = tomo[bb]
    seg_pp = seg_pp[bb]

    with h5py.File(output_path, "a") as f:
        f.create_dataset("raw", data=tomo, compression="gzip")
        f.create_dataset("labels/compartments", data=seg_pp, compression="gzip")


def main():
    for ds in ["05_stem750_sv_training", "06_hoi_wt_stem750_fm"]:
        annotation_folder = f"output/{ds}/segmentations"
        annotations = sorted(glob(os.path.join(annotation_folder, "*.tif")))

        output_root = f"output/compartment_gt/v2/{ds}"
        os.makedirs(output_root, exist_ok=True)

        image_folder = f"output/{ds}/tomograms"
        for ann_path in tqdm(annotations):
            fname = Path(ann_path).stem
            im_path = os.path.join(image_folder, f"{fname}.h5")
            assert os.path.exists(im_path)
            process_compartment_gt(im_path, ann_path, output_root, view=False)


if __name__ == "__main__":
    main()
