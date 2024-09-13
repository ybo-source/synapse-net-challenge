import argparse
import os

import h5py
import numpy as np

from elf.evaluation.matching import label_overlap, intersection_over_union
from skimage.segmentation import relabel_sequential
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from tqdm import tqdm

from skimage.morphology import ball

from synaptic_reconstruction.inference.vesicles import segment_vesicles
from synaptic_reconstruction.inference.util import _get_file_paths

from vigra.analysis import watershedsNew
from skimage.segmentation import watershed
from elf.segmentation import watershed
from elf.parallel import seeded_watershed
from skimage import filters

MODEL_PATH = "/scratch-grete/projects/nim00007/data/synaptic_reconstruction/models/cooper/vesicles/3D-UNet-for-Vesicle-Segmentation-vesicles-010508model_v1r45_0105mr45_0105mr45.zip"  # noqa


def extract_gt_bounding_box(raw, vesicle_gt, coordinates, halo=[2, 32, 32]):
    bb = np.where(vesicle_gt > 0)
    bb = tuple(slice(
        max(int(b.min() - ha), 0),
        min(int(b.max()) + ha, sh)
    ) for b, sh, ha in zip(bb, raw.shape, halo))
    raw, vesicle_gt = raw[bb], vesicle_gt[bb]

    #TODO check if offset is needed
    offset = np.array([b.start for b in bb])[None]
    coordinates -= offset

    seed_ids = np.arange(1, len(coordinates) + 1)
    coordinates = tuple(coordinates[:, i] for i in range(coordinates.shape[1]))

    return raw, vesicle_gt, coordinates, seed_ids


# Postprocess the vesicle shape (if still necessary after fixing the IMOD extraction).
def postprocess_vesicle_shape(vesicle_gt, prediction, coordinates, radii, seed_ids, raw):
    _, boundaries = prediction.squeeze()
    boundary_threshold = 0.05
    distance_map = distance_transform_edt(boundaries < boundary_threshold)
    distance_map = 1. - (distance_map - distance_map.min()) / distance_map.max()
    distance_map += boundaries

    #TODO edge filter??
    sobel_edges = filters.sobel(raw)
    distance_map += sobel_edges

    shape = distance_map.shape
    labels_pp = np.zeros_like(distance_map, dtype="uint32")
    coordinates = [
        (coordinates[0][i], coordinates[1][i], coordinates[2][i])
        for i in range(len(coordinates[0]))
    ]

    assert len(seed_ids) == len(coordinates) == len(radii)
    for label_id, coord, radius in zip(seed_ids, coordinates, radii):
        radius = int(radius)
        roi = tuple(
            slice(max(co - radius, 0), min(co + radius, sh)) for co, sh in zip(coord, shape)
        )

        #handeling the case where part of the sphere mask would go out of bounds
        radius_clipped_left = [co - max(co - radius, 0) for co in coord]
        radius_clipped_right = [min(co + radius, sh) - co for co, sh in zip(coord, shape)]
        mask_slice = tuple(
            slice(radius + 1 - rl, radius + 1 + rr) for rl, rr in zip(radius_clipped_left, radius_clipped_right)
        )

        mask = ball(radius)[mask_slice]

        dist = distance_map[roi]
        assert mask.shape == dist.shape

        #TODO? change dilation as well?
        bg_mask_dilation=2
        bg_mask = ~binary_dilation(mask, iterations=bg_mask_dilation)

        #TODO decide on seed erosion
        seed_erosion = 5
        fg_mask = binary_erosion(mask, iterations=seed_erosion)
        centroid = tuple(co - b.start for co, b in zip(coord, roi))
        fg_mask[centroid] = 1

        seeds = np.zeros_like(mask, dtype="uint32")
        seeds[fg_mask] = 1
        seeds[bg_mask] = 2
        #TODO choose between watershedsNews and from skimage.segmentation import watershed
        #pp = watershedsNew(dist.astype("float32"), seeds=seeds)[0]
        ##pp = watershed(-dist, markers=seeds, mask=mask)
        halo = (4, 32, 32)
        labels_pp = seeded_watershed(dist, seeds, labels_pp, mask.shape, halo)
        
        #labels_pp[roi][pp == 1] = label_id

    return labels_pp


def find_additional_vesicles(vesicle_gt, segmentation, matching_threshold=0.5):
    segmentation = relabel_sequential(segmentation)[0]

    # Match the vesicles in the segmentation to the ground-truth.
    overlap, _ = label_overlap(segmentation, vesicle_gt)
    overlap = intersection_over_union(overlap)

    # Get the segmentation IDs.
    seg_ids = np.unique(segmentation)

    # Filter out IDs with a larger overlap than the matching threshold:
    # These likely correspond to a vesicle covered by the ground-truth.
    filter_ids = []
    for seg_id in seg_ids[1:]:
        max_overlap = overlap[seg_id, :].max()
        if max_overlap > matching_threshold:
            filter_ids.append(seg_id)

    # Get the additional vesicles by removing filtered vesicles.
    additional_vesicles = segmentation.copy()
    additional_vesicles[np.isin(segmentation, filter_ids)] = 0
    additional_vesicles = relabel_sequential(additional_vesicles)[0]

    return additional_vesicles


def postprocess_vesicle_gt(raw, vesicle_gt, coordinates, radii):
    """Run post-processing for the vesicle ground-truth extracted from IMOD.
    This includes the following steps:
    - Extract the bounding box around the annotated vesicles in the ground-truth.
    - Run prediction and segmentation for the raw data.
    - Refine the vesicle shape (if still necessary after fixing issues with IMOD export).
    - Find additional vesicles from the segmentation
      (= vesicles in the prediction that are not in the ground-truth)
      We can use them to define areas in the training data we don't use for training.
      Or to insert them as additional vesicles for training.
    """
    assert raw.shape == vesicle_gt.shape

    # Extract the bounding box of the data that contains the vesicles in the GT.
    raw, vesicle_gt, coordinates, seed_ids = extract_gt_bounding_box(raw, vesicle_gt, coordinates)

    # Get the model predictions and segmentation for this data.
    segmentation, prediction = segment_vesicles(raw, MODEL_PATH, return_predictions=True)

    # Additional post-processing to improve the shape of the vesicles.
    vesicle_gt = postprocess_vesicle_shape(vesicle_gt, prediction, coordinates, radii, seed_ids, raw)

    # Get vesicles in the prediction that are not part of the ground-truth.
    additional_vesicles = find_additional_vesicles(vesicle_gt, segmentation)

    return raw, vesicle_gt, additional_vesicles


def _mask_additional_vesicles(vesicle_gt, additional_vesicles, dilation_radius=2):
    mask = additional_vesicles != 0
    mask = binary_dilation(mask, iterations=dilation_radius)

    masked_vesicles = vesicle_gt.copy().astype("int16")
    masked_vesicles[mask] = -1

    return masked_vesicles


def create_vesicle_ground_truth_versions(input_path, output_path, gt_key):
    with h5py.File(input_path, "r") as f:
        raw = f["raw"][:]
        vesicle_gt = f[gt_key][:]
        coordinates = f["labels/imod/vesicles/coordinates"][:]
        radii = f["labels/imod/vesicles/radii"][:]

    # Extract raw data, vesicle gt and additional vesicles from the gt bounding box.
    raw, vesicle_gt, additional_vesicles = postprocess_vesicle_gt(raw, vesicle_gt, coordinates, radii)

    # Create a new ground-truth version where all additional vesicles are masked out with -1.
    masked_vesicles = _mask_additional_vesicles(vesicle_gt, additional_vesicles)

    # Create a new ground_truth version by merging ground-truth and additional vesicles.
    combined_vesicles = vesicle_gt.copy()
    offset = vesicle_gt.max() + 1
    extra_vesicles = additional_vesicles.copy()
    extra_mask = additional_vesicles != 0
    extra_vesicles[extra_mask] += offset
    combined_vesicles[extra_mask] = extra_vesicles[extra_mask]

    # Save all ground-truth data.
    with h5py.File(output_path, "a") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        # The original vesicle ground-truth (but cut to the bounding box).
        f.create_dataset("labels/vesicles/imod", data=vesicle_gt, compression="gzip")
        # The additional vesicles that were extracted from the segmentation.
        f.create_dataset("labels/vesicles/additional_vesicles", data=additional_vesicles, compression="gzip")
        # The ground-truth where additional vesicles are masked out.
        f.create_dataset("labels/vesicles/masked_vesicles", data=masked_vesicles, compression="gzip")
        # The ground-truth where additional vesicles are added to the original ground-truth.
        f.create_dataset("labels/vesicles/combined_vesicles", data=combined_vesicles, compression="gzip")


def process_files(input_path, output_root, label_key):
    input_files, input_root = _get_file_paths(input_path, ext=".h5")
    for path in tqdm(input_files):
        input_folder, fname = os.path.split(path)
        if input_root is None:
            output_path = os.path.join(output_root, fname)
        else:  # If we have nested input folders then we preserve the folder structure in the output.
            rel_folder = os.path.relpath(input_folder, input_root)
            output_path = os.path.join(output_root, rel_folder, fname)
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        create_vesicle_ground_truth_versions(path, output_path, label_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-k", "--label_key", default="labels/vesicles")
    args = parser.parse_args()

    process_files(args.input_path, args.output_folder, args.label_key)


if __name__ == "__main__":
    main()
