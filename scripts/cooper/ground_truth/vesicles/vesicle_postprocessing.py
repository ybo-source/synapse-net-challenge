import argparse
import os

import h5py
import numpy as np

from elf.evaluation.matching import label_overlap, intersection_over_union
from skimage.segmentation import relabel_sequential
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from synaptic_reconstruction.inference.vesicles import segment_vesicles
from synaptic_reconstruction.inference.util import _get_file_paths

MODEL_PATH = "/scratch-grete/projects/nim00007/data/synaptic_reconstruction/models/cooper/vesicles/3D-UNet-for-Vesicle-Segmentation-vesicles-010508model_v1r45_0105mr45_0105mr45.zip"  # noqa


def extract_gt_bounding_box(raw, vesicle_gt, halo=[2, 32, 32]):
    bb = np.where(vesicle_gt > 0)
    bb = tuple(slice(
        max(int(b.min() - ha), 0),
        min(int(b.max()) + ha, sh)
    ) for b, sh, ha in zip(bb, raw.shape, halo))
    raw, vesicle_gt = raw[bb], vesicle_gt[bb]
    return raw, vesicle_gt


# TODO
# Postprocess the vesicle shape (if still necessary after fixing the IMOD extraction).
def postprocess_vesicle_shape(vesicle_gt, prediction):
    return vesicle_gt


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


def postprocess_vesicle_gt(raw, vesicle_gt):
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
    raw, vesicle_gt = extract_gt_bounding_box(raw, vesicle_gt)

    # Get the model predictions and segmentation for this data.
    segmentation, prediction = segment_vesicles(raw, MODEL_PATH, return_predictions=True)

    # Additional post-processing to improve the shape of the vesicles.
    vesicle_gt = postprocess_vesicle_shape(vesicle_gt, prediction)

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

    # Extract raw data, vesicle gt and additional vesicles from the gt bounding box.
    raw, vesicle_gt, additional_vesicles = postprocess_vesicle_gt(raw, vesicle_gt)

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
