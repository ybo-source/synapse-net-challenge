import numpy as np

from elf.evaluation.matching import label_overlap, intersection_over_union
from skimage.segmentation import relabel_sequential


def find_additional_objects(ground_truth, segmentation, matching_threshold=0.5):
    segmentation = relabel_sequential(segmentation)[0]

    # Match the objects in the segmentation to the ground-truth.
    overlap, _ = label_overlap(segmentation, ground_truth)
    overlap = intersection_over_union(overlap)

    # Get the segmentation IDs.
    seg_ids = np.unique(segmentation)

    # Filter out IDs with a larger overlap than the matching threshold:
    # These likely correspond to an object covered by the ground-truth.
    filter_ids = []
    for seg_id in seg_ids[1:]:
        max_overlap = overlap[seg_id, :].max()
        if max_overlap > matching_threshold:
            filter_ids.append(seg_id)

    # Get the additional objects by removing filtered objects.
    additional_objects = segmentation.copy()
    additional_objects[np.isin(segmentation, filter_ids)] = 0
    additional_objects = relabel_sequential(additional_objects)[0]

    return additional_objects
