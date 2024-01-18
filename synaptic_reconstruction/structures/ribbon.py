import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from tqdm import tqdm


def segment_ribbon(
    ribbon_prediction: np.array,
    vesicle_segmentation: np.array,
    n_slices_exclude: int = 15,
    max_vesicle_distance: int = 10,
    min_vesicles_per_ribbon: int = 15,
    require_ribbon=True,
):
    """Derive ribbon segmentation from ribbon predictions by
    filtering out ribbons that don't have sufficient associated vesicles.

    Args:
        ribbon_prediction: Binary prediction for ribbons in the tomogram.
        vesicle_segmentation: The vesicle segmentation.
        n_slices_exclude: The number of slices to exclude on the top / bottom
            in order to avoid segmentation errors due to imaging artifacts in top and bottom.
        max_vesicle_distance: The maximal distance to associate a vesicle with a ribbon.
        min_vesicles_per_ribbon: The minimal number of vesicles that have to be associated
            with an object in order to count it as a ribbon.
    """
    assert ribbon_prediction.shape == vesicle_segmentation.shape

    original_shape = ribbon_prediction.shape

    # Cut away the exclude mask.
    slice_mask = np.s_[n_slices_exclude:-n_slices_exclude]
    ribbon_prediction = ribbon_prediction[slice_mask]
    vesicle_segmentation = vesicle_segmentation[slice_mask]

    # Compute the distance to ribbon and the corresponding index.
    ribbon_dist, ribbon_idx = distance_transform_edt(~ribbon_prediction, return_indices=True)
    # Label the ribbon predictions.
    ribbon_segmentation = label(ribbon_prediction)

    # Count the number of vesicles associated with each foreground object in the ribbon prediction.
    vesicle_counts = {}
    props = regionprops(vesicle_segmentation)
    for prop in tqdm(props):
        bb = prop.bbox
        bb = np.s_[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]

        vesicle_mask = vesicle_segmentation[bb] == prop.label
        dist, idx = ribbon_dist[bb], ribbon_idx[(slice(None),) + bb]
        dist[~vesicle_mask] = np.inf

        min_dist_point = np.argmin(dist)
        min_dist_point = np.unravel_index(min_dist_point, vesicle_mask.shape)

        if dist[min_dist_point] > max_vesicle_distance:
            continue

        ribbon_coord = tuple(idx_[min_dist_point] for idx_ in idx)
        ribbon_id = ribbon_segmentation[ribbon_coord]
        assert ribbon_id != 0

        if ribbon_id in vesicle_counts:
            vesicle_counts[ribbon_id] += 1
        else:
            vesicle_counts[ribbon_id] = 1

    # Create the output segmentation for the full output shape,
    # keeping only the ribbons with sufficient number of associated vesicles.
    full_ribbon_segmentation = np.zeros(original_shape, dtype="uint8")
    output_id = 1

    max_count = max(vesicle_counts.values())
    if max_count < min_vesicles_per_ribbon and require_ribbon:
        print("The max count", max_count, "is smaller than", min_vesicles_per_ribbon)
        print("Resetting the min count to it")
        min_vesicles_per_ribbon = max_count

    for ribbon_id, vesicle_count in vesicle_counts.items():
        if vesicle_count >= min_vesicles_per_ribbon:
            full_ribbon_segmentation[slice_mask][ribbon_segmentation == ribbon_id] = output_id
            output_id += 1

    return full_ribbon_segmentation
