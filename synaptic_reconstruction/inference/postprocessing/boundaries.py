import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from tqdm import tqdm


def segment_boundary_next_to_pd(
    boundary_prediction: np.array,
    pd_segmentation: np.array,
    n_slices_exclude: int = 15,
):
    """Derive boundary segmentation from boundary predictions by
    selecting the boundary fragment closest to the PD.

    Args:
        boundary_prediction: Binary prediction for boundaries in the tomogram.
        vesicle_segmentation: The presynaptic density segmentation.
        n_slices_exclude: The number of slices to exclude on the top / bottom
            in order to avoid segmentation errors due to imaging artifacts in top and bottom.
    """
    assert boundary_prediction.shape == pd_segmentation.shape

    original_shape = boundary_prediction.shape

    # Cut away the exclude mask.
    slice_mask = np.s_[n_slices_exclude:-n_slices_exclude]
    boundary_prediction = boundary_prediction[slice_mask]
    pd_segmentation = pd_segmentation[slice_mask]

    # Compute the distance to ribbon and the corresponding index.
    pd_dist = distance_transform_edt(pd_segmentation == 0)

    # Label the boundary predictions.
    boundary_segmentation = label(boundary_prediction)

    # Find the boundary fragment closest to the PD.
    min_pd_dist = np.inf
    min_boundary_id = None

    props = regionprops(boundary_segmentation)
    for prop in tqdm(props):
        bb = prop.bbox
        bb = np.s_[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]

        label_id = prop.label
        boundary_mask = boundary_segmentation[bb] == label_id
        dist = pd_dist[bb]
        dist[~boundary_mask] = np.inf

        min_dist = np.min(dist)
        if min_dist < min_pd_dist:
            min_pd_dist = min_dist
            min_boundary_id = label_id

    assert min_boundary_id is not None

    # Create the output segmentation for the full output shape,
    # keeping only the boundary fragment closest to the PD.
    full_boundary_segmentation = np.zeros(original_shape, dtype="uint8")
    full_boundary_segmentation[slice_mask][boundary_segmentation == min_boundary_id] = 1

    return full_boundary_segmentation
