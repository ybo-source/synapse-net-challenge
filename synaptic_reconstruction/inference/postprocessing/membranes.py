import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops

from elf.parallel import label


def segment_membrane_next_to_object(
    boundary_prediction: np.array,
    object_segmentation: np.array,
    n_slices_exclude: int,
    radius: int = 25,
):
    """Derive boundary segmentation from boundary predictions by
    selecting large boundary fragment closest to the object.

    Args:
        boundary_prediction: Binary prediction for boundaries in the tomogram.
        object_segmentation: The object segmentation.
        n_slices_exclude: The number of slices to exclude on the top / bottom
            in order to avoid segmentation errors due to imaging artifacts in top and bottom.
        radius: The radius for membrane fragments that are considered.
    """
    assert boundary_prediction.shape == object_segmentation.shape

    original_shape = boundary_prediction.shape

    # Cut away the exclude mask.
    slice_mask = np.s_[n_slices_exclude:-n_slices_exclude]
    boundary_prediction = boundary_prediction[slice_mask]
    object_segmentation = object_segmentation[slice_mask]

    # Label the boundary predictions.
    boundary_segmentation = np.zeros(boundary_prediction.shape, dtype="uint32")
    boundary_segmentation = label(boundary_prediction, boundary_segmentation, block_shape=(32, 256, 256))

    # Compute the distance to object and the corresponding index.
    object_dist = distance_transform_edt(object_segmentation == 0)

    # Find the distances to the object and fragment size.
    ids = []
    distances = []
    sizes = []

    props = regionprops(boundary_segmentation)
    for prop in props:
        bb = prop.bbox
        bb = np.s_[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]

        label_id = prop.label
        boundary_mask = boundary_segmentation[bb] == label_id
        dist = object_dist[bb].copy()
        dist[~boundary_mask] = np.inf

        min_dist = np.min(dist)
        size = prop.area

        ids.append(prop.label)
        distances.append(min_dist)
        sizes.append(size)

    ids, distances, sizes = np.array(ids), np.array(distances), np.array(sizes)

    mask = distances < radius
    if mask.sum() > 0:
        ids, sizes = ids[mask], sizes[mask]
    min_boundary_id = ids[np.argmax(sizes)]

    # Create the output segmentation for the full output shape,
    # keeping only the boundary fragment closest to the PD.
    full_boundary_segmentation = np.zeros(original_shape, dtype="uint8")
    full_boundary_segmentation[slice_mask][boundary_segmentation == min_boundary_id] = 1

    return full_boundary_segmentation
