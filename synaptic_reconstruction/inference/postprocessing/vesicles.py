import numpy as np

from scipy.ndimage import binary_closing
from skimage.measure import regionprops
from tqdm import tqdm


def close_holes(vesicle_segmentation, closing_iterations=4, min_size=0, verbose=False):
    assert vesicle_segmentation.ndim == 3
    props = regionprops(vesicle_segmentation)
    closed_segmentation = np.zeros_like(vesicle_segmentation)

    for prop in tqdm(props, desc="Close holes in segmentation", disable=not verbose):
        if prop.area < min_size:
            continue
        bb = prop.bbox
        bb = tuple(slice(beg, end) for beg, end in zip(bb[:3], bb[3:]))
        mask = vesicle_segmentation[bb] == prop.label
        closed_mask = np.logical_or(binary_closing(mask, iterations=closing_iterations), mask)
        closed_segmentation[bb][closed_mask] = prop.label

    return closed_segmentation


def filter_zborder_objects(segmentation):
    """Filter any object that touches the upper or lower z-border.
    """
    props = regionprops(segmentation)

    filter_ids = []
    for prop in props:
        bbox = prop.bbox
        z_start, z_stop = bbox[0], bbox[3]
        if z_start == 0 or z_stop == segmentation.shape[0]:
            filter_ids.append(prop.label)

    segmentation[np.isin(segmentation, filter_ids)] = 0
    return segmentation


def filter_border_vesicles(vesicle_segmentation, seg_ids=None, border_slices=4):
    props = regionprops(vesicle_segmentation)

    filtered_ids = []
    for prop in tqdm(props, desc="Filter vesicles at the tomogram border"):
        seg_id = prop.label
        if (seg_ids is not None) and (seg_id not in seg_ids):
            continue

        bb = prop.bbox
        bb = tuple(slice(beg, end) for beg, end in zip(bb[:3], bb[3:]))
        mask = vesicle_segmentation[bb] == seg_id

        # Compute the mass per slice. Only keep the vesicle if the maximum of the mass is central.
        mass_per_slice = [m.sum() for m in mask]
        max_slice = np.argmax(mass_per_slice)
        if (max_slice >= border_slices) and (max_slice < mask.shape[0] - border_slices):
            filtered_ids.append(seg_id)

    # print(len(filtered_ids), "/", len(seg_ids))
    return filtered_ids
