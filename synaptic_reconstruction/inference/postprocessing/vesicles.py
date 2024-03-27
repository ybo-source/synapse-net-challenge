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
