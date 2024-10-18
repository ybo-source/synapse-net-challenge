import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from scipy.ndimage import distance_transform_edt, binary_closing
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.morphology import remove_small_holes

from synaptic_reconstruction.inference.util import get_prediction, _Scaler


def _segment_compartments(
    prediction,
    boundary_threshold=0.4,  # Threshold for the boundary distance computation.
    large_seed_distance=30,  # The distance threshold for computing large seeds (= components).
    small_seed_distance=4,   # The distance threshold for small seeds that will be filtered in the result.
    verbose=False,
):
    t0 = time.time()

    # Compute the boundary distances.
    distances = distance_transform_edt(prediction < boundary_threshold)

    # Compute the large seeds. These are the seeds for the compartments to be kept.
    seeds = label(distances > large_seed_distance)

    # Compute additional small seeds. These are added so that the large seeds don't
    # flood over boundaries, but will be filtered from the segmentation result.
    ndim = distances.ndim
    if ndim == 2:
        small_seeds = label(distances > small_seed_distance)
    elif ndim == 3:  # In the 3d case we compute individual seeds per slice.
        small_seeds = np.zeros_like(seeds)
        offset = 0
        for z in range(small_seeds.shape[0]):
            this_seeds = label(distances[z] > small_seed_distance)
            this_seeds[this_seeds != 0] += offset
            offset = this_seeds.max()
            small_seeds[z] = this_seeds
    else:
        raise RuntimeError

    # We only keep small seeds that don't intersect with a large seed.
    props = regionprops(small_seeds, seeds)
    keep_seeds = [prop.label for prop in props if prop.max_intensity == 0]
    keep_mask = np.isin(small_seeds, keep_seeds)

    # Add up the small seeds we keep with the large seeds.
    seed_offset = seeds.max()
    seeds[keep_mask] = (small_seeds[keep_mask] + seed_offset)

    # Get the initial segmentation via watershed.
    raw_segmentation = watershed(prediction, markers=seeds)

    # Thee are the large seed ids that we will keep.
    keep_ids = list(range(1, seed_offset + 1))

    # Structure lement for 2d dilation in 3d.
    structure_element = np.ones((3, 3))  # 3x3 structure for XY plane
    structure_3d = np.zeros((1, 3, 3))  # Only applied in the XY plane
    structure_3d[0] = structure_element

    # Iterate over the ids, only keep large seeds and remove holes in their respective masks.
    props = regionprops(raw_segmentation)
    segmentation = np.zeros_like(raw_segmentation)
    for prop in props:
        if prop.label not in keep_ids:
            continue

        # Get bounding box and mask.
        bb = tuple(slice(start, stop) for start, stop in zip(prop.bbox[:ndim], prop.bbox[ndim:]))
        mask = raw_segmentation[bb] == prop.label

        # Fill small holes and apply closing.
        mask = remove_small_holes(mask, area_threshold=500)
        mask = np.logical_or(binary_closing(mask, iterations=4), mask)
        if ndim == 3:
            mask = np.logical_or(binary_closing(mask, iterations=8, structure=structure_3d), mask)
        segmentation[bb][mask] = prop.label

    if verbose:
        print("Segmentation time in", time.time() - t0, "s")

    # For debugging / development.
    # import napari
    # v = napari.Viewer()
    # v.add_image(prediction)
    # v.add_image(distances)
    # v.add_labels(seeds)
    # v.add_labels(small_seeds)
    # v.add_labels(segmentation)
    # napari.run()

    return segmentation


def segment_compartments(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    verbose: bool = True,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Segment synaptic compartments in an input volume.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if `model` is not provided.
        model: Pre-loaded model. Either `model_path` or `model` is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        return_predictions: Whether to return the predictions (foreground, boundaries) alongside the segmentation.
        scale: The scale factor to use for rescaling the input volume before prediction.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask
        and the predictions if return_predictions is True.
    """
    if verbose:
        print("Segmenting compartments in volume of shape", input_volume.shape)

    # Create the scaler to handle prediction with a different scaling factor.
    scaler = _Scaler(scale, verbose)
    input_volume = scaler.scale_input(input_volume)

    # Run prediction.
    pred = get_prediction(input_volume, tiling=tiling, model_path=model_path, model=model, verbose=verbose)

    # Remove channel axis if necessary.
    if pred.ndim != input_volume.ndim:
        assert pred.ndim == input_volume.ndim + 1
        assert pred.shape[0] == 1
        pred = pred[0]

    # Run the compartment segmentation.
    # We may want to expose some of the parameters here.
    seg = _segment_compartments(pred, verbose=verbose)
    seg = scaler.rescale_output(seg, is_segmentation=True)

    if return_predictions:
        pred = scaler.rescale_output(pred, is_segmentation=False)
        return seg, pred
    return seg
