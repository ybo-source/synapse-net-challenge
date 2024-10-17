import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from synaptic_reconstruction.inference.util import get_prediction, _Scaler


def _segment_compartments(prediction, boundary_threshold, seed_distance, filter_distance, verbose):
    t0 = time.time()
    distances = distance_transform_edt(prediction < boundary_threshold)
    seeds = label(distances > seed_distance)
    segmentation = watershed(prediction, markers=seeds)

    # Filter by max distance.
    props = regionprops(segmentation, distances)
    filter_ids = [prop.label for prop in props if prop.max_intensity < filter_distance]
    segmentation[np.isin(segmentation, filter_ids)] = 0

    if verbose:
        print("Segmentation time in", time.time() - t0, "s")

    # import napari
    # v = napari.Viewer()
    # v.add_image(prediction)
    # v.add_image(distances)
    # v.add_labels(seeds)
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

    # The parameters for the segmentation. The second two probably need to be optimized further.
    boundary_threshold = 0.5
    # The distance for thresholding the boundary distacne for computing connected components.
    seed_distance = 5
    # Segments with a smaller max distance than this will be filtered.
    filter_distance = 40
    seg = _segment_compartments(
        pred, boundary_threshold=boundary_threshold,
        filter_distance=filter_distance, seed_distance=seed_distance, verbose=verbose
    )
    seg = scaler.rescale_output(seg, is_segmentation=True)

    if return_predictions:
        pred = scaler.rescale_output(pred, is_segmentation=False)
        return seg, pred
    return seg
