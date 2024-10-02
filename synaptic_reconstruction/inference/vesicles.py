import time
from typing import Dict, List, Optional, Tuple, Union

import elf.parallel as parallel
import numpy as np

import torch

from skimage.transform import rescale, resize
from synaptic_reconstruction.inference.util import get_prediction, get_default_tiling
from synaptic_reconstruction.inference.postprocessing.vesicles import filter_zborder_objects


def _run_distance_segmentation_parallel(
    foreground, boundaries, verbose, min_size,
    # blocking shapes for parallel computation
    distance_threshold=8,
    block_shape=(128, 256, 256),
    halo=(48, 48, 48)
):
    # compute the boundary distances
    t0 = time.time()
    bd_dist = parallel.distance_transform(boundaries < 0.9, halo=halo, verbose=verbose, block_shape=block_shape)
    bd_dist[foreground < 0.5] = 0
    if verbose:
        print("Compute distance transform in", time.time() - t0, "s")

    # get the segmentation via seeded watershed
    t0 = time.time()
    seeds = parallel.label(bd_dist > distance_threshold, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    t0 = time.time()
    dist = parallel.distance_transform(seeds == 0, halo=halo, verbose=verbose, block_shape=block_shape)
    if verbose:
        print("Compute distance transform in", time.time() - t0, "s")

    t0 = time.time()
    mask = (foreground + boundaries) > 0.5
    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        dist, seeds, block_shape=block_shape,
        out=seg, mask=mask, verbose=verbose, halo=halo,
    )
    if verbose:
        print("Compute watershed in", time.time() - t0, "s")

    t0 = time.time()
    ids, sizes = parallel.unique(seg, return_counts=True, block_shape=block_shape, verbose=verbose)
    filter_ids = ids[sizes < min_size]
    seg[np.isin(seg, filter_ids)] = 0
    if verbose:
        print("Size filter in", time.time() - t0, "s")
    return seg

def _run_segmentation_parallel(
    foreground, boundaries, verbose, min_size,
    # blocking shapes for parallel computation
    block_shape=(128, 256, 256),
    halo=(48, 48, 48)
):

    # get the segmentation via seeded watershed
    t0 = time.time()
    seeds = parallel.label((foreground - boundaries) > 0.5, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    t0 = time.time()
    dist = parallel.distance_transform(seeds == 0, halo=halo, verbose=verbose, block_shape=block_shape)
    if verbose:
        print("Compute distance transform in", time.time() - t0, "s")

    t0 = time.time()
    mask = (foreground + boundaries) > 0.5
    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        dist, seeds, block_shape=block_shape,
        out=seg, mask=mask, verbose=verbose, halo=halo,
    )
    if verbose:
        print("Compute watershed in", time.time() - t0, "s")

    t0 = time.time()
    ids, sizes = parallel.unique(seg, return_counts=True, block_shape=block_shape, verbose=verbose)
    filter_ids = ids[sizes < min_size]
    seg[np.isin(seg, filter_ids)] = 0
    if verbose:
        print("Size filter in", time.time() - t0, "s")
    return seg


def segment_vesicles(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    min_size: int = 500,
    verbose: bool = True,
    distance_based_segmentation: bool = False,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
    exclude_boundary: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Segment vesicles in an input volume.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if 'model' is not provided.
        model: Pre-loaded model. Either model_path or model is required.
        tiling: The tiling configuration for the prediction.
        min_size: The minimum size of a vesicle to be considered.
        verbose: Whether to print timing information.
        distance_based_segmentation: Whether to use distance-based segmentation.
        return_predictions: Whether to return the predictions (foreground, boundaries) alongside the segmentation.
        scale: The scale factor to use for rescaling the input volume before prediction.
        exclude_boundary: Whether to exclude vesicles that touch the upper / lower border in z.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask
        and the predictions if return_predictions is True.
    """
    #make sure either model path or model is passed
    if model is None and model_path is None:
        raise ValueError("Either 'model_path' or 'model' must be provided.")

    if verbose:
        print("Segmenting vesicles in volume of shape", input_volume.shape)

    if return_predictions:
        assert scale is None

    if scale is not None:
        original_shape = input_volume.shape
        input_volume = rescale(input_volume, scale, preserve_range=True).astype(input_volume.dtype)
        if verbose:
            print("Rescaled volume from", original_shape, "to", input_volume.shape)

    if tiling is None:
        tiling = get_default_tiling()

    pred = get_prediction(input_volume, tiling, model_path, model, verbose)
    foreground, boundaries = pred[:2]

    #deal with 2D segmentation case
    kwargs = {}
    if len(input_volume.shape) == 2:
        kwargs['block_shape'] = (256, 256)
        kwargs['halo'] = (48, 48) 

    if distance_based_segmentation:
        seg = _run_distance_segmentation_parallel(
            foreground, boundaries, verbose=verbose, min_size=min_size, **kwargs 
        )
    else:
        seg = _run_segmentation_parallel(
            foreground, boundaries, verbose=verbose, min_size=min_size, **kwargs 
        )

    if exclude_boundary:
        seg = filter_zborder_objects(seg)

    if scale is not None:
        assert seg.ndim == input_volume.ndim
        seg = resize(seg, original_shape, preserve_range=True, order=0, anti_aliasing=False).astype(seg.dtype)
        assert seg.shape == original_shape

    if return_predictions:
        return seg, pred
    return seg
