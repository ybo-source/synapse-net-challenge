import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch_em
import elf.parallel as parallel
import numpy as np
import xarray
from tqdm import tqdm

from skimage.transform import rescale, resize
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, remove_small_holes

DEFAULT_TILING = {
    "tile": {"x": 512, "y": 512, "z": 64},
    "halo": {"x": 64, "y": 64, "z": 8},
}


def _run_segmentation(
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


def _get_prediction_torch_em(
    input_volume, model_path,
    block_shape=(128, 256, 256),
    halo=(48, 48, 48),
    verbose=True
):
    t0 = time.time()
    # get foreground and boundary predictions from the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch_em.util.load_model(checkpoint=model_path, device=device)
    with torch.no_grad():
        pred = torch_em.util.prediction.predict_with_halo(
            input_volume, model, gpu_ids=device,
            block_shape=block_shape, halo=halo,
            preprocess=None,
        )
    if verbose:
        print("Prediction time in", time.time() - t0, "s")
    return pred


def segment_mitochondria(
    input_volume: np.ndarray,
    model_path: str,
    tiling: Dict[str, Dict[str, int]] = DEFAULT_TILING,
    min_size: int = 10000,
    verbose: bool = True,
    distance_based_segmentation: bool = False,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Segment mitochondria in an input volume.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint.
        tiling: The tiling configuration for the prediction.
        min_size: The minimum size of a mitochondria to be considered.
        verbose: Whether to print timing information.
        distance_based_segmentation: Whether to use distance-based segmentation.
        return_predictions: Whether to return the predictions (foreground, boundaries) alongside the segmentation.
        scale: The scale factor to use for rescaling the input volume before prediction.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask and the predictions if return_predictions is True.
    """
    if verbose:
        print("Segmenting mitochondria in volume of shape", input_volume.shape)

    if return_predictions:
        assert scale is None

    if scale is not None:
        original_shape = input_volume.shape
        input_volume = rescale(input_volume, scale, preserve_range=True).astype(input_volume.dtype)
        if verbose:
            print("Rescaled volume from", original_shape, "to", input_volume.shape)
    
    # get block_shape and halo
    block_shape = [tiling["tile"]["z"], tiling["tile"]["x"], tiling["tile"]["y"]]
    halo = [tiling["halo"]["z"], tiling["halo"]["x"], tiling["halo"]["y"]]

    t0 = time.time()
    
    pred = _get_prediction_torch_em(input_volume, model_path, block_shape=block_shape, halo=halo)

    foreground, boundaries = pred[:2]
    if verbose:
        print("Run prediction in", time.time() - t0, "s")

    seg = _run_segmentation(
        foreground, boundaries, verbose=verbose, min_size=min_size
    )

    if scale is not None:
        assert seg.ndim == input_volume.ndim
        seg = resize(seg, original_shape, preserve_range=True, order=0, anti_aliasing=False).astype(seg.dtype)
        assert seg.shape == original_shape

    if return_predictions:
        return seg, pred
    return seg
