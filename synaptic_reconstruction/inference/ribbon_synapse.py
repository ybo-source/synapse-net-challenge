from typing import Dict, Sequence, Optional, Union

import numpy as np
import torch

from skimage.transform import rescale, resize
from synaptic_reconstruction.inference.util import get_prediction, get_default_tiling


def segment_ribbon_synapse_structures(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    structure_names: Sequence[str] = ("ribbon", "PD", "membrane"),
    verbose: bool = False,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    threshold: Optional[Union[float, Dict[str, float]]] = None,
    scale: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Segment ribbon synapse structures.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if 'model' is not provided.
        model: Pre-loaded model. Either model_path or model is required.
        structure_names: Names of the structures to be segmented.
            The default network segments the ribbon, presynaptic density (pd) an local memrane.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        threshold: The threshold for binarizing predictions.
        scale: The scale factor to use for rescaling the input volume before prediction.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask
        and the predictions if return_predictions is True.
    """
    if verbose:
        print(f"Segmenting synaptic structures: {structure_names} in volume of shape", input_volume.shape)

    if scale is not None:
        original_shape = input_volume.shape
        input_volume = rescale(input_volume, scale, preserve_range=True).astype(input_volume.dtype)
        if verbose:
            print("Rescaled volume from", original_shape, "to", input_volume.shape)

    if tiling is None:
        tiling = get_default_tiling()
    predictions = get_prediction(
        input_volume=input_volume, tiling=tiling, model_path=model_path, model=model, verbose=verbose
    )
    assert len(structure_names) == predictions.shape[0]

    if scale is not None:
        assert predictions.ndim == input_volume.ndim + 1
        original_shape = (predictions.shape[0],) + original_shape
        predictions = resize(predictions, original_shape, preserve_range=True,).astype(predictions.dtype)
        assert predictions.shape == original_shape

    predictions = {name: predictions[i] for i, name in enumerate(structure_names)}
    if threshold is not None:
        for name in structure_names:
            # We can either have a single threshold value or a threshold per structure
            # that is given as a dictionary.
            this_threshold = threshold if isinstance(threshold, float) else threshold[name]
            predictions[name] = predictions[name] > this_threshold

    return predictions
