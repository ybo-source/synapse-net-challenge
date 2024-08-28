import time
from typing import Dict

import bioimageio.core
import numpy as np
import torch
import torch_em
import xarray

from torch_em.util.prediction import predict_with_halo


def get_prediction(
    input_volume: np.ndarray,  # [z, y, x]
    model_path: str,
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    verbose: bool = True,
    with_channels: bool = False,

):
    """
    Run prediction on a given volume.

    This function will automatically choose the correct prediction implementation,
    depending on the model type.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        with_channels: Whether to predict with channels.

    Returns:
        The predicted volume.
    """
    is_bioimageio = model_path.endswith(".zip")

    # We standardize the data for the whole volume beforehand.
    # If we have channels then the standardization is done independently per channel.
    if with_channels:
        # TODO Check that this is the correct axis.
        input_volume = torch_em.transform.raw.standardize(input_volume, axis=[1, 2, 3])
    else:
        input_volume = torch_em.transform.raw.standardize(input_volume)

    if is_bioimageio:
        # TODO determine if we use the old or new API and select the corresponding function
        pred = get_prediction_bioimageio_old(input_volume, model_path, tiling, verbose)
    else:
        pred = get_prediction_torch_em(input_volume, model_path, tiling, verbose, with_channels)
    return pred


def get_prediction_bioimageio_old(
    input_volume: np.ndarray,  # [z, y, x]
    model_path: str,
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    verbose: bool = True,
):
    """
    Run prediction using bioimage.io functionality on a given volume.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.

    Returns:
        The predicted volume.
    """
    # get foreground and boundary predictions from the model
    t0 = time.time()
    model = bioimageio.core.load_resource_description(model_path)
    with bioimageio.core.create_prediction_pipeline(model) as pp:
        input_ = xarray.DataArray(input_volume[None, None], dims=tuple("bczyx"))
        pred = bioimageio.core.predict_with_tiling(pp, input_, tiling=tiling, verbose=verbose)[0].squeeze()
    if verbose:
        print("Prediction time in", time.time() - t0, "s")
    return pred


def get_prediction_torch_em(
    input_volume: np.ndarray,  # [z, y, x]
    model_path: str,
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    verbose: bool = True,
    with_channels: bool = False,
) -> np.ndarray:
    """
    Run prediction using torch-em on a given volume.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        with_channels: Whether to predict with channels.

    Returns:
        The predicted volume.
    """
    # get block_shape and halo
    block_shape = [tiling["tile"]["z"], tiling["tile"]["x"], tiling["tile"]["y"]]
    halo = [tiling["halo"]["z"], tiling["halo"]["x"], tiling["halo"]["y"]]

    t0 = time.time()
    # get foreground and boundary predictions from the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch_em.util.load_model(checkpoint=model_path, device=device)
    with torch.no_grad():
        pred = predict_with_halo(
            input_volume, model, gpu_ids=[device],
            block_shape=block_shape, halo=halo,
            preprocess=None,
            with_channels=with_channels
        )
    if verbose:
        print("Prediction time in", time.time() - t0, "s")
    return pred
