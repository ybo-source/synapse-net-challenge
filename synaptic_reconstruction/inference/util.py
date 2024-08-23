import time
from typing import Dict
import numpy as np
import torch
import torch_em
from torch_em.util.prediction import predict_with_halo


def get_prediction_torch_em(
    input_volume: np.ndarray,  # [z, y, x]
    model_path: str,
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    verbose: bool = True,
) -> np.ndarray:
    """
    Run prediction using torch-em on a given volume.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.

    Returns:
        The prediction volume.
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
        )
    if verbose:
        print("Prediction time in", time.time() - t0, "s")
    return pred
