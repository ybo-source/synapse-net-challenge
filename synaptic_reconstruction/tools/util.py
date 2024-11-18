import os
from typing import Dict, List, Optional, Union

import torch
import numpy as np
import pooch
import warnings

from ..inference.vesicles import segment_vesicles
from ..inference.mitochondria import segment_mitochondria


def get_model(model_type: str, device: Optional[Union[str, torch.device]] = None) -> torch.nn.Module:
    """Get the model for the given segmentation type.

    Args:
        model_type: The model type.
            One of 'vesicles', 'mitochondria', 'active_zone', 'compartments' or 'inner_ear_structures'.
        device: The device to use.

    Returns:
        The model.
    """
    device = get_device(device)
    model_registry = get_model_registry()
    model_path = model_registry.fetch(model_type)
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning
    )
    model = torch.load(model_path)
    model.to(device)
    return model


# TODO: distinguish between 2d and 3d vesicle model segmentation
def run_segmentation(
    image: np.ndarray,
    model: torch.nn.Module,
    model_type: str,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    scale: Optional[List[float]] = None,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """Run synaptic structure segmentation.

    Args:
        image: ...
        model: ...
        model_type: ...
        tiling: ...
        scale: ...
        verbose: ...

    Returns:
        The segmentation.
    """
    if model_type == "vesicles":
        segmentation = segment_vesicles(image, model=model, tiling=tiling, scale=scale, verbose=verbose)
    elif model_type == "mitochondria":
        segmentation = segment_mitochondria(image, model=model, tiling=tiling, scale=scale, verbose=verbose)
    elif model_type == "active_zone":
        raise NotImplementedError
    elif model_type == "compartments":
        raise NotImplementedError
    elif model_type == "inner_ear_structures":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return segmentation


def get_cache_dir():
    cache_dir = os.path.expanduser(pooch.os_cache("synapse-net"))
    return cache_dir


def get_model_registry():
    registry = {
        "mitochondria": "xyz",
        "vesicles": "sha256:e75714ea7bedd537d8eff822cb4c566b208dba1301fadf9d338a3914a353a331"
        # "sha256:ab66416f979473f2f8bfa1f6e461d4a29e2bc17901e95cc65751218143e16c83",
        # "sha256:b17f6072fd6752a0caf32400a938cfe9f011941027d849014447123caad288e3",
    }
    urls = {
        "mitochondria": "https://github.com/computational-cell-analytics/synapse-net/releases/download/v0.0.1/mitochondria_model.zip",  # noqa
        "vesicles": "https://owncloud.gwdg.de/index.php/s/7B0ILPf0A7VRt1G/download"
        # "https://owncloud.gwdg.de/index.php/s/tiyODdXOlSBNJIt/download"
        # "https://owncloud.gwdg.de/index.php/s/tiyODdXOlSBNJIt",
    }
    cache_dir = get_cache_dir()
    models = pooch.create(
        path=os.path.join(cache_dir, "models"),
        base_url="",
        registry=registry,
        urls=urls,
    )
    return models


def _get_default_device():
    # check that we're in CI and use the CPU if we are
    # otherwise the tests may run out of memory on MAC if MPS is used.
    if os.getenv("GITHUB_ACTIONS") == "true":
        return "cpu"
    # Use cuda enabled gpu if it's available.
    if torch.cuda.is_available():
        device = "cuda"
    # As second priority use mps.
    # See https://pytorch.org/docs/stable/notes/mps.html for details
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    # Use the CPU as fallback.
    else:
        device = "cpu"
    return device


def get_device(device: Optional[Union[str, torch.device]] = None) -> Union[str, torch.device]:
    """Get the torch device.

    If no device is passed the default device for your system is used.
    Else it will be checked if the device you have passed is supported.

    Args:
        device: The input device.

    Returns:
        The device.
    """
    if device is None or device == "auto":
        device = _get_default_device()
    else:
        device_type = device if isinstance(device, str) else device.type
        if device_type.lower() == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA backend is not available.")
        elif device_type.lower() == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                raise RuntimeError("PyTorch MPS backend is not available or is not built correctly.")
        elif device_type.lower() == "cpu":
            pass  # cpu is always available
        else:
            raise RuntimeError(f"Unsupported device: {device}\n"
                               "Please choose from 'cpu', 'cuda', or 'mps'.")
    return device


def _available_devices():
    available_devices = []
    for i in ["cuda", "mps", "cpu"]:
        try:
            device = get_device(i)
        except RuntimeError:
            pass
        else:
            available_devices.append(device)
    return available_devices


def get_current_tiling(tiling: dict, default_tiling: dict, image_shape):
    # get tiling values from qt objects
    for k, v in tiling.items():
        for k2, v2 in v.items():
            tiling[k][k2] = v2.value()
    # check if user inputs tiling/halo or not
    if default_tiling == tiling:
        if len(image_shape) == 2:
            # if its 2d image expand x,y and set z to 1
            tiling = {
                "tile": {
                    "x": 512,
                    "y": 512,
                    "z": 1
                },
                "halo": {
                    "x": 64,
                    "y": 64,
                    "z": 1
                }
            }
    elif len(image_shape) == 2:
        # if its a 2d image set z to 1
        tiling["tile"]["z"] = 1
        tiling["halo"]["z"] = 1
        
    return tiling
