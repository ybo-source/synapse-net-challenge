import os
from typing import Dict, List, Optional, Union

import torch
import numpy as np
import pooch

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
    if model_type.startswith("vesicles"):
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


def get_model_training_resolution(model_type):
    resolutions = {
        "active_zone": 1.44,
        "compartments": 3.47,
        "mitochondria": 1.0,  # FIXME: this is a dummy value, we need to determine the real one
        "vesicles_2d": 1.35,
        "vesicles_3d": 1.35,
        "vesicles_cryo": 0.88,
    }
    return resolutions[model_type]


def get_model_registry():
    registry = {
        "active_zone": "a18f29168aed72edec0f5c2cb1aa9a4baa227812db6082a6538fd38d9f43afb0",
        "compartments": "527983720f9eb215c45c4f4493851fd6551810361eda7b79f185a0d304274ee1",
        "mitochondria": "24625018a5968b36f39fa9d73b121a32e8f66d0f2c0540d3df2e1e39b3d58186",
        "vesicles_2d": "eb0b74f7000a0e6a25b626078e76a9452019f2d1ea6cf2033073656f4f055df1",
        "vesicles_3d": "b329ec1f57f305099c984fbb3d7f6ae4b0ff51ec2fa0fa586df52dad6b84cf29",
        "vesicles_cryo": "782f5a21c3cda82c4e4eaeccc754774d5aaed5929f8496eb018aad7daf91661b",
    }
    urls = {
        "active_zone": "https://owncloud.gwdg.de/index.php/s/zvuY342CyQebPsX/download",
        "compartments": "https://owncloud.gwdg.de/index.php/s/DnFDeTmDDmZrDDX/download",
        "mitochondria": "https://owncloud.gwdg.de/index.php/s/1T542uvzfuruahD/download",
        "vesicles_2d": "https://owncloud.gwdg.de/index.php/s/d72QIvdX6LsgXip/download",
        "vesicles_3d": "https://owncloud.gwdg.de/index.php/s/A425mkAOSqePDhx/download",
        "vesicles_cryo": "https://owncloud.gwdg.de/index.php/s/e2lVdxjCJuZkLJm/download",
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
