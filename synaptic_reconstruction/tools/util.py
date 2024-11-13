import os
import shutil
from typing import Callable, Optional, Tuple, Union
import torch_em
import torch
import numpy as np
import pooch
import requests
from torch_em.util.prediction import predict_with_halo


def download_and_organize_file(url, mode_type, app_name="synapse-net/models"):
    """
    Download a file from a URL and organize it in the specified directory within Pooch's default cache location.
    If the file has no extension, it saves it as "best.pt" inside a new or existing folder.
    
    Parameters:
        url (str): The URL of the file to download.
        app_name (str): The application name for Pooch's cache directory.
        
    Returns:
        str: Path to the directory containing the file.
    """
    # Get Pooch's default cache directory for the application
    cache_dir = pooch.os_cache(app_name)
    
    # Download the file using Pooch
    file_path = pooch.retrieve(url, path=cache_dir, known_hash=None)
    dir_name, file_name = os.path.split(file_path)
    os.rename(file_path, os.path.join(dir_name, mode_type, "best.pt"))
    return os.path.join(dir_name, mode_type, "best.pt")
    
    
    # # If file_path is a directory or has no extension, organize it into a folder with "best.pt"
    # if os.path.isdir(file_path) or os.path.splitext(file_path)[1] == "":
    #     # Create a directory for the file if needed
    #     dir_path = file_path if os.path.isdir(file_path) else os.path.splitext(file_path)[0]
    #     os.makedirs(dir_path, exist_ok=True)
        
    #     # Move the file into the directory with the name "best.pt"
    #     new_file_path = os.path.join(dir_path, "best.pt")
    #     if file_path != new_file_path:
    #         os.rename(file_path, new_file_path)
        
    #     return dir_path
    # else:
    #     # Return the directory where the file resides
    #     return os.path.dirname(file_path)


def organize_file_path(path):
    # Check if path is a file or directory
    if os.path.isfile(path):
        # Split path into directory, base name, and extension
        dir_name, file_name = os.path.split(path)
        base_name, ext = os.path.splitext(file_name)

        # Check if the file has no extension
        if not ext:
            # Temporary rename to avoid conflict with directory name
            temp_file_path = os.path.join(dir_name, base_name + "_temp")
            os.rename(path, temp_file_path)
            
            # Create a new directory with the original file name
            new_dir = os.path.join(dir_name, base_name)
            os.makedirs(new_dir, exist_ok=True)

            # Move the file to the new directory and rename it to "best.pt"
            new_path = os.path.join(new_dir, "best.pt")
            shutil.move(temp_file_path, new_path)
            
            print(f"Moved file to: {new_path}")
            return new_dir
        else:
            print("File already has an extension.")
            return dir_name
    elif os.path.isdir(path):
        print("Path is a directory.")
        return path
    else:
        print("Path does not exist.")
        return None


def download_model(url, out, name=None):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename


def get_cache_dir():
    cache_dir = os.path.expanduser(pooch.os_cache("synapse-net"))
    return cache_dir


def get_model_registry():
    registry = {
        "- choose -": None,
        "mitochondria": "xyz",
        "vesicles": "sha256:ab66416f979473f2f8bfa1f6e461d4a29e2bc17901e95cc65751218143e16c83",
            #"sha256:b17f6072fd6752a0caf32400a938cfe9f011941027d849014447123caad288e3",
    }
    urls = {
        "- choose -": None,
        "mitochondria": "https://github.com/computational-cell-analytics/synapse-net/releases/download/v0.0.1/mitochondria_model.zip",
        "vesicles": "https://owncloud.gwdg.de/index.php/s/tiyODdXOlSBNJIt/download"
            #"https://owncloud.gwdg.de/index.php/s/tiyODdXOlSBNJIt",
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
        print("Using apple MPS device.")
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


def get_model(model_path: str, model_class=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(model_path):
        if ".pt" in model_path:
            model_path = os.path.dirname(model_path)
        # model = torch_em.util.load_model(checkpoint=model_path, device=device)
        model_state = torch.load(model_path, map_location=device, weights_only=True)
        model = model_class()
        model.load_state_dict(model_state)
        print("Model loaded from checkpoint:", model_path)
        return model
    else:
        print(f"Model checkpoint not found at {model_path}.")
        return None


def run_prediction(
    input: np.ndarray,
    model: torch.nn.Module,
    block_shape: Tuple[int, int, int],
    halo: Tuple[int, int, int],
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch_em.transform.raw.standardize(input)
    return predict_with_halo(
        input_=input,
        model=model,
        gpu_ids=[device],
        block_shape=block_shape,
        halo=halo,
    )
