import os
import time
import warnings
from glob import glob
from typing import Dict, Optional

# Suppress annoying import warnings.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import bioimageio.core
import imageio.v3 as imageio
import numpy as np
import torch
import torch_em
import xarray

from elf.io import open_file
from torch_em.util.prediction import predict_with_halo
from tqdm import tqdm


def get_prediction(
    input_volume: np.ndarray,  # [z, y, x]
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    verbose: bool = True,
    with_channels: bool = False,
):
    """
    Run prediction on a given volume.

    This function will automatically choose the correct prediction implementation,
    depending on the model type.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint if 'model' is not provided.
        model: Pre-loaded model. Either model_path or model is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        with_channels: Whether to predict with channels.

    Returns:
        The predicted volume.
    """
    # We always use the same default halo.
    halo = {"x": 64, "y": 64, "z": 16}

    if model is not None:
        is_bioimageio = None
    else:
        is_bioimageio = model_path.endswith(".zip")
    

    # We standardize the data for the whole volume beforehand.
    # If we have channels then the standardization is done independently per channel.
    if with_channels:
        # TODO Check that this is the correct axis.
        input_volume = torch_em.transform.raw.standardize(input_volume, axis=(1, 2, 3))
    else:
        input_volume = torch_em.transform.raw.standardize(input_volume)

    if is_bioimageio:
        # TODO determine if we use the old or new API and select the corresponding function
        pred = get_prediction_bioimageio_old(input_volume, model_path, tiling, verbose)
    else:
        if model is None:
            # torch_em expects the root folder of a checkpoint path instead of the checkpoint itself.          
            if model_path.endswith("best.pt"):
                model_path = os.path.split(model_path)[0]
        print(f"tiling {tiling}")
        # Create updated_tiling with the same structure
        updated_tiling = {
            'tile': {},
            'halo': tiling['halo']  # Keep the halo part unchanged
        }
        # Update tile dimensions
        for dim in tiling['tile']:
            updated_tiling['tile'][dim] = tiling['tile'][dim] - tiling['halo'][dim]
        print(f"updated_tiling {updated_tiling}")
        pred = get_prediction_torch_em(input_volume, updated_tiling, model_path, model, verbose, with_channels)

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
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    verbose: bool = True,
    with_channels: bool = False,
) -> np.ndarray:
    """
    Run prediction using torch-em on a given volume.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint if 'model' is not provided.
        model: Pre-loaded model. Either model_path or model is required.
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        if os.path.isdir(model_path):  # Load the model from a torch_em checkpoint.
            model = torch_em.util.load_model(checkpoint=model_path, device=device)
        else:  # Load the model directly from a serialized pytorch model.
            model = torch.load(model_path)

    # Run prediction with the model.
    with torch.no_grad():
        
        #deal with 2D segmentation case
        if len(input_volume.shape) == 2:
            block_shape = [block_shape[1], block_shape[2]]
            halo = [halo[1], halo[2]]

        pred = predict_with_halo(
            input_volume, model, gpu_ids=[device],
            block_shape=block_shape, halo=halo,
            preprocess=None,
            with_channels=with_channels
        )
    if verbose:
        print("Prediction time in", time.time() - t0, "s")
    return pred


def _get_file_paths(input_path, ext=".mrc"):
    if not os.path.exists(input_path):
        raise Exception(f"Input path not found {input_path}")

    if os.path.isfile(input_path):
        input_files = [input_path]
        input_root = None
    else:
        input_files = sorted(glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True))
        input_root = input_path

    return input_files, input_root


def _load_input(img_path, extra_files, i):
    # Load the input data data
    with open_file(img_path, "r") as f:

        # Try to automatically derive the key with the raw data.
        keys = list(f.keys())
        if len(keys) == 1:
            key = keys[0]
        elif "data" in keys:
            key = "data"
        elif "raw" in keys:
            key = "raw"

        input_volume = f[key][:]
    assert input_volume.ndim == 3

    # For now we assume this is always tif.
    if extra_files is not None:
        extra_input = imageio.imread(extra_files[i])
        assert extra_input.shape == input_volume.shape
        input_volume = np.stack([input_volume, extra_input], axis=0)

    return input_volume


def inference_helper(
    input_path: str,
    output_root: str,
    segmentation_function: callable,
    data_ext: str = ".mrc",
    extra_input_path: Optional[str] = None,
    extra_input_ext: str = ".tif",
    force: bool = False,
):
    """
    Helper function to run segmentation for mrc files.

    Args:
        input_path: The path to the input data.
            Can either be a folder. In this case all mrc files below the folder will be segmented.
            Or can be a single mrc file. In this case only this mrc file will be segmented.
        output_root: The path to the output directory where the segmentation results will be saved.
        segmentation_function: The function performing the segmentation.
            This function must take the input_volume as the only argument and must return only the segmentation.
            If you want to pass additional arguments to this function the use 'funtools.partial'
        data_ext: File extension for the image data. By default '.mrc' is used.
        extra_input_path: Filepath to extra inputs that need to be concatenated to the raw data loaded from mrc.
            This enables cristae segmentation with an extra mito channel.
        extra_input_ext: File extension for the extra inputs (by default .tif).
        force: Whether to rerun segmentation for output files that are already present.
    """
    # Get the input files. If input_path is a folder then this will load all
    # the mrc files beneath it. Otherwise we assume this is an mrc file already
    # and just return the path to this mrc file.
    input_files, input_root = _get_file_paths(input_path, data_ext)

    # Load extra inputs if the extra_input_path was specified.
    if extra_input_path is None:
        extra_files = None
    else:
        extra_files, _ = _get_file_paths(extra_input_path, extra_input_ext)
        assert len(input_files) == len(extra_files)

    for i, img_path in tqdm(enumerate(input_files), total=len(input_files)):
        # Determine the output file name.
        input_folder, input_name = os.path.split(img_path)
        fname = os.path.splitext(input_name)[0] + "_prediction.tif"
        if input_root is None:
            output_path = os.path.join(output_root, fname)
        else:  # If we have nested input folders then we preserve the folder structure in the output.
            rel_folder = os.path.relpath(input_folder, input_root)
            output_path = os.path.join(output_root, rel_folder, fname)

        # Check if the output path is already present.
        # If it is we skip the prediction, unless force was set to true.
        if os.path.exists(output_path) and not force:
            continue

        # Load the input volume. If we have extra_files then this concatenates the
        # data across a new first axis (= channel axis).
        input_volume = _load_input(img_path, extra_files, i)
        segmentation = segmentation_function(input_volume)

        # Write the result to tif.
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        imageio.imwrite(output_path, segmentation, compression="zlib")
        print(f"Saved segmentation to {output_path}.")


def get_default_tiling():
    """Determine the tile shape and halo depending on the available VRAM.
    """
    if torch.cuda.is_available():
        print("Determining suitable tiling")

        # We always use the same default halo.
        halo = {"x": 64, "y": 64, "z": 16} #before 64,64,8

        # Determine the GPU RAM and derive a suitable tiling.
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9

        if vram >= 80:
            tile = {"x": 640, "y": 640, "z": 80}
        elif vram >= 40:
            tile = {"x": 512, "y": 512, "z": 64}
        elif vram >= 20:
            tile = {"x": 352, "y": 352, "z": 48}
        else:
            # TODO determine tilings for smaller VRAM
            raise NotImplementedError

        print(f"using tile size: {tile}")
        tiling = {"tile": tile, "halo": halo}

    # I am not sure what is reasonable on a cpu. For now choosing very small tiling.
    # (This will not work well on a CPU in any case.)
    else:
        print("Using default tiling")
        tiling = {
            "tile": {"x": 96, "y": 96, "z": 16},
            "halo": {"x": 16, "y": 16, "z": 4},
        }

    return tiling


def parse_tiling(tile_shape, halo):
    """
    Helper function to parse tiling parameter input from the command line.

    Args:
        tile_shape: The tile shape. If None the default tile shape is used.
        halo: The halo. If None the default halo is used.

    Returns:
        dict: the tiling specification
    """
    default_tiling = get_default_tiling()

    if tile_shape is None:
        tile_shape = default_tiling["tile"]
    else:
        assert len(tile_shape) == 3
        tile_shape = dict(zip("zyx", tile_shape))

    if halo is None:
        halo = default_tiling["halo"]
    else:
        assert len(halo) == 3
        halo = dict(zip("zyx", halo))

    tiling = {"tile": tile_shape, "halo": halo}
    return tiling
