import time

import bioimageio.core
import elf.parallel as parallel
import numpy as np
import xarray

from skimage.transform import rescale, resize

DEFAULT_TILING = {
    "tile": {"x": 512, "y": 512, "z": 64},
    "halo": {"x": 64, "y": 64, "z": 8},
}


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
    seeds = np.zeros(bd_dist.shape, dtype="uint64")
    seeds = parallel.label(bd_dist > distance_threshold, seeds, block_shape=block_shape, verbose=verbose)
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
    seeds = np.zeros(foreground.shape, dtype="uint64")
    seeds = parallel.label(
        (foreground - boundaries) > 0.5,
        out=seeds,
        block_shape=block_shape,
        verbose=verbose,
    )
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
    input_volume, model_path,
    tiling=DEFAULT_TILING,
    min_size=500, verbose=True,
    distance_based_segmentation=False,
    return_predictions=False,
    scale=None,
):
    if verbose:
        print("Segmenting vesicles in volume of shape", input_volume.shape)

    if return_predictions:
        assert scale is None

    if scale is not None:
        original_shape = input_volume.shape
        input_volume = rescale(input_volume, scale, preserve_range=True).astype(input_volume.dtype)
        if verbose:
            print("Rescaled volume from", original_shape, "to", input_volume.shape)

    t0 = time.time()
    # get foreground and boundary predictions from the model
    model = bioimageio.core.load_resource_description(model_path)
    with bioimageio.core.create_prediction_pipeline(model) as pp:
        input_ = xarray.DataArray(input_volume[None, None], dims=tuple("bczyx"))
        pred = bioimageio.core.predict_with_tiling(pp, input_, tiling=tiling, verbose=verbose)[0].squeeze()

    foreground, boundaries = pred[:2]
    if verbose:
        print("Run prediction in", time.time() - t0, "s")

    if distance_based_segmentation:
        seg = _run_distance_segmentation_parallel(
            foreground, boundaries, verbose=verbose, min_size=min_size,
        )
    else:
        seg = _run_segmentation_parallel(
            foreground, boundaries, verbose=verbose, min_size=min_size
        )

    if scale is not None:
        assert seg.ndim == input_volume.ndim
        seg = resize(seg, original_shape, preserve_range=True, order=0, anti_aliasing=False).astype(seg.dtype)
        assert seg.shape == original_shape

    if return_predictions:
        return seg, pred
    return seg
