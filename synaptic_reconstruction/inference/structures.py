import time

from skimage.transform import rescale, resize
from synaptic_reconstruction.inference.util import get_prediction, DEFAULT_TILING


# TODO rename to inner_ear, add type annotations and doc string
def segment_structures(
    input_volume,
    model_path,
    structure_names,
    verbose=False,
    tiling=DEFAULT_TILING,
    threshold=None,
    scale=None,
):
    """Segment synaptic structures in the inner ear.
    """
    if verbose:
        print(f"Segmenting synaptic structures: {structure_names} in volume of shape", input_volume.shape)

    t0 = time.time()

    if scale is not None:
        original_shape = input_volume.shape
        input_volume = rescale(input_volume, scale, preserve_range=True).astype(input_volume.dtype)
        if verbose:
            print("Rescaled volume from", original_shape, "to", input_volume.shape)

    predictions = get_prediction(input_volume, model_path, tiling, verbose)
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

    if verbose:
        print("Run prediction in", time.time() - t0, "s")

    return predictions
