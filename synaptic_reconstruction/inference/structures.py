import time

import bioimageio.core
import xarray

from .vesicles import DEFAULT_TILING


def segment_structures(
    input_volume, model_path, structure_names,
    verbose=False, tiling=DEFAULT_TILING, threshold=None, scale=None,
):
    if verbose:
        print(f"Segmenting synaptic structures: {structure_names} in volume of shape", input_volume.shape)

    t0 = time.time()

    # TODO
    assert scale is None

    model = bioimageio.core.load_resource_description(model_path)
    with bioimageio.core.create_prediction_pipeline(model) as pp:
        input_ = xarray.DataArray(input_volume[None, None], dims=tuple("bczyx"))
        predictions = bioimageio.core.predict_with_tiling(
            pp, input_, tiling=tiling, verbose=verbose
        )[0].values.squeeze()
    assert len(structure_names) == predictions.shape[0]
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
