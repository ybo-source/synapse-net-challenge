import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import vigra
import torch

import elf.segmentation as eseg
import nifty
from elf.tracking.tracking_utils import compute_edges_from_overlap
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from synaptic_reconstruction.inference.util import apply_size_filter, get_prediction, _Scaler


def _multicut(ws, prediction, beta, n_threads):
    rag = eseg.features.compute_rag(ws, n_threads=n_threads)
    edge_features = eseg.features.compute_boundary_mean_and_length(rag, prediction, n_threads=n_threads)
    edge_probs, edge_sizes = edge_features[:, 0], edge_features[:, 1]
    edge_costs = eseg.multicut.compute_edge_costs(edge_probs, edge_sizes=edge_sizes, beta=beta)
    node_labels = eseg.multicut.multicut_kernighan_lin(rag, edge_costs)
    seg = eseg.features.project_node_labels_to_pixels(rag, node_labels, n_threads)
    return seg


def _segment_compartments_2d(
    prediction,
    distances=None,
    boundary_threshold=0.4,
    beta=0.6,
    n_threads=1,
    run_multicut=True,
    min_size=500,
):
    if distances is None:
        distances = distance_transform_edt(prediction < boundary_threshold).astype("float32")

    # replace with skimage?
    maxima = vigra.analysis.localMaxima(distances, marker=np.nan, allowAtBorder=True, allowPlateaus=True)
    maxima = label(np.isnan(maxima))

    hmap = distances
    hmap = (hmap.max() - hmap)
    hmap /= hmap.max()
    hmap_ws = hmap + prediction
    ws = watershed(hmap_ws, markers=maxima)

    hmap_mc = 0.8 * prediction + 0.2 * hmap
    seg = _multicut(ws, hmap_mc, beta, n_threads)
    seg = apply_size_filter(seg, min_size)
    return seg


def _merge_segmentation_3d(seg_2d, beta=0.5, min_z_extent=10):
    edges = compute_edges_from_overlap(seg_2d, verbose=False)

    uv_ids = np.array([[edge["source"], edge["target"]] for edge in edges])
    overlaps = np.array([edge["score"] for edge in edges])

    n_nodes = int(seg_2d.max() + 1)
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)

    costs = eseg.multicut.compute_edge_costs(overlaps)
    # set background weights to be maximally repulsive
    bg_edges = (uv_ids == 0).any(axis=1)
    costs[bg_edges] = -8.0

    node_labels = eseg.multicut.multicut_decomposition(graph, -1 * costs, beta=beta)

    segmentation = nifty.tools.take(node_labels, seg_2d)

    if min_z_extent is not None and min_z_extent > 0:
        props = regionprops(segmentation)
        filter_ids = []
        for prop in props:
            box = prop.bbox
            z_extent = box[3] - box[0]
            if z_extent < min_z_extent:
                filter_ids.append(prop.label)
        if filter_ids:
            segmentation[np.isin(segmentation, filter_ids)] = 0

    return segmentation


def _segment_compartments_3d(
    prediction,
    boundary_threshold=0.4,
    n_slices_exclude=5,
    min_z_extent=10,
):
    distances = distance_transform_edt(prediction < boundary_threshold).astype("float32")
    seg_2d = np.zeros(prediction.shape, dtype="uint32")

    offset = 0
    for z in range(seg_2d.shape[0]):
        if z < n_slices_exclude or z >= seg_2d.shape[0] - n_slices_exclude:
            continue
        seg_z = _segment_compartments_2d(prediction[z], distances=distances[z], run_multicut=True, min_size=500)
        seg_z[seg_z != 0] += offset
        offset = int(seg_z.max())
        seg_2d[z] = seg_z

    seg = _merge_segmentation_3d(seg_2d, min_z_extent)
    return seg


def segment_compartments(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    verbose: bool = True,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Segment synaptic compartments in an input volume.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if `model` is not provided.
        model: Pre-loaded model. Either `model_path` or `model` is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        return_predictions: Whether to return the predictions (foreground, boundaries) alongside the segmentation.
        scale: The scale factor to use for rescaling the input volume before prediction.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask
        and the predictions if return_predictions is True.
    """
    if verbose:
        print("Segmenting compartments in volume of shape", input_volume.shape)

    # Create the scaler to handle prediction with a different scaling factor.
    scaler = _Scaler(scale, verbose)
    input_volume = scaler.scale_input(input_volume)

    # Run prediction.
    pred = get_prediction(input_volume, tiling=tiling, model_path=model_path, model=model, verbose=verbose)

    # Remove channel axis if necessary.
    if pred.ndim != input_volume.ndim:
        assert pred.ndim == input_volume.ndim + 1
        assert pred.shape[0] == 1
        pred = pred[0]

    # Run the compartment segmentation.
    # We may want to expose some of the parameters here.
    t0 = time.time()
    if input_volume.ndim == 2:
        seg = _segment_compartments_2d(pred)
    else:
        seg = _segment_compartments_3d(pred)
    if verbose:
        print("Run segmentation in", time.time() - t0, "s")
    seg = scaler.rescale_output(seg, is_segmentation=True)

    if return_predictions:
        pred = scaler.rescale_output(pred, is_segmentation=False)
        return seg, pred
    return seg
