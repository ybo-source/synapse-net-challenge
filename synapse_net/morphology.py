import multiprocessing as mp
import warnings
from concurrent import futures
from typing import Dict, List, Optional, Tuple

import trimesh

import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt, convolve
from skimage.graph import MCP
from skimage.measure import regionprops, marching_cubes, find_contours
from skimage.morphology import skeletonize, medial_axis, label
from skimage.segmentation import find_boundaries


def _size_filter_ids(ids, props, min_size):
    if ids is None:
        ids = [prop.label for prop in props if prop.area > min_size]
    else:
        sizes = {prop.label: prop.area for prop in props}
        ids = [vid for vid in ids if sizes[vid] > min_size]
    return ids


def _compute_radii_naive(vesicles, resolution, ids, min_size):
    props = regionprops(vesicles)

    ids = _size_filter_ids(ids, props, min_size)
    radii = {prop.label: resolution[0] * (prop.axis_minor_length + prop.axis_major_length) / 4
             for prop in props if prop.label in ids}
    assert len(radii) == len(ids)
    return ids, radii


def _compute_radii_distances(vesicles, resolution, ids, min_size, derive_distances_2d):
    vesicle_boundaries = find_boundaries(vesicles, mode="thick")

    if derive_distances_2d:

        def dist(input_):
            return distance_transform_edt(input_ == 0, sampling=resolution[1:])

        with futures.ThreadPoolExecutor(mp.cpu_count()) as tp:
            distances = list(tp.map(dist, vesicle_boundaries))
        distances = np.stack(distances)

    else:
        distances = distance_transform_edt(vesicle_boundaries == 0, sampling=resolution)

    props = regionprops(vesicles, intensity_image=distances)
    ids = _size_filter_ids(ids, props, min_size)
    radii = {prop.label: prop.intensity_max for prop in props if prop.label in ids}
    assert len(radii) == len(ids)

    return ids, radii


def compute_radii(
    vesicles: np.ndarray,
    resolution: Tuple[float, float, float],
    ids: Optional[List[int]] = None,
    derive_radius_from_distances: bool = True,
    derive_distances_2d: bool = True,
    min_size: int = 500,
) -> Tuple[List[int], Dict[int, float]]:
    """Compute the radii for a vesicle segmentation.

    Args:
        vesicles: The vesicle segmentation.
        resolution: The pixel / voxel size of the data.
        ids: Vesicle ids to restrict the radius computation to.
        derive_radius_from_distances: Whether to derive the radii
            from the distance to the vesicle boundaries, or from the
            axis fitted to the vesicle by scikit-image regionprops.
        derive_distances_2d: Whether to derive the radii individually in 2d
            or in 3d. Deriving the radii in 3d is beneficial for anisotropic
            data or data that suffers from the missing wedge effect.
        min_size: The minimal size for extracting the radii.

    Returns:
        The ids of the extracted radii.
        The radii that were computed.
    """
    if derive_radius_from_distances:
        ids, radii = _compute_radii_distances(
            vesicles, resolution,
            ids=ids, min_size=min_size, derive_distances_2d=derive_distances_2d
        )
    else:
        assert not derive_distances_2d
        ids, radii = _compute_radii_naive(vesicles, resolution, ids=ids, min_size=min_size)
    return ids, radii


def compute_object_morphology(
    object_: np.ndarray,
    structure_name: str,
    resolution: Tuple[float, float, float] = None
) -> pd.DataFrame:
    """Compute the volume and surface area of a 2D or 3D object.

    Args:
        object_: 2D or 3D binary object array.
        structure_name: Name of the structure being analyzed.
        resolution: The pixel / voxel size of the data.

    Returns:
        Morphology information containing volume and surface area.
    """
    if object_.ndim == 2:
        # Use find_contours for 2D data
        contours = find_contours(object_, level=0.5)

        # Compute perimeter (total length of all contours)
        perimeter = sum(
            np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)).sum()
            for contour in contours
        )

        # Compute area (number of positive pixels)
        area = np.sum(object_ > 0)

        # Adjust for resolution if provided
        if resolution is not None:
            area *= resolution[0] * resolution[1]
            perimeter *= resolution[0]

        morphology = pd.DataFrame({
            "structure": [structure_name],
            "area [pixel^2]" if resolution is None else "area [nm^2]": [area],
            "perimeter [pixel]" if resolution is None else "perimeter [nm]": [perimeter],
        })

    elif object_.ndim == 3:
        # Use marching_cubes for 3D data
        verts, faces, normals, _ = marching_cubes(
            object_,
            spacing=(1.0, 1.0, 1.0) if resolution is None else resolution,
        )

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        surface = mesh.area
        if mesh.is_watertight:
            volume = np.abs(mesh.volume)
        else:
            warnings.warn("Could not compute mesh surface for the volume; setting it to NaN.")
            volume = np.nan

        morphology = pd.DataFrame({
            "structure": [structure_name],
            "volume [pixel^3]" if resolution is None else "volume [nm^3]": [volume],
            "surface [pixel^2]" if resolution is None else "surface [nm^2]": [surface],
        })

    else:
        raise ValueError("Input object must be a 2D or 3D numpy array.")

    return morphology


def _find_endpoints(component):
    # Define a 3x3 kernel to count neighbors
    kernel = np.ones((3, 3), dtype=int)
    neighbor_count = convolve(component.astype(int), kernel, mode="constant", cval=0)
    endpoints = np.argwhere((component == 1) & (neighbor_count == 2))  # Degree = 1
    return endpoints


def _compute_longest_path(component, endpoints):
    # Use the first endpoint as the source
    src = tuple(endpoints[0])
    cost = np.where(component, 1, np.inf)  # Cost map: 1 for skeleton, inf for background
    mcp = MCP(cost)
    _, traceback = mcp.find_costs([src])

    # Use the second endpoint as the destination
    dst = tuple(endpoints[-1])

    # Trace back the path
    path = np.zeros_like(component, dtype=bool)
    current = dst

    # Extract offsets from the MCP object
    offsets = np.array(mcp.offsets)
    nrows, ncols = component.shape

    while current != src:
        path[current] = True
        current_offset_index = traceback[current]
        if current_offset_index < 0:
            # No valid path found
            break
        offset = offsets[current_offset_index]
        # Move to the predecessor
        current = (current[0] - offset[0], current[1] - offset[1])
        # Ensure indices are within bounds
        if not (0 <= current[0] < nrows and 0 <= current[1] < ncols):
            break

    path[src] = True  # Include the source
    return path


def _prune_skeleton_longest_path(skeleton):
    pruned_skeleton = np.zeros_like(skeleton, dtype=bool)

    # Label connected components in the skeleton
    labeled_skeleton, num_labels = label(skeleton, return_num=True)

    for label_id in range(1, num_labels + 1):
        # Isolate the current connected component
        component = (labeled_skeleton == label_id)

        # Find the endpoints of the component
        endpoints = _find_endpoints(component)
        if len(endpoints) < 2:
            continue  # Skip if there are no valid endpoints
        elif len(endpoints) == 2:  # Nothing to prune
            pruned_skeleton |= component
            continue

        # Compute the longest path using MCP
        longest_path = _compute_longest_path(component, endpoints)
        pruned_skeleton |= longest_path

    return pruned_skeleton.astype(skeleton.dtype)


def skeletonize_object(
    segmentation: np.ndarray,
    method: str = "skeletonize",
    prune: bool = True,
    min_prune_size: int = 10,
) -> np.ndarray:
    """Skeletonize a 3D object by inidividually skeletonizing each slice.

    Args:
        segmentation: The segmented object.
        method: The method to use for skeletonization. Either 'skeletonize' or 'medial_axis'.
        prune: Whether to prune the skeleton.
        min_prune_size: The minimal size of components after pruning.

    Returns:
        The skeletonized object.
    """
    assert method in ("skeletonize", "medial_axis")
    seg_thin = np.zeros_like(segmentation)
    skeletor = skeletonize if method == "skeletonize" else medial_axis
    # Parallelize?
    for z in range(segmentation.shape[0]):
        skeleton = skeletor(segmentation[z])

        if prune:
            skeleton = _prune_skeleton_longest_path(skeleton)
            if min_prune_size > 0:
                skeleton = label(skeleton)
                ids, sizes = np.unique(skeleton, return_counts=True)
                ids, sizes = ids[1:], sizes[1:]
                skeleton = np.isin(skeleton, ids[sizes >= min_prune_size])

        seg_thin[z] = skeleton
    return seg_thin
