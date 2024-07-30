import multiprocessing as mp
import warnings
from concurrent import futures

import trimesh

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, marching_cubes
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
    vesicles, resolution, ids=None, derive_radius_from_distances=True, derive_distances_2d=True, min_size=500
):
    """Compute the radii for a vesicle segmentation.
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


# TODO adjust the surface for open vs. closed structures
def compute_object_morphology(object_, structure_name, resolution=None):
    verts, faces, normals, _ = marching_cubes(object_, spacing=(1.0, 1.0, 1.0) if resolution is None else resolution)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    surface = mesh.area
    if mesh.is_watertight:
        volume = np.abs(mesh.volume)
    else:
        warnings.warn("Could not compute mesh volume and setting it to NaN.")
        volume = np.nan

    morphology = pd.DataFrame({
        "structure": [structure_name],
        "volume [pixel^3]" if resolution is None else "volume [nm^3]": [volume],
        "surface [pixel^2]" if resolution is None else "surface [nm^2]": [surface],
    })
    return morphology
