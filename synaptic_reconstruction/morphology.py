import warnings

import meshplex
import numpy as np
import pandas as pd
from skimage.measure import regionprops, marching_cubes, mesh_surface_area


def compute_radii(vesicles, resolution, ids=None):
    """Compute the radii for a vesicle segmentation.
    """
    props = regionprops(vesicles)
    if ids is None:
        radii = [resolution[0] * (prop.axis_minor_length + prop.axis_major_length) / 2
                 for prop in props]
    else:
        radii = [resolution[0] * (prop.axis_minor_length + prop.axis_major_length) / 2
                 for prop in props if prop.label in ids]
        assert len(radii) == len(ids)
    return radii


# TODO adjust the surface for open vs. closed structures
def compute_object_morphology(object_, structure_name, resolution=None):
    verts, faces, _, _ = marching_cubes(object_, spacing=(1.0, 1.0, 1.0) if resolution is None else resolution)

    try:
        mesh = meshplex.MeshTri(np.array(verts), np.array(faces))
        volume = np.sum(mesh.cell_volumes)
    except Exception as e:
        warnings.warn(str(e))
        volume = np.nan
    surface = mesh_surface_area(verts, faces)

    morphology = pd.DataFrame({
        "structure": [structure_name],
        "volume [pixel^3]" if resolution is None else "volume [nm^3]": [volume],
        "surface [pixel^2]" if resolution is None else "surface [nm^2]": [surface],
    })
    return morphology
