from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import affine_transform


def rotate_3d_array(arr, rotation_matrix, center, order):
    # Translate the array to center it at the origin
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -center

    # Translation back to original position
    translation_back = np.eye(4)
    translation_back[:3, 3] = center

    # Construct the full transformation matrix: Translation -> Rotation -> Translation back
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix  # Apply the PCA rotation

    # Combine the transformations: T_back * R * T_origin
    full_transformation = translation_back @ transformation_matrix @ translation_to_origin

    # Apply affine_transform (we extract the 3x3 rotation matrix and the translation vector)
    rotated_arr = affine_transform(
        arr,
        full_transformation[:3, :3],  # Rotation part
        offset=full_transformation[:3, 3],  # Translation part
        output_shape=arr.shape,  # Keep output shape the same
        order=order
    )
    return rotated_arr


# Find the rotation that aligns the data with the PCA
def _find_rotation(segmentation):
    foreground_coords = np.argwhere(segmentation > 0)

    pca = PCA(n_components=3)
    pca.fit(foreground_coords)

    rotation_matrix = pca.components_

    return rotation_matrix


def extract_and_align_foreground(
    segmentation: np.ndarray,
    raw: Optional[np.ndarray] = None,
    extract_bb: bool = True,
):
    """Extract and align the bounding box containing foreground from the segmentation.

    This function will find the closest fitting, non-axis-aligned rectangular bounding box
    that contains the segmentation foreground. It will then rotate the data, so that it is
    axis-aligned.

    Args:
        segmentation: The input segmentation.
        raw: The raw data.
        extract_bb: Whether to cout out the bounding box.

    Returns:
        TODO
    """
    rotation_matrix = _find_rotation(segmentation)

    # Calculate the center of the original array.
    center = np.array(segmentation.shape) / 2.0

    # Rotate the array.
    segmentation = rotate_3d_array(segmentation, rotation_matrix, center, order=0)

    if extract_bb:
        bb = np.where(segmentation != 0)
        bb = tuple(
            slice(int(b.min()), int(b.max()) + 1) for b in bb
        )
    else:
        bb = np.s_[:]

    if raw is not None:
        raw = rotate_3d_array(raw, rotation_matrix, center, order=1)

    if raw is not None:
        return segmentation[bb], raw[bb]

    return segmentation[bb]


if __name__ == "__main__":
    import h5py
    import napari

    segmentation_path = "tomogram-000.h5"

    with h5py.File(segmentation_path, "r") as f:
        raw = f["/raw"][:]
        segmentation = f["/labels/vesicles"][:]

    segmentation, raw = extract_and_align_foreground(segmentation, raw)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(segmentation)
    napari.run()
