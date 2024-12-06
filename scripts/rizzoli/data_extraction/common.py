import os
from glob import glob

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np

from micro_sam.inference import batched_inference
from natsort import natsorted
from skimage.measure import label
from skimage.filters import gaussian
from skimage.segmentation import watershed


def segment_vesicles_with_sam(predictor, image, vesicle_annotations):
    vesicles = label(vesicle_annotations)
    vesicle_ids, coordinates = np.unique(vesicles, return_index=True)
    vesicle_ids, coordinates = vesicle_ids[1:], coordinates[1:]

    points = np.unravel_index(coordinates, vesicles.shape)
    points = np.concatenate([points[0][:, None], points[1][:, None]], axis=1)
    points = points[:, ::-1]
    point_labels = np.ones(len(points), dtype=int)

    points = points[:, None]
    point_labels = point_labels[:, None]

    segmentation = batched_inference(predictor, image, batch_size=16, points=points, point_labels=point_labels)
    return segmentation


def read_and_crop_folder(folder):
    stack = read_volume(folder)
    membrane = read_labels(folder, stack.shape, "membrane")
    if membrane.sum() == 0:
        return None, None, None, None

    vesicles = read_labels(folder, stack.shape, ["vesicles", "labeled_vesicles"])
    active_zone = read_labels(folder, stack.shape, "active_zone")

    bb = np.where(membrane == 1)
    bb = tuple(slice(int(b.min()), int(b.max() + 1)) for b in bb)

    stack = stack[bb]
    vesicles = vesicles[bb]
    membrane = membrane[bb]
    active_zone = active_zone[bb]

    return stack, vesicles, membrane, active_zone


def read_volume(folder):
    files = natsorted(glob(os.path.join(folder, "*.tif")))
    images = []
    for ff in files:
        images.append(imageio.imread(ff))
    return np.stack(images)


def _fill_mask(labels):
    filled_labels = np.zeros_like(labels)

    center = tuple(sh // 2 for sh in labels.shape[1:])

    for z in range(labels.shape[0]):
        hmap = gaussian(labels[z], sigma=1)
        seeds = np.zeros(hmap.shape, dtype="uint16")

        seeds[center] = 1
        seeds[:, [0, -1]] = 2
        seeds[[0, -1], :] = 2

        seg = watershed(hmap, markers=seeds)
        seg = (seg == 1).astype(labels.dtype)
        filled_labels[z] = seg

    return filled_labels


def read_labels(folder, shape, annotation_names, fill_mask=False):
    annotations = {
        "active_zone": (0, 1),
        "membrane": (2, 3),
        "vesicles": (6, 7),
        "labeled_vesicles": (8, 9)
    }

    if isinstance(annotation_names, str):
        annotation_names = [annotation_names]

    indices = [annotations[name] for name in annotation_names]
    labels = np.zeros(shape, dtype="uint32")

    # Get all .txt files matching the pattern.
    files = natsorted(glob(os.path.join(folder, "*.txt")))

    # Fill the array with annotations
    for z, filename in enumerate(files):
        data = np.loadtxt(filename, delimiter=",")

        # Fill each annotation type
        for j, (x_index, y_index) in enumerate(indices):
            x, y = data[:, x_index], data[:, y_index]

            x = np.clip(x, 0, shape[1] - 1)
            y = np.clip(y, 0, shape[2] - 1)

            valid = x > 0  # Assuming x > 0 indicates valid data points
            if valid.any():
                z_vals = np.full(x[valid].shape, z, dtype=int)
                labels[z_vals, x[valid].astype(int), y[valid].astype(int)] = j + 1

    if fill_mask:
        labels = _fill_mask(labels)

    return labels


def visualize_labels(folder):
    # Set parameters
    pixel_size = 3.067  # nm
    thickness = 70 / pixel_size

    # Get all .txt files matching the pattern
    files = natsorted(glob(os.path.join(folder, "*.txt")))

    # Plotting setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Process each file
    for i, filename in enumerate(files):
        # Load data from file
        data = np.loadtxt(filename, delimiter=",")

        # Active zone
        x, y = data[:, 0], data[:, 1]
        valid = x > 0
        if valid.any():
            z = np.zeros_like(x[valid]) + (i * thickness)
            ax.scatter(x[valid], y[valid], z, c="red", marker="o", s=9)

        # Membrane
        x, y = data[:, 2], data[:, 3]
        valid = x > 0
        if valid.any():
            z = np.zeros_like(x[valid]) + (i * thickness)
            ax.scatter(x[valid], y[valid], z, c="yellow", marker="o", s=9)

        # Vesicles
        x, y = data[:, 6], data[:, 7]
        valid = x > 0
        if valid.any():
            z = np.zeros_like(x[valid]) + (i * thickness)
            ax.scatter(x[valid], y[valid], z, edgecolors="cyan", facecolors="none", marker="o", s=28)

        # Labeled vesicles
        x, y = data[:, 8], data[:, 9]
        valid = x > 0
        if valid.any():
            z = np.zeros_like(x[valid]) + (i * thickness)
            ax.scatter(x[valid], y[valid], z, c="blue", marker="o", s=28)

    # Set equal scaling and show the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.axis("equal")
    plt.show()
