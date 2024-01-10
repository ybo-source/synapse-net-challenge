# TODO:
# - how do we save the results for further analysis?

import argparse
import os
import multiprocessing as mp
import warnings

import imageio.v3 as imageio
import napari

from elf.io import open_file
from napari_skimage_regionprops import add_table

from skimage.transform import rescale

from ..distance_measurements import create_distance_lines, measure_pairwise_object_distances


def _downsample(data, scale, is_seg=False):
    rescale_factor = 1.0 / scale
    if is_seg:
        data = rescale(data, rescale_factor, order=0, anti_aliasing=False, preserve_range=True).astype(data.dtype)
    else:
        data = rescale(data, rescale_factor, preserve_range=True).astype(data.dtype)
    return data


# TODO add widget to update the number of neighbors for the lines
# TODO take pixel size as argument for the distance measurement
def measure_distances(tomogram, segmentation, distance_measurement_path, n_neighbors=3, bb=None, scale=2):

    if not os.path.exists(distance_measurement_path):
        warnings.warn(
            "Could not find measurement result at {distance_measurement_path}."
            "Will compute the distance measurement result, this will take some time!"
        )
        resolution = None  # TODO expose as parameter
        cpu_count = mp.cpu_count()
        measure_pairwise_object_distances(
            segmentation, "boundary", n_threads=cpu_count, resolution=resolution, save_path=distance_measurement_path,
        )

    lines, properties = create_distance_lines(
        distance_measurement_path, n_neighbors=n_neighbors, bb=bb, scale=scale
    )

    if scale > 1:
        tomogram = _downsample(tomogram, scale=scale)
        segmentation = _downsample(segmentation, scale=scale, is_seg=True)
    assert tomogram.shape == segmentation.shape, f"{tomogram.shape}, {segmentation.shape}"

    viewer = napari.Viewer()
    viewer.add_image(tomogram, visible=False)
    viewer.add_labels(segmentation)
    line_layer = viewer.add_shapes(lines,  shape_type="line", properties=properties, name="distances")
    add_table(line_layer, viewer)
    napari.run()


def main():
    parser = argparse.ArgumentParser("Measure and visualize distances.")
    parser.add_argument("-i", "--image_path", help="Filepath for the tomogram", required=True)
    parser.add_argument("-s", "--segmentation_path", help="Filepath for the segmentation", required=True)
    parser.add_argument("-m", "--measurement_path", help="Filepath for the distance measurement result", required=True)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--scale", type=int, default=2)

    args = parser.parse_args()
    with open_file(args.image_path, "r") as f:
        tomogram = f["data"][:]
    segmentation = imageio.imread(args.segmentation_path)

    measure_distances(
        tomogram, segmentation, args.measurement_path,
        n_neighbors=args.neighbors, scale=args.scale,
    )
