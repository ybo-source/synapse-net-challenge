import argparse
import os
from glob import glob

import tifffile
import napari
import numpy as np

from elf.io import open_file
from tqdm import tqdm


def _load_volume(path, key=None, crop_shape=None):

    if key is None:  # Load from tiff.

        # If we don't have a crop then just load the whole tif.
        if crop_shape is None:
            return tifffile.imread(path)

        # Otherwise only load the respective slices to avoid running out of memory.
        with tifffile.TiffFile(path) as f:
            n_slices = len(f.pages)
            z_start = max(n_slices // 2 - crop_shape[0] // 2, 0)
            z_stop = min(n_slices // 2 + crop_shape[0] // 2, n_slices)

            data = []
            for z, page in enumerate(f.pages):
                if z < z_start or z >= z_stop:
                    continue

                page_data = page.asarray()
                crop = tuple(
                    slice(max(sh // 2 - csh // 2, 0), min(sh // 2 + csh // 2, sh))
                    for sh, csh in zip(page_data.shape, crop_shape[1:])
                )
                data.append(page_data[crop])

            data = np.stack(data)

    else:  # Load from mrc

        with open_file(path, "r") as f:
            data = f[key]
            if crop_shape is not None:
                crop = tuple(
                    slice(max(sh // 2 - csh // 2, 0), min(sh // 2 + csh // 2, sh))
                    for sh, csh in zip(data.shape, crop_shape)
                )
            else:
                crop = np.s_[:]
            data = data[crop]

    return data


def _get_file_paths(input_path, ext=".mrc"):
    if not os.path.exists(input_path):
        raise Exception(f"Input path not found {input_path}")

    if os.path.isfile(input_path):
        input_files = [input_path]

    else:
        input_files = sorted(glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True))

    return input_files


def visualize_segmentation(args):

    input_paths = _get_file_paths(args.input_path)
    seg_paths = args.segmentation_path
    assert len(seg_paths) > 0

    if args.segmentation_names is None:
        segmentation_names = [f"seg-{i}" for i in range(len(seg_paths))]
    else:
        segmentation_names = args.segmentation_names
        assert len(segmentation_names) == len(seg_paths)

    segmentation_paths = {}
    for name, path in zip(segmentation_names, seg_paths):
        this_seg_paths = _get_file_paths(path, ext=".tif")
        assert len(this_seg_paths) == len(input_paths)
        segmentation_paths[name] = this_seg_paths

    for i, img_path in tqdm(enumerate(input_paths), total=len(input_paths), desc="Visualize segmentation"):
        tomogram = _load_volume(img_path, key="data", crop_shape=args.crop_shape)

        segmentations = {}
        for name, paths in segmentation_paths.items():
            seg = _load_volume(paths[i], key=None, crop_shape=args.crop_shape)
            segmentations[name] = seg

        v = napari.Viewer()
        v.add_image(tomogram)
        for name, seg in segmentations.items():
            v.add_labels(seg, name=name)
        v.title = img_path
        napari.run()


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation results in napari.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The path to the .mrc file or a folder structure with .mrc files containing the image data."
    )
    parser.add_argument(
        "--segmentation_path", "-s", required=True, nargs="+",
        help="The path(s) to the .tif file with the segmentation or a folder structure with the segmentation data."
        "You can pass multiple paths to visualize different segmentations (e.g. mito and vesicles) together."

    )
    parser.add_argument(
        "--segmentation_names", "-n", nargs="+",
        help="Display names for the segmentation layers."
    )
    parser.add_argument(
        "--crop_shape", type=int, nargs=3,
        help="Shape for extracting a central crop from the tomogram. This is necassary for laptops with limited RAM."
    )

    args = parser.parse_args()
    visualize_segmentation(args)


if __name__ == "__main__":
    main()
