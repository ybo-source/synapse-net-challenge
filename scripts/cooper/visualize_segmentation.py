import os
import argparse
import imageio.v3 as iio
import napari
from elf.io import open_file
from glob import glob
from tqdm import tqdm


def _get_file_paths(input_path, ext=".mrc"):
    if not os.path.exists(input_path):
        raise Exception(f"Input path not found {input_path}")

    if os.path.isfile(input_path):
        input_files = [input_path]

    else:
        input_files = sorted(glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True))

    return input_files


def _visualize(img, seg, seg2=None):
    v = napari.Viewer()
    if img is not None:
        v.add_image(img)
    if seg is not None:
        v.add_labels(seg)
    if seg2 is not None:
        v.add_labels(seg2)
    napari.run()


def visualize_segmentation(args):
    img = None
    seg = None
    seg2 = None
    
    image_paths = _get_file_paths(args.image_path)
    seg1_paths = _get_file_paths(args.segmentation_path)
    if not args.second_segmentation_path == "":
        seg2_paths = _get_file_paths(args.second_segmentation_path)
        if len(image_paths) != len(seg1_paths) or len(image_paths) != len(seg2_paths):
            raise Exception(f"Lengths of the paths do not match: img_path: {len(image_paths)} seg_path: {len(seg1_paths)} seg2_path: {len(seg2_paths)}")
    else:
        assert len(image_paths) == len(seg1_paths)
        seg2_paths = None

    if seg2_paths is None:
        for img_path, seg1_path in tqdm(zip(image_paths, seg1_paths)):
            with open_file(img_path, "r") as f:
                img = f["data"][:]
            with iio.imopen(seg1_path, "r") as f:
                seg = f.read()
            _visualize(img, seg)
    else:
        for img_path, seg1_path, seg2_path in tqdm(zip(image_paths, seg1_paths, seg2_paths)):
            with open_file(img_path, "r") as f:
                img = f["data"][:]
            with iio.imopen(seg1_path, "r") as f:
                seg = f.read()
            with iio.imopen(seg2_path, "r") as f:
                seg2 = f.read()
            _visualize(img, seg, seg2)


def main():
    parser = argparse.ArgumentParser(description="Segment mitochodria")
    parser.add_argument(
        "--image_path", "-i", default="", required=True,
        help="The path to the .mrc file containing the image/raw data."
    )
    parser.add_argument(
        "--segmentation_path", "-s", default="", required=True,
        help="The path to the .tif file containing the segmentation data. e.g. mitochondria"
    )
    parser.add_argument(
        "--second_segmentation_path", "-ss", default="",
        help="A second path to the .tif file containing the segmentation data. e.g. cristae"
    )

    args = parser.parse_args()

    visualize_segmentation(args)


if __name__ == "__main__":
    main()
