import argparse

import napari
import imageio.v3 as imageio

from elf.io import open_file


def check_segmentation(tomo_path, show):
    with open_file(tomo_path, "r") as f:
        tomo = f["data"][:]
    vesicle_path = tomo_path.replace(".rec", "_vesicles.tif")
    vesicles = imageio.imread(vesicle_path)

    if show == 1:
        v = napari.Viewer()
        v.add_image(tomo)
        napari.run()
    elif show == 2:
        binary_vesicles = (vesicles > 0).astype("uint8") * 255
        v = napari.Viewer()
        v.add_image(tomo)
        v.add_image(binary_vesicles, colormap="green", blending="additive")
        napari.run()
    elif show == 3:
        v = napari.Viewer()
        v.add_image(tomo)
        v.add_labels(vesicles)
        napari.run()


def main():
    # path on laptop
    parser = argparse.ArgumentParser()
    parser.add_argument("tomo_path")
    parser.add_argument("-s", "--show", required=True, type=int)

    args = parser.parse_args()
    assert args.show in (1, 2, 3)

    check_segmentation(args.tomo_path, args.show)


if __name__ == "__main__":
    main()
