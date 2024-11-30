import napari
import numpy as np

import imageio.v3 as imageio
from elf.io import open_file
from skimage.filters import gaussian


def visualize_synapse():
    scale = 3
    access = np.s_[::scale, ::scale, ::scale]
    resolution = (scale * 0.868,) * 3

    tomo_path = "./data/36859_J2_66K_TS_R04_MF05_rec_2Kb1dawbp_crop.mrc"
    with open_file(tomo_path, "r") as f:
        raw = f["data"][access]
    raw = gaussian(raw)

    compartment = imageio.imread("./data/segmented/compartment.tif")
    vesicles = imageio.imread("./data/segmented/vesicles.tif")
    mitos = imageio.imread("./data/segmented/mitos.tif")
    active_zone = imageio.imread("./data/segmented/active_zone.tif")
    vesicle_ids = np.unique(vesicles)[1:]

    v = napari.Viewer()
    v.add_image(raw[:, ::-1], scale=resolution)
    v.add_labels(mitos[:, ::-1], scale=resolution)
    v.add_labels(vesicles[:, ::-1], colormap={ves_id: "orange" for ves_id in vesicle_ids}, scale=resolution)
    v.add_labels(active_zone[:, ::-1], colormap={1: "blue"}, scale=resolution)
    v.add_labels(compartment[:, ::-1], colormap={1: "red"}, scale=resolution)
    v.scale_bar.visible = True
    v.scale_bar.unit = "nm"
    v.scale_bar.font_size = 16
    napari.run()


def main():
    visualize_synapse()


main()
