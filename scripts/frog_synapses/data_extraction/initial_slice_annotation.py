import os
from glob import glob

import imageio.v3 as imageio
import magicgui
import napari
import numpy as np

from common import read_volume, read_labels
from micro_sam.util import get_sam_model
from micro_sam.inference import batched_inference

from skimage.measure import label


def annotate_stack(folder, output_image, output_label):

    stack = read_volume(folder)
    vesicles = read_labels(folder, stack.shape, ["vesicles", "labeled_vesicles"])
    membrane = read_labels(folder, stack.shape, "membrane")
    if membrane.sum() == 0:
        return

    bb = np.where(membrane == 1)
    bb = tuple(
        slice(int(b.min()), int(b.max() + 1)) for b in bb
    )

    stack = stack[bb]
    vesicles = vesicles[bb]
    membrane = membrane[bb]

    z = stack.shape[0] // 2
    stack, vesicles, membrane = stack[z], vesicles[z], membrane[z]
    # compartment = label(1 - membrane, connectivity=1)
    # shape = compartment.shape
    # cid = compartment[shape[0] // 2, shape[1] // 2]
    # compartment = compartment == cid

    vesicles = label(vesicles)
    vesicle_ids, coordinates = np.unique(vesicles, return_index=True)
    vesicle_ids, coordinates = vesicle_ids[1:], coordinates[1:]

    points = np.unravel_index(coordinates, vesicles.shape)
    points = np.concatenate([points[0][:, None], points[1][:, None]], axis=1)
    points = points[:, ::-1]
    point_labels = np.ones(len(points), dtype=int)

    points = points[:, None]
    point_labels = point_labels[:, None]

    predictor = get_sam_model(model_type="vit_b_em_organelles")
    segmentation = batched_inference(predictor, stack, batch_size=16, points=points, point_labels=point_labels)

    @magicgui.magicgui
    def save(v: napari.Viewer):
        seg = v.layers["segmentation"].data
        imageio.imwrite(output_image, stack, compression="zlib")
        imageio.imwrite(output_label, seg, compression="zlib")

    v = napari.Viewer()
    v.add_image(stack)
    v.add_labels(segmentation)

    v.window.add_dock_widget(save)
    v.title = output_image

    napari.run()


def annotate_folder(folder, output_images, output_labels):
    skip = [
        "block10U3A_four.tif",
        "block10U3A_three.tif",
        "block10U3A_two.tif",
        "block30UB_four.tif",
    ]

    sub_folders = sorted(os.listdir(folder))
    for subf in sub_folders:
        fname = f"{os.path.basename(folder)}_{subf}.tif"
        if fname in skip:
            continue

        out_image = os.path.join(output_images, fname)
        out_label = os.path.join(output_labels, fname)

        if os.path.exists(out_image):
            assert os.path.exists(out_label)
            continue

        print("Annotating", out_image)

        annotate_stack(os.path.join(folder, subf), out_image, out_label)


def run_initial_annotations(output_images, output_labels):
    folders = [
        "/home/pape/Work/data/silvio/frog-em/block10U3A",
        "/home/pape/Work/data/silvio/frog-em/block184B",
        "/home/pape/Work/data/silvio/frog-em/block30UB",
    ]

    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    for folder in folders:
        annotate_folder(folder, output_images, output_labels)


def check_initial_annotations(output_images, output_labels):
    images = sorted(glob(os.path.join(output_images, "*.tif")))
    labels = sorted(glob(os.path.join(output_labels, "*.tif")))

    for im, lab in zip(images, labels):
        title = im
        im = imageio.imread(im)
        lab = imageio.imread(lab)

        v = napari.Viewer()
        v.add_image(im)
        v.add_labels(lab)
        v.title = title
        napari.run()


def main():
    output_images = "./initial_annotations/images"
    output_labels = "./initial_annotations/labels"

    run_initial_annotations(output_images, output_labels)

    print("All annotations done!")
    print("Check annotations.")
    check_initial_annotations(output_images, output_labels)


if __name__ == "__main__":
    main()
