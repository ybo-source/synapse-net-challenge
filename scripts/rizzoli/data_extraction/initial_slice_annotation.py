import os
from glob import glob

import imageio.v3 as imageio
import magicgui
import napari

from common import read_and_crop_folder, segment_vesicles_with_sam
from micro_sam.util import get_sam_model


def annotate_stack(folder, output_image, output_label):

    stack, vesicles, membrane, _ = read_and_crop_folder(folder)

    z = stack.shape[0] // 2
    stack, vesicles, membrane = stack[z], vesicles[z], membrane[z]
    # compartment = label(1 - membrane, connectivity=1)
    # shape = compartment.shape
    # cid = compartment[shape[0] // 2, shape[1] // 2]
    # compartment = compartment == cid

    predictor = get_sam_model(model_type="vit_b_em_organelles")
    segmentation = segment_vesicles_with_sam(predictor, stack, vesicles)

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
