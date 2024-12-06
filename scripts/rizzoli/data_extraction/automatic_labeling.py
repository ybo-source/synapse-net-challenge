
from micro_sam.util import get_sam_model

from common import read_and_crop_folder, segment_vesicles_with_sam


def check_slice():
    import napari

    folder = "/home/pape/Work/data/silvio/frog-em/block10U3A/five"
    stack, vesicles, _, _ = read_and_crop_folder(folder)

    predictor = get_sam_model(model_type="vit_b", checkpoint_path="./initial_annotations/best.pt")

    z = stack.shape[0] // 2 - 1
    image, vesicle_annotations = stack[z], vesicles[z]

    # TODO pull in surrounding points as negatives
    segmentation = segment_vesicles_with_sam(predictor, image, vesicle_annotations)

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation)
    napari.run()


def main():
    check_slice()


if __name__ == "__main__":
    main()
