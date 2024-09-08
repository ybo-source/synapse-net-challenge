import h5py
import numpy as np

from synaptic_reconstruction.inference.vesicles import segment_vesicles

MODEL_PATH = "/scratch-grete/projects/nim00007/data/synaptic_reconstruction/models/cooper/vesicles/3D-UNet-for-Vesicle-Segmentation-vesicles-010508model_v1r45_0105mr45_0105mr45.zip"  # noqa


def extract_gt_bounding_box(raw, vesicle_gt, halo=[2, 32, 32]):
    bb = np.where(vesicle_gt > 0)
    bb = tuple(slice(
        max(int(b.min() - ha), 0),
        min(int(b.max()) + ha, sh)
    ) for b, sh, ha in zip(bb, raw.shape, halo))
    raw, vesicle_gt = raw[bb], vesicle_gt[bb]
    return raw, vesicle_gt


# TODO
def postprocess_vesicle_shape(vesicle_gt, prediction):
    return vesicle_gt


# TODO
def find_additional_vesicles(vesicle_gt, segmentation):
    additional_vesicles = ""
    return additional_vesicles


def postprocess_vesicle_gt(raw, vesicle_gt):
    """Run post-processing for the vesicle ground-truth extracted from IMOD.
    This includes the following steps:
    - ...
    """
    assert raw.shape == vesicle_gt.shape

    # Extract the bounding box of the data that contains the vesicles in the GT.
    raw, vesicle_gt = extract_gt_bounding_box(raw, vesicle_gt)

    # Get the model predictions and segmentation for this data.
    segmentation, prediction = segment_vesicles(raw, MODEL_PATH, return_predictions=True)
    # Just for debugging.
    with h5py.File("pred.h5", "a") as f:
        f.create_dataset("segmentation", data=segmentation, compression="gzip")
        f.create_dataset("prediction", data=prediction, compression="gzip")

    # Additional post-processing to improve the shape of the vesicles.
    vesicle_gt = postprocess_vesicle_shape(vesicle_gt, prediction)

    # Get vesicles in the prediction that are not part of the ground-truth.
    additional_vesicles = find_additional_vesicles(vesicle_gt, segmentation)

    return vesicle_gt, additional_vesicles


def main():
    # This is just a random tomogram for testing.
    path = "./tomogram-000.h5"
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        vesicle_gt = f["/labels/vesicles"][:]

    vesicle_gt, additional_vesicles = postprocess_vesicle_gt(raw, vesicle_gt)


if __name__ == "__main__":
    main()
