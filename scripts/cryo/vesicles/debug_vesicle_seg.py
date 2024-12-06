import h5py
import napari

from synaptic_reconstruction.inference.vesicles import distance_based_vesicle_segmentation


def debug_vesicle_seg(path, pred_path):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]

    with h5py.File(pred_path, "r") as f:
        seg = f["/vesicles/segment_from_DA_cryo_v2_masked"][:]
        fg = f["/prediction_DA_cryo_v2_masked/foreground"][:]
        bd = f["/prediction_DA_cryo_v2_masked/boundaries"][:]

    vesicles = distance_based_vesicle_segmentation(
        fg, bd, verbose=True, min_size=500, distance_threshold=4,
    )

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(fg, visible=False)
    v.add_image(bd, visible=False)
    v.add_labels(seg)
    v.add_labels(vesicles)
    napari.run()


def main():
    path = "/home/pape/Work/data/fernandez-busnadiego/vesicle_gt/v3/vesicles-33K-L1.h5"
    pred_path = "./prediction/vesicles-33K-L1.h5"
    debug_vesicle_seg(path, pred_path)


main()
