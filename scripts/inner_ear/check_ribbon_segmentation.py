import os
import napari

from elf.io import open_file
from synaptic_reconstruction.structures import segment_ribbon


def check_ribbon_segmentation(tomo, ribbon_pred, vesicles):

    ribbon_segmentation = segment_ribbon(ribbon_pred, vesicles)

    v = napari.Viewer()
    v.add_image(tomo)
    v.add_image(ribbon_pred)
    v.add_labels(vesicles)
    v.add_labels(ribbon_segmentation)
    napari.run()


def main():
    raw_root = "/home/pape/Work/data/moser/em-susi/04_wild_type_strong_stimulation/NichtAnnotiert"
    vesicle_seg_root = "/home/pape/Work/data/moser/em-susi/results/vesicles/v1/segmentations/NichtAnnotiert"
    seg_root = "/home/pape/Work/data/moser/em-susi/results/synaptic_structures/v2/segmentations/NichtAnnotiert"

    fname = "M1aModiolar/1/Emb71M1aGridA5sec2.5mod18.rec"
    raw_path = os.path.join(raw_root, fname)

    fname = fname.replace(".rec", ".h5")
    vesicle_path = os.path.join(vesicle_seg_root, fname)
    seg_path = os.path.join(seg_root, fname)

    assert os.path.exists(vesicle_path)
    assert os.path.exists(seg_path)

    with open_file(raw_path, "r") as f:
        tomo = f["data"][:]

    with open_file(vesicle_path, "r") as f:
        vesicles = f["seg"][:]

    with open_file(seg_path, "r") as f:
        ribbon_pred = f["seg"][0]

    check_ribbon_segmentation(tomo, ribbon_pred, vesicles)


if __name__ == "__main__":
    main()
