# Evaluate and visualize segmentation keys on the blocks with manual annotations.
import os
from glob import glob

import h5py
import napari

from elf.evaluation import matching

DATA_ROOT = "./v1"
RES_ROOT = "./segmented_gt"


def evaluate_and_visualize_results(model_names, model_folders, model_keys, skip_prefixes=None):
    tomo_root = os.path.join(DATA_ROOT, "tomograms")
    label_root = os.path.join(DATA_ROOT, "labels")
    tomograms = sorted(glob(os.path.join(tomo_root, "**/*.h5"), recursive=True))

    for tomo_path in tomograms:
        if skip_prefixes is not None and tomo_path.starswith(tuple(skip_prefixes)):
            continue

        rel_path = os.path.relpath(tomo_path, tomo_root)
        print("Run evaluation for", rel_path)

        with h5py.File(tomo_path, "r") as f:
            tomo = f["raw"][:]

        label_path = os.path.join(label_root, rel_path)
        with h5py.File(label_path, "r") as f:
            annotations = f["vesicles/corrected"][:]

        predictions = {}
        for name, folder, key in zip(model_names, model_folders, model_keys):
            seg_path = os.path.join(RES_ROOT, folder, rel_path)
            with h5py.File(seg_path, "r") as f:
                seg = f[key][:]
            predictions[name] = seg

        for name, seg in predictions.items():
            scores = matching(seg, annotations, threshold=0.5)
            print(name, ":", scores)

        v = napari.Viewer()
        v.add_image(tomo)
        v.add_labels(annotations)
        for name, seg in predictions.items():
            v.add_labels(seg, name=name)
        v.title = rel_path
        napari.run()


def main():
    model_names = ["010508", "new_combined"]
    model_folders = ["010508_model", "new_model"]
    model_keys = ["/vesicles/segment_from_boundaries_indv", "/vesicles/segment_from_combined_vesicles"]

    evaluate_and_visualize_results(model_names, model_folders, model_keys)


if __name__ == "__main__":
    main()
