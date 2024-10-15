import os
from glob import glob

import h5py
import imageio.v3 as imageio
import napari
import numpy as np

from skimage.measure import label
# from skimage.morphology import remove_small_holes
from tqdm import tqdm


def process_labels(labels):
    labels = label(labels)

    min_size = 75
    ids, sizes = np.unique(labels, return_counts=True)
    filter_ids = ids[sizes < min_size]
    labels[np.isin(labels, filter_ids)] = 0

    # labels = remove_small_holes(labels, area_threshold=min_size)
    return labels


def postprocess_annotation(im_path, ann_path, output_folder, view=False):
    fname = os.path.basename(im_path)

    out_path = os.path.join(output_folder, fname.replace(".tif", ".h5"))
    if os.path.exists(out_path):
        return

    labels = imageio.imread(ann_path)

    # Skip empty labels.
    if labels.max() == 0:
        print("Skipping", im_path)
        return

    image = imageio.imread(im_path)
    labels = process_labels(labels)

    if view:
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(labels)
        napari.run()
        return

    with h5py.File(out_path, "a") as f:
        f.create_dataset("data", data=image, compression="gzip")
        f.create_dataset("labels/compartments", data=labels, compression="gzip")


def postprocess_annotations(view):
    images = sorted(glob("output/images/*.tif"))
    annotations = sorted(glob("output/annotations/*.tif"))

    output_folder = "output/postprocessed_annotations"
    os.makedirs(output_folder, exist_ok=True)
    for im, ann in tqdm(zip(images, annotations), total=len(images)):
        postprocess_annotation(im, ann, output_folder, view=view)


def main():
    postprocess_annotations(view=False)


if __name__ == "__main__":
    main()
