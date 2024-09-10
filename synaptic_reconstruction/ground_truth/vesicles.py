import json
import os
import warnings
from pathlib import Path

import mrcfile
import numpy as np
from elf.io import open_file
from skimage.measure import label

from ..imod import export_point_annotations, export_segmentation, get_label_names


def _check_volume(raw, vesicles, labels, title=None, **extra_segmentations):
    import napari
    from nifty.tools import takeDict

    if labels is None:
        label_vol = None
    else:
        labels[0] = 0
        label_vol = takeDict(labels, vesicles)

    v = napari.Viewer()
    if raw is not None:
        v.add_image(raw)
    if vesicles is not None:
        v.add_labels(vesicles)
    if labels is not None:
        v.add_labels(label_vol)
    for name, seg in extra_segmentations.items():
        v.add_labels(seg, name=name)
    if title is not None:
        v.title = title
    napari.run()


def _export_segmentations(imod_path, data_path, object_ids):
    extra_seg = None
    for object_id in object_ids:
        seg = export_segmentation(imod_path, data_path, object_id=object_id, require_object=True)
        seg = label(seg)
        if extra_seg is None:
            extra_seg = seg
        else:
            label_offset = int(extra_seg.max())
            mask = seg != 0
            extra_seg[mask] = seg[mask] + label_offset
    return extra_seg


def write_vesicle_training_volume(
    data_path,
    imod_path,
    output_path,
    original_path=None,
    exclude_labels=None,
    exclude_label_patterns=None,
    contour_label_patterns=None,
    visualize=False,
    resolution=None,
):
    if resolution is None:
        with mrcfile.open(data_path, "r") as f:
            resolution = f.voxel_size.tolist()
        resolution = np.array([np.round(res / 10, 3) for res in resolution])
    assert len(resolution) == 3

    with open_file(data_path, "r") as f:
        vol = f["data"][:]

    vesicle_seg, labels, label_names, coords, radii = export_point_annotations(
        imod_path, vol.shape, exclude_labels=exclude_labels, exclude_label_patterns=exclude_label_patterns,
        resolution=resolution[0], return_coords_and_radii=True
    )

    if contour_label_patterns is not None:
        all_label_names, label_types = get_label_names(imod_path, return_types=True)
        mesh_object_ids = {
            obj_id: name for obj_id, name in all_label_names.items()
            if label_types[obj_id] == "closed contours" and any(pattern in name for pattern in contour_label_patterns)
        }

        # TODO double check this
        extra_seg = _export_segmentations(imod_path, data_path, mesh_object_ids)
        # extra_seg = imod_meshes_to_segmentations(imod_path, vol.shape, mesh_object_ids)
        seg_id_offset = vesicle_seg.max() + 1
        label_id_offset = max(list(labels.values())) + 1

        for i, (name, seg) in enumerate(extra_seg.items()):
            seg_id = seg_id_offset + i
            seg_mask = seg_id == 1
            if seg_mask.all():
                warnings.warn(f"All foreground mesh for {imod_path}: {name} is skipped.")
                continue
            vesicle_seg[seg_mask] = seg_id
            label_id = [i for i, pattern in enumerate(contour_label_patterns) if pattern in name]
            assert len(label_id) == 1
            labels[int(seg_id)] = int(label_id[0] + label_id_offset)

    print("Extracted the following labels:", label_names)
    print("With counts:", {k: v for k, v in zip(*np.unique(list(labels.values()), return_counts=True))})
    if visualize:
        _check_volume(vol, vesicle_seg, labels)

    with open_file(output_path, "a") as f:
        f.create_dataset("raw", data=vol, compression="gzip")

        ds = f.create_dataset("labels/vesicles", data=vesicle_seg, compression="gzip")
        ds.attrs["labels"] = json.dumps(labels)
        ds.attrs["label_names"] = json.dumps(label_names)

        f.create_dataset("labels/imod/vesicles/coordinates", data=coords)
        f.create_dataset("labels/imod/vesicles/radii", data=radii)

        if original_path is not None:
            f.attrs["filename"] = original_path


def extract_vesicle_training_data(
    data_folder,
    gt_folder,
    output_folder,
    to_label_path=None,
    skip_no_labels=False,
    exclude=None,
    exclude_labels=None,
    exclude_label_patterns=None,
    contour_label_patterns=None,
    visualize=False,
):
    os.makedirs(output_folder, exist_ok=True)

    train_id = 0
    for root, dirs, files in os.walk(data_folder):
        dirs.sort()
        files.sort()

        # check if we exclude this directory
        if exclude is not None and root in exclude:
            print("Skipping", root)
            continue

        for fname in files:
            if Path(fname).suffix not in (".mrc", ".rec"):
                continue

            output_path = os.path.join(output_folder, f"tomogram-{train_id:03}.h5")
            if os.path.exists(output_path):
                train_id += 1
                continue

            file_path = os.path.join(root, fname)
            relative_path = os.path.relpath(file_path, data_folder)

            if to_label_path is None:
                imod_path = os.path.join(gt_folder, relative_path.replace(Path(relative_path).suffix, ".imod"))
            else:
                imod_path = to_label_path(gt_folder, relative_path)

            if not os.path.exists(imod_path):
                if skip_no_labels:
                    print("Skipping", file_path, "because no matching labels were found at", imod_path)
                    train_id += 1
                    continue
                else:
                    raise RuntimeError(f"Can't find labels for {file_path} at {imod_path}.")

            print("Processing", file_path, "with target", output_path)
            write_vesicle_training_volume(
                file_path, imod_path, output_path, relative_path,
                exclude_labels=exclude_labels,
                exclude_label_patterns=exclude_label_patterns,
                contour_label_patterns=contour_label_patterns,
                visualize=visualize,
            )
            train_id += 1
