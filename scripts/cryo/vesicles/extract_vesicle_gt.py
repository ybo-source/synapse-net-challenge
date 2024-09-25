import os
from pathlib import Path

import imageio.v3 as imageio
import napari
import numpy as np

from elf.io import open_file
from scipy.ndimage import distance_transform_edt, binary_closing, binary_erosion
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, find_boundaries
from tqdm import trange, tqdm


def get_bounding_box(boundaries, halo=[2, 16, 16]):
    fg_coords = np.where(boundaries != 0)
    bb = tuple(
        slice(
            max(0, int(coords.min() - ha)), min(sh, int(coords.max() + ha))
        ) for sh, ha, coords in zip(boundaries.shape, halo, fg_coords)
    )
    return bb


def close_vesicles(vesicles):
    props = regionprops(vesicles)
    for prop in tqdm(props, desc="Close Vesicles"):
        bb = tuple(
            slice(int(start), int(stop)) for start, stop in zip(prop.bbox[:3], prop.bbox[3:])
        )
        ves = vesicles[bb]
        ves = binary_closing(ves, iterations=2)
        vesicles[bb][ves] = prop.label
    return vesicles


def make_instances(boundaries, apply_closing=True):
    seg = np.zeros(boundaries.shape, dtype="int16")
    seed_map = label(boundaries)
    bg_id = int(seed_map.max()) + 1

    eps = 1e-6
    for z in trange(seg.shape[0], desc="Processing slices"):
        bd = binary_erosion(boundaries[z], iterations=1)
        props = regionprops(seed_map[z])
        seeds = np.zeros(bd.shape, dtype="int16")
        for prop in props:
            center = tuple(int(ce) for ce in prop.centroid)
            seeds[center] = prop.label

        bg_distances = distance_transform_edt(bd == 0)
        bg = bg_distances > 20
        seeds[bg] = bg_id

        fg_distances = distance_transform_edt(bd == 1)

        hmap = 1.0 - bg_distances / (bg_distances.max() + eps)
        hmap += fg_distances / (fg_distances.max() + eps)
        zseg = watershed(hmap, markers=seeds)
        zseg[zseg == bg_id] = 0
        seg[z] = zseg

    hmap = find_boundaries(boundaries)
    mask = np.logical_or(seg > 0, boundaries > 0)
    seg = watershed(hmap, markers=seg, mask=mask)

    if apply_closing:
        seg = close_vesicles(seg)
    return seg


def extract_vesicles(raw_file, seg_file, correction_file, apply_closing=True, output_path=None):
    with open_file(raw_file, "r") as f:
        raw = f["data"][:]
    with open_file(seg_file, "r") as f:
        boundaries = f["data"][:]

    bb = get_bounding_box(boundaries)
    raw = raw[bb]
    boundaries = boundaries[bb]

    if os.path.exists(correction_file):
        print("Read corrected vesilces")
        seg = imageio.imread(correction_file)
        seg = label(seg)
    else:
        print("Create vesicle segmentation from boundary annotations")
        seg = make_instances(boundaries, apply_closing=apply_closing)

    if output_path is None:
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(boundaries)
        v.add_labels(seg)
        napari.run()
    else:
        with open_file(output_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/vesicles", data=seg, compression="gzip")
            f.create_dataset("labels/vescile_boundaries", data=boundaries, compression="gzip")


def process_33K_L1(output_root):
    raw_file = "/home/pape/Work/data/fernandez-busnadiego/tomos_anotated_18924/33K/L1_ts_002_newstack_rec_deconv_bin4_260224.mrc"  # noqa
    seg_file = "/home/pape/Work/data/fernandez-busnadiego/tomos_anotated_18924/33K/L1_ts_002_SV_bin4_2622024.mrc"  # noqa
    correction_file = "vesicles-33K-L1.tif"

    fname = Path(correction_file).stem
    output_path = None if output_root is None else os.path.join(output_root, f"{fname}.h5")
    extract_vesicles(raw_file, seg_file, correction_file, output_path=output_path)


def process_64K_LAM12(output_root):
    raw_file = "/home/pape/Work/data/fernandez-busnadiego/tomos_anotated_18924/64K/Lam12_ts_006_newstack_rec_deconv_bin4_250823.mrc"  # noqa
    seg_file = "/home/pape/Work/data/fernandez-busnadiego/tomos_anotated_18924/64K/Lam12_ts_006_SV_bin4_250823.mrc"  # noqa
    correction_file = "vesicles-64K-LAM12.tif"

    fname = Path(correction_file).stem
    output_path = None if output_root is None else os.path.join(output_root, f"{fname}.h5")
    extract_vesicles(raw_file, seg_file, correction_file, apply_closing=False, output_path=output_path)


def main():
    # output_root = None
    output_root = "/home/pape/Work/data/fernandez-busnadiego/vesicle_gt/v1"

    if output_root is not None and not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    process_33K_L1(output_root)
    # process_64K_LAM12(output_root)


if __name__ == "__main__":
    main()
