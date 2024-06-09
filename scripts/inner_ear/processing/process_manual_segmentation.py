import os
import warnings

from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import nifty.tools as nt
from elf.io import open_file
from skimage.transform import rescale
from tqdm import tqdm

from synaptic_reconstruction.imod import export_segmentation, export_point_annotations
from synaptic_reconstruction.file_utils import get_data_path

from parse_table import parse_table

# Files to skip because of issues in IMODMOP.
# (Currently no issues.)
SKIP_FILES = [
    # Drawn outside of the bounds. (PD)
    "/scratch-emmy/usr/nimcpape/data/moser/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/pillar/3/manuell/Emb71M1aGridB1sec1.5pil1_PD.mod",
    # Naming convention / matching
    "/scratch-emmy/usr/nimcpape/data/moser/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/pillar/3/manuell/Emb71M1aGridB1sec1.5pil1_2PDs_vesikel.mod",
    # AssertionError: {1: ' RA-V', 2: ' MP-V', 3: ' '}
    "/scratch-emmy/usr/nimcpape/data/moser/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/pillar/4/manuell/Emb71M1aGridB2sec1pil_Vesikel.mod"
]
# Non-point object
# /scratch-emmy/usr/nimcpape/data/moser/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/pillar/2/manuell


def process_labels(labels, label_names):
    if 4 in label_names and label_names[4] == " MP-V":
        label_names = {k - 2: v for k, v in label_names.items()}
        labels = {k: v - 2 for k, v in labels.items()}
    elif 4 in label_names and label_names[4] == " RA-V":
        label_names = {k - 3: v for k, v in label_names.items()}
        labels = {k: v - 3 for k, v in labels.items()}
    else:
        pass

    # Check correct mapping of the label names.
    assert "RA-V" in label_names[1], f"{label_names}"
    assert "MP-V" in label_names[2], f"{label_names}"
    if 3 in label_names:
        assert "Docked" in label_names[3], f"{label_names}"
    labels[0] = 0
    return labels, label_names


def export_vesicles(input_path, data_path, export_path):
    with open_file(data_path, "r") as f:
        shape = f["data"].shape
    try:
        segmentation, labels, label_names = export_point_annotations(
            input_path, shape,
        )
    except AssertionError:
        print("Contains non-point vesicle annotations:")
        print(input_path)
        return False

    labels, label_names = process_labels(labels, label_names)

    segmentation = nt.takeDict(labels, segmentation)

    imageio.imwrite(export_path, segmentation, compression="zlib")
    return True


def process_folder(folder, have_pd):
    data_path = get_data_path(folder)
    annotation_folders = glob(os.path.join(folder, "manuell*"))
    assert len(annotation_folders) > 0, folder

    def process_annotations(file_, structure_name):
        fname = os.path.basename(file_)
        if structure_name.lower() in fname.lower():
            export_path = str(Path(file_).with_suffix(".tif"))

            if file_ in SKIP_FILES:
                print("Skipping", file_)
                return True

            if os.path.exists(export_path):
                return True

            print("Exporting", file_)
            if structure_name == "Vesikel":
                return export_vesicles(file_, data_path, export_path)

            export_segmentation(
                imod_path=file_, mrc_path=data_path, output_path=export_path,
            )
            return True
        else:
            return False

    structure_names = ("PD", "Ribbon", "Membrane", "Vesikel") if have_pd else\
        ("Ribbon", "Membrane", "Vesikel")

    for annotation_folder in annotation_folders:
        have_structures = {
            structure_name: False for structure_name in structure_names
        }
        annotation_files = glob(os.path.join(annotation_folder, "*.mod")) +\
            glob(os.path.join(annotation_folder, "*.3dmod"))

        for file_ in annotation_files:
            for structure_name in structure_names:
                is_structure = process_annotations(file_, structure_name)
                if is_structure:
                    have_structures[structure_name] = True

        for structure_name, have_structure in have_structures.items():
            # assert have_structure, f"{structure_name} is missing in {annotation_folder}"
            if not have_structure:
                warnings.warn(f"{structure_name} is missing in {annotation_folder}")


def process_manual_segmentation(table):
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue
        assert os.path.exists(folder), folder

        annotation = row["Manuelle Annotierung"].strip().lower()
        assert annotation in ("ja", "teilweise", "nein"), annotation
        have_manual = annotation in ("ja", "teilweise")
        have_pd = row["PD vorhanden? "] == "ja"

        if have_manual:
            process_folder(folder, have_pd)

    extra_folder = os.path.join(
        "/scratch-emmy/usr/nimcpape/data/moser/Electron-Microscopy-Susi/Analyse",
        "WT strong stim/Mouse 1/modiolar/1/Tomo neues EM"
    )
    process_folder(extra_folder, True)


def _rescale(data, scale, is_seg=True):
    if is_seg:
        data = rescale(
            data, scale, order=0, preserve_range=True, anti_aliasing=False
        ).astype(data.dtype)
    else:
        data = rescale(data, scale, preserve_range=True).astype(data.dtype)
    return data


def export_manual_segmentation_for_training(table, output_folder, root):
    os.makedirs(output_folder, exist_ok=True)

    def export_segmentation(folder, ii, is_new, name):
        pd_annotations = glob(os.path.join(folder, "**", "**PD**.tif"))
        ribbon_annotations = glob(os.path.join(folder, "**", "**ribbon**.tif")) +\
            glob(os.path.join(folder, "**", "**Ribbon**.tif"))
        membrane_annotations = glob(os.path.join(folder, "**", "**membrane**.tif")) +\
            glob(os.path.join(folder, "**", "**Membrane**.tif"))
        if any([
            len(pd_annotations) == 0,
            len(ribbon_annotations) == 0,
            len(membrane_annotations) == 0,
        ]):
            warnings.warn(f"Missing annotations for {folder}, skipping!")
            return ii

        output_file = os.path.join(output_folder, f"tomogram-{ii:03}.h5")
        if os.path.exists(output_file):
            return ii + 1

        print("Exporting segmentation to", output_file)
        data_path = get_data_path(folder)
        with open_file(data_path, "r") as f:
            tomo = f["data"][:]

        pd = np.zeros(tomo.shape, dtype="uint8")
        for pd_ann in pd_annotations:
            seg = imageio.imread(pd_ann)
            pd[seg > 0] = 1

        ribbon = np.zeros(tomo.shape, dtype="uint8")
        for ribbon_ann in ribbon_annotations:
            seg = imageio.imread(ribbon_ann)
            ribbon[seg > 0] = 1

        membrane = np.zeros(tomo.shape, dtype="uint8")
        for membrane_ann in membrane_annotations:
            seg = imageio.imread(membrane_ann)
            membrane[seg > 0] = 1

        if is_new:
            scale = 1.47
            tomo = _rescale(tomo, scale=scale, is_seg=False)
            pd = _rescale(pd, scale=scale)
            membrane = _rescale(membrane, scale=scale)
            ribbon = _rescale(ribbon, scale=scale)
            assert tomo.shape == pd.shape == ribbon.shape == membrane.shape, \
                f"{tomo.shape}, {pd.shape}, {membrane.shape}, {ribbon.shape}"

        with open_file(output_file, "a") as f:
            f.create_dataset("raw", data=tomo, compression="gzip")
            f.create_dataset("labels/presynapse", data=pd, compression="gzip")
            f.create_dataset("labels/membrane", data=membrane, compression="gzip")
            f.create_dataset("labels/ribbons", data=ribbon, compression="gzip")
            f.attrs["path"] = name

        return ii + 1

    i_export = 0
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue
        assert os.path.exists(folder), folder

        annotation = row["Manuelle Annotierung"].strip().lower()
        assert annotation in ("ja", "teilweise", "nein"), annotation
        have_manual = annotation in ("ja", "teilweise")
        have_pd = row["PD vorhanden? "] == "ja"

        name = os.path.relpath(folder, root)

        is_new = row["EM alt vs. Neu"].lower() == "neu"
        if have_manual and have_pd:
            i_export = export_segmentation(folder, i_export, is_new, name)

    # export the extra annotation
    extra_folder = os.path.join(
        "/scratch-emmy/usr/nimcpape/data/moser/Electron-Microscopy-Susi/Analyse",
        "WT strong stim/Mouse 1/modiolar/1/Tomo neues EM"
    )
    name = os.path.relpath(extra_folder, root)
    export_segmentation(extra_folder, i_export, is_new=True, name=name)


def main():
    data_root = "/scratch-emmy/usr/nimcpape/data/moser"

    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)
    process_manual_segmentation(table)

    # output_folder = "/scratch-emmy/usr/nimcpape/data/moser/new-train-data"
    # export_manual_segmentation_for_training(table, output_folder, data_root)


if __name__ == "__main__":
    main()
