import os
from glob import glob
from pathlib import Path

from tqdm import tqdm

from synaptic_reconstruction.imod import export_segmentation
from synaptic_reconstruction.file_utils import get_data_path


from parse_table import parse_table

# These files lead to a segmentation fault in IMODMOP.
SKIP_FILES = [
    "/scratch-emmy/usr/nimcpape/data/moser/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/pillar/1/manuell/Emb71M1aGridA3sec3pil12_Membrane.mod",
    "/scratch-emmy/usr/nimcpape/data/moser/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/pillar/3/manuell/Emb71M1aGridB1sec1.5pil1_PD.mod",
    "/scratch-emmy/usr/nimcpape/data/moser/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/pillar/8/manuell/Emb71M1bGridL4sec3pilneu_Ribbon.mod"
]


def process_folder(folder, have_pd):
    data_path = get_data_path(folder)
    annotation_folders = glob(os.path.join(folder, "manuell*"))
    assert len(annotation_folders) > 0, folder

    def process_annotations(file_, structure_name):
        fname = os.path.basename(file_)
        if structure_name.lower() in fname.lower():
            export_path = str(Path(file_).with_suffix(".tif"))
            if os.path.exists(export_path):
                return True
            if file_ in SKIP_FILES:
                print("Skipping", file_)
                return True
            print("Exporting", file_)
            export_segmentation(
                imod_path=file_,
                mrc_path=data_path,
                output_path=export_path,
            )
            return True
        else:
            return False

    structure_names = ("PD", "Ribbon", "Membrane") if have_pd else ("Ribbon", "Membrane")

    for annotation_folder in annotation_folders:
        have_structures = {structure_name: False for structure_name in structure_names}
        annotation_files = glob(os.path.join(annotation_folder, "*.mod")) +\
            glob(os.path.join(annotation_folder, "*.3dmod"))

        for file_ in annotation_files:
            for structure_name in structure_names:
                is_structure = process_annotations(file_, structure_name)
                if is_structure:
                    have_structures[structure_name] = True

        for structure_name, have_structure in have_structures.items():
            assert have_structure, f"{structure_name} is missing in {annotation_folder}"


def process_manual_segmentation(table):
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue
        assert os.path.exists(folder), folder

        have_manual = row["Manuelle Annotierung"] == "ja"
        have_pd = row["PD vorhanden? "] == "ja"

        if have_manual:
            process_folder(folder, have_pd)


# TODO
def export_manual_segmentation_for_training(data_root, output_folder):
    pass


def main():
    table_path = "./Übersicht.xlsx"
    data_root = "/scratch-emmy/usr/nimcpape/data/moser"
    table = parse_table(table_path, data_root)

    # process_manual_segmentation(table)
    output_folder = "/scratch-emmy/usr/nimcpape/data/moser/new-train-data"
    export_manual_segmentation(data_root, output_folder)


if __name__ == "__main__":
    main()
