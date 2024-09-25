import os
import sys

import imageio.v3 as imageio
from elf.io import open_file
from tqdm import tqdm

sys.path.append("../processing")

from parse_table import parse_table, get_data_root  # noqa


def extract_folder(folder, output_path, rescale, show):
    # Check if this was already processed.
    group_root = "labels/imod"
    with open_file(output_path, "r") as f:
        if group_root in f:
            return

    # Export the segmentations from imod.
    output_folder = os.path.join(folder, "manuell")

    segmentation_names = {"vesicles": "Vesikel", "ribbon": "Ribbon", "PD": "PD", "membrane": "Membrane"}
    segmentation_paths = {name: os.path.join(output_folder, f"{nname}.tif")
                          for name, nname in segmentation_names.items()}

    missing_segmentations = [name for name, path in segmentation_paths.items() if not os.path.exists(path)]
    if missing_segmentations:
        print("Skipping", folder, "because of missing segmentations")
        return

    # Note: we actually don't have a tomogram with full IMOD annotations
    # that comes from the new microscope, so we don't need to implement this.
    if rescale:
        print("Skipping", folder, "because of RESCALE.")
        return

    segmentations = {}
    for name, path in segmentation_paths.items():
        segmentations[name] = imageio.imread(path)

    if show:
        import napari

        with open_file(output_path, "r") as f:
            raw = f["raw"][:]

        v = napari.Viewer()
        v.add_image(raw)
        for name, seg in segmentations.items():
            v.add_labels(seg, name=name)
        napari.run()

        return

    with open_file(output_path, "a") as f:
        for name, seg in segmentations.items():
            f.create_dataset(f"{group_root}/{name}", data=seg, compression="gzip")


def extract_structures_for_sophias_data(data_root, output_root, show=True):

    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)
    table["condition"] = table["Bedingung"] + "/" + "Mouse " + table["Maus"].astype(str) + "/" + table["Ribbon-Orientierung"]  # noqa

    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        folder_name = f"{row.condition.replace(' ', '-').replace('/', '_')}"
        output_folder = os.path.join(output_root, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        fname = f"{int(row['OwnCloud-Unterordner'])}.h5"
        output_path = os.path.join(output_folder, fname)

        # Only extract structures if we have this tomogram in the automatic structure export.
        if not os.path.exists(output_path):
            continue

        micro = row["EM alt vs. Neu"]
        if micro == "alt":
            extract_folder(folder, output_path, rescale=False, show=show)

        elif micro == "neu":
            extract_folder(folder, output_path, rescale=True, show=show)

        elif micro == "beides":
            extract_folder(folder, output_path, rescale=False, show=show)

            if output_root is not None:
                output_path = output_path[:-3] + "_new.h5"

            folder_new = os.path.join(folder, "Tomo neues EM")
            if not os.path.exists(folder_new):
                folder_new = os.path.join(folder, "neues EM")
            assert os.path.exists(folder_new), folder_new

            if not os.path.exists(output_path):
                continue

            extract_folder(folder_new, output_path, rescale=True, show=show)


def main():

    # Export for tomograms from Sophia.
    data_root = get_data_root()
    output_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/inner_ear_data"
    extract_structures_for_sophias_data(data_root, output_root, show=False)


if __name__ == "__main__":
    main()
