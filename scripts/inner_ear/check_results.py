import os
from glob import glob

import h5py
import napari

from elf.io import open_file


def visualize_all_data(data_root, segmentation_version=None):
    for root, dirs, files in os.walk(data_root):
        dirs.sort()

        for ff in files:
            raw_path = os.path.join(root, ff)
            if not (raw_path.endswith(".rec") or raw_path.endswith(".mrc")):
                continue

            title = os.path.relpath(raw_path, data_root)
            if segmentation_version is None:
                with open_file(raw_path, "r") as f:
                    tomo = f["data"][:]

                v = napari.Viewer()
                v.add_image(tomo)
                v.title = title
                napari.run()

            else:
                seg_folder = os.path.join(root, "automatisch", "v1")
                seg_files = glob(os.path.join(seg_folder, "*.h5"))
                if len(seg_files) == 0:
                    print("No segmentations for", title, "skipping!")

                with open_file(raw_path, "r") as f:
                    tomo = f["data"][:]

                segmentations = {}
                for seg_file in seg_files:
                    seg_name = seg_file.split("_")[-1].rstrip(".h5")
                    with h5py.File(seg_file, "r") as f:
                        seg = f["segmentation"][:] if "segmentation" in f else f["prediction"][:]
                    segmentations[seg_name] = seg

                v = napari.Viewer()
                v.add_image(tomo)
                for name, seg in segmentations.items():
                    v.add_labels(seg, name=name)
                v.title = title
                napari.run()


def main():
    data_root = "/home/pape/Work/data/moser/em-synapses/Electron-Microscopy-Susi/Analyse"
    segmentation_version = 1

    visualize_all_data(data_root, segmentation_version=segmentation_version)


# Tomos With Artifacts:
# Analyse/WT strong stim/Mouse 1/modiolar/14/Emb71M1aGridA3sec3mod14.rec
# Analyse/WT strong stim/Mouse 1/modiolar/18/Emb71M1aGridA1sec1mod3.rec
# Analyse/WT strong stim/Mouse 1/modiolar/8/Emb71M1aGridA1sec1mod2.rec
if __name__ == "__main__":
    main()
