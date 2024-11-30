import os

import h5py
import napari

from magicgui import magicgui


def correct_manual_az(raw_path, seg_path):
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:]

    seg_key = "az_thin_proofread"
    with h5py.File(seg_path, "r") as f:
        seg = f[seg_key][:]

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg)

    @magicgui(call_button="save")
    def save():
        seg = v.layers["seg"].data
        with h5py.File(seg_path, "a") as f:
            f[seg_key][:] = seg

    v.window.add_dock_widget(save)

    napari.run()


def main():
    to_correct = [
        # ("Munc13DKO", "B_M13DKO_080212_CTRL4.8_crop"),
        # ("SNAP25", "A_SNAP25_12082_KO1.2_6_crop"),
        # ("SNAP25", "B_SNAP25_120812_CTRL1.3_13_crop"),
        ("SNAP25", "B_SNAP25_12082_KO1.2_6_crop")
    ]
    for ds, fname in to_correct:
        raw_path = os.path.join("imig_data", ds, f"{fname}.h5")
        seg_path = os.path.join("proofread_az", ds, f"{fname}.h5")
        assert os.path.exists(raw_path)
        assert os.path.exists(seg_path)
        correct_manual_az(raw_path, seg_path)


if __name__ == "__main__":
    main()
