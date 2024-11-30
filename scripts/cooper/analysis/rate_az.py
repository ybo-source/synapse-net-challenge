import os
from glob import glob

import h5py
import pandas as pd

import napari

from magicgui.widgets import PushButton, Container
from scipy.ndimage import binary_dilation, binary_closing
from tqdm import tqdm


# Create the widget
def create_widget(tab_path, ds, fname):
    if os.path.exists(tab_path):
        tab = pd.read_excel(tab_path)
    else:
        tab = None

    # Create buttons
    good_button = PushButton(label="Good")
    avg_button = PushButton(label="Avg")
    bad_button = PushButton(label="Bad")

    def _update_table(rating):
        nonlocal tab

        this_tab = pd.DataFrame(
            {"Dataset": [ds], "Tomogram": [fname], "Rating": [rating]}
        )
        if tab is None:
            tab = this_tab
        else:
            tab = pd.concat([tab, this_tab])
        tab.to_excel(tab_path, index=False)

    # Connect actions to button clicks
    good_button.clicked.connect(lambda: _update_table("Good"))
    avg_button.clicked.connect(lambda: _update_table("Average"))
    bad_button.clicked.connect(lambda: _update_table("Bad"))

    # Arrange buttons in a vertical container
    container = Container(widgets=[good_button, avg_button, bad_button])
    return container


def rate_az():
    raw_paths = sorted(glob(os.path.join("imig_data/**/*.h5"), recursive=True))
    seg_paths = sorted(glob(os.path.join("proofread_az/**/*.h5"), recursive=True))

    tab_path = "./az_quality.xlsx"
    for rp, sp in tqdm(zip(raw_paths, seg_paths), total=len(raw_paths)):
        with h5py.File(rp, "r") as f:
            raw = f["raw"][:]
        with h5py.File(sp, "r") as f:
            seg = f["az_thin_proofread"][:]

        seg_pp = binary_dilation(seg, iterations=2)
        seg_pp = binary_closing(seg_pp, iterations=2)

        ds, fname = os.path.split(rp)
        ds = os.path.basename(ds)
        fname = os.path.splitext(fname)[0]
        widget = create_widget(tab_path, ds, fname)

        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(seg, colormap={1: "green"}, opacity=1)
        v.add_labels(seg_pp)
        v.window.add_dock_widget(widget, area="right")
        napari.run()


rate_az()
