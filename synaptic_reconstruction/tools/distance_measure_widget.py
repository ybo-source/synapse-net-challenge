import os

import napari
import napari.layers
import pandas as pd

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .base_widget import BaseWidget
from .. import distance_measurements

try:
    from napari_skimage_regionprops import add_table
except ImportError:
    add_table = None


def _save_distance_table(save_path, data):
    ext = os.path.splitext(save_path)[1]
    if ext == "":  # No file extension given, By default we save to CSV.
        file_path = f"{save_path}.csv"
        data.to_csv(file_path, index=False)
    elif ext == ".csv":  # Extension was specified as csv
        file_path = save_path
        data.to_csv(file_path, index=False)
    elif ext == ".xlsx":  # We also support excel.
        file_path = save_path
        data.to_excel(file_path, index=False)
    else:
        raise ValueError("Invalid extension for table: {ext}. We support .csv or .xlsx.")
    return file_path


class DistanceMeasureWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.image_selector_name1 = "Segmentation 1"
        self.image_selector_name2 = "Segmentation 2"
        # Create the image selection dropdown
        # TODO: update the names to make it easier to distinguish what is what.
        self.segmentation1_selector_widget = self._create_layer_selector(self.image_selector_name1, layer_type="Labels")
        self.segmentation2_selector_widget = self._create_layer_selector(self.image_selector_name2, layer_type="Labels")

        # create save path
        self.settings = self._create_settings_widget()

        # create buttons
        self.measure_pairwise_button = QPushButton("Measure Distance Pairwise")
        self.measure_segmentation_to_object_button = QPushButton("Measure Distance Segmentation to Object")

        # Connect buttons to functions
        self.measure_pairwise_button.clicked.connect(self.on_measure_pairwise)
        self.measure_segmentation_to_object_button.clicked.connect(self.on_measure_segmentation_to_object)
        # self.load_model_button.clicked.connect(self.on_load_model)

        # Add the widgets to the layout
        layout.addWidget(self.segmentation1_selector_widget)
        layout.addWidget(self.segmentation2_selector_widget)
        layout.addWidget(self.settings)
        # layout.addWidget(self.measure_pairwise_button)
        layout.addWidget(self.measure_segmentation_to_object_button)

        self.setLayout(layout)

    def on_measure_segmentation_to_object(self):
        segmentation1_data = self._get_layer_selector_data(self.image_selector_name1)
        segmentation2_data = self._get_layer_selector_data(self.image_selector_name2)
        if segmentation1_data is None or segmentation2_data is None:
            show_info("Please choose both segmentation layers.")
            return

        (distances,
         endpoints1,
         endpoints2,
         seg_ids) = distance_measurements.measure_segmentation_to_object_distances(
            segmentation=segmentation1_data,
            segmented_object=segmentation2_data,
            distance_type="boundary",
        )

        if self.save_path.text() != "":
            data = {"label": seg_ids, "distance": distances}
            axis_names = "zyx" if endpoints1.shape[1] == 3 else "yx"
            data.update({f"begin-{ax}": endpoints1[:, i] for i, ax in enumerate(axis_names)})
            data.update({f"end-{ax}": endpoints2[:, i] for i, ax in enumerate(axis_names)})
            data = pd.DataFrame(data)
            file_path = _save_distance_table(self.save_path.text(), data)

        lines, properties = distance_measurements.create_object_distance_lines(
            distances=distances,
            endpoints1=endpoints1,
            endpoints2=endpoints2,
            seg_ids=seg_ids
        )

        # Add the lines layer
        line_layer = self.viewer.add_shapes(
            lines,
            name="Distance Lines",
            shape_type="line",  # Specify the shape type as 'line'
            edge_width=2,
            edge_color="red",
            blending="additive",  # Use 'additive' for blending if needed
        )

        # FIXME: this doesn't work yet
        if add_table is not None:
            add_table(line_layer, self.viewer)

        if self.save_path.text() != "":
            show_info(f"Added distance lines and saved file to {file_path}.")
        else:
            show_info("Added distance lines.")

    def on_measure_pairwise(self):
        if self.image is None:
            show_info("Please choose a segmentation.")
            return
        if self.save_path.value() is None:
            show_info("Please choose a save path.")
            return
        show_info("Not implemented yet.")
        return
        # distance_measurements.measure_pairwise_object_distances(
        #     segmentation=segmentation, distance_type="boundary",
        #     save_path=self.save_path
        #     )
        # lines, properties = distance_measurements.create_distance_lines(
        #     measurement_path=self.save_path
        # )

        # # Add the lines layer
        # self.viewer.add_lines(
        #     lines, name="Distance Lines", visible=True, edge_width=2, edge_color="red", edge_blend="additive"
        # )

    def _create_settings_widget(self):
        setting_values = QWidget()
        # setting_values.setToolTip(get_tooltip("embedding", "settings"))
        setting_values.setLayout(QVBoxLayout())

        self.save_path, layout = self._add_path_param(name="Save Table", select_type="file", value="")
        setting_values.layout().addLayout(layout)

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings
