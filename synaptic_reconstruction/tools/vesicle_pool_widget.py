import os

import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .base_widget import BaseWidget
from synaptic_reconstruction.tools.util import _save_table

try:
    from napari_skimage_regionprops import add_table
except ImportError:
    add_table = None


class VesiclePoolWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.image_selector_name = "Distances"
        self.image_selector_name1 = "Segmentation"
        # # Create the image selection dropdown.
        self.image_selector_widget = self._create_layer_selector(self.image_selector_name, layer_type="Shapes")
        self.segmentation1_selector_widget = self._create_layer_selector(self.image_selector_name1, layer_type="Labels")

        # Create new layer name
        self.new_layer_name_param, new_layer_name_layout = self._add_string_param(
            name="New Layer Name",
            value="",
        )

        # pooled group name
        self.pooled_group_name_param, pooled_group_name_layout = self._add_string_param(
            name="Pooled Group Name",
            value="",
        )

        # Create query string
        self.query_param, query_layout = self._add_string_param(
            name="Query String",
            value="",
        )

        # Create advanced settings.
        self.settings = self._create_settings_widget()

        # Create and connect buttons.
        self.measure_button1 = QPushButton("Measure Vesicle Morphology")
        self.measure_button1.clicked.connect(self.on_pool_vesicles)


        # Add the widgets to the layout.
        layout.addWidget(self.image_selector_widget)
        layout.addWidget(self.segmentation1_selector_widget)
        layout.addLayout(query_layout)
        layout.addLayout(new_layer_name_layout)
        layout.addLayout(pooled_group_name_layout)
        # layout.addWidget(self.settings)
        layout.addWidget(self.measure_button1)
        # layout.addWidget(self.measure_button2)

        self.setLayout(layout)

    def _create_shapes_layer(self, name, pooling):
        print(name, pooling)
        return 

    def on_pool_vesicles(self):
        distances = self._get_layer_selector_data(self.image_selector_name, return_metadata=True)
        segmentation = self._get_layer_selector_data(self.image_selector_name1)
        morphology = self._get_layer_selector_data(self.image_selector_name1, return_metadata=True)

        if segmentation is None:
            show_info("INFO: Please choose a segmentation.")
            return
        if self.query_param.text() == "":
            show_info("INFO: Please enter a query string.")
            return
        # resolve string
        query = self.query_param.text()

        if self.new_layer_name_param.text() == "":
            show_info("INFO: Please enter a new layer name.")
            return
        new_layer_name = self.new_layer_name_param.text()
        if self.pooled_group_name_param.text() == "":
            show_info("INFO: Please enter a pooled group name.")
            return
        pooled_group_name = self.pooled_group_name_param.text()
        
        # Get distances layer
        # distance_layer_name =   # query.get("distance_layer_name", None)
        # if distance_layer_name in self.viewer.layers:
        #     distances = self._get_layer_selector_data(self.image_selector_name1, return_metadata=True)
        if distances is None:
            show_info("INFO: Distances layer could not be found or has no values.")
            return
        vesicle_pool = self._compute_vesicle_pool(segmentation, distances, morphology, new_layer_name, pooled_group_name, query)

        # # get metadata from layer if available
        # metadata = self._get_layer_selector_data(self.image_selector_name1, return_metadata=True)
        # resolution = metadata.get("voxel_size", None)
        # if resolution is not None:
        #     resolution = [v for v in resolution.values()]
        # # if user input is present override metadata
        # if self.voxel_size_param.value() != 0.0:  # changed from default
        #     resolution = segmentation.ndim * [self.voxel_size_param.value()]

    def _compute_vesicle_pool(self, segmentation, distances, morphology, new_layer_name, pooled_group_name, query):
        print(segmentation, distances, morphology, new_layer_name, pooled_group_name, query)
        vesicle_pool = {
            "segmentation": segmentation,
        }
        return vesicle_pool 

    def _create_settings_widget(self):
        setting_values = QWidget()
        setting_values.setLayout(QVBoxLayout())

        self.save_path, layout = self._add_path_param(name="Save Table", select_type="file", value="")
        setting_values.layout().addLayout(layout)

        self.voxel_size_param, layout = self._add_float_param(
            "voxel_size", 0.0, min_val=0.0, max_val=100.0,
        )
        setting_values.layout().addLayout(layout)

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings
