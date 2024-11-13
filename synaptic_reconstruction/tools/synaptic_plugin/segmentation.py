from typing import TYPE_CHECKING
import h5py
from magicgui import magic_factory, widgets
import napari
import napari.layers
from napari.utils.notifications import show_info
from napari import Viewer
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QSpinBox, QLineEdit, QGroupBox, QFormLayout, QFrame, QComboBox
from superqt import QCollapsible
from elf.io import open_file
from .base_widget import BaseWidget
import os
from synaptic_reconstruction.inference.vesicles import segment_vesicles

# Custom imports for model and prediction utilities
from ..util import get_device, get_model_registry, run_prediction, _available_devices

# if TYPE_CHECKING:
#     import napari


# def _make_collapsible(widget, title):
#     parent_widget = QWidget()
#     parent_widget.setLayout(QVBoxLayout())model_path
class SegmentationWidget(BaseWidget):
    def __init__(self):
        super().__init__()
        
        self.model = None
        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()
        
        # Create the image selection dropdown
        self.image_selector_widget = self.create_image_selector()

        # Add your buttons here
        # self.load_model_button = QPushButton('Load Model')
        self.predict_button = QPushButton('Run Prediction')
        
        # Connect buttons to functions
        self.predict_button.clicked.connect(self.on_predict)
        # self.load_model_button.clicked.connect(self.on_load_model)
        
        # create model selector
        self.model_selector_widget = self.load_model_widget()
        
        # create advanced settings
        self.settings = self._create_settings_widget()

        # Add the widgets to the layout
        layout.addWidget(self.image_selector_widget)
        layout.addWidget(self.model_selector_widget)
        layout.addWidget(self.settings)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)

    def create_image_selector(self):
        selector_widget = QWidget()
        self.image_selector = QComboBox()
        
        title_label = QLabel("Select Image Layer:")

        # Populate initial options
        self.update_image_selector()
        
        # Connect selection change to update self.image
        self.image_selector.currentIndexChanged.connect(self.update_image_data)

        # Connect to Napari layer events to update the list
        self.viewer.layers.events.inserted.connect(self.update_image_selector)
        self.viewer.layers.events.removed.connect(self.update_image_selector)

        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.image_selector)
        selector_widget.setLayout(layout)
        return selector_widget

    def update_image_selector(self, event=None):
        """Update dropdown options with current image layers in the viewer."""
        self.image_selector.clear()

        # Add each image layer's name to the dropdown
        image_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        self.image_selector.addItems(image_layers)

    def update_image_data(self):
        """Update the self.image attribute with data from the selected layer."""
        selected_layer_name = self.image_selector.currentText()
        if selected_layer_name in self.viewer.layers:
            self.image = self.viewer.layers[selected_layer_name].data
        else:
            self.image = None  # Reset if no valid selection

    def load_model_widget(self):
        model_widget = QWidget()
        title_label = QLabel("Select Model:")
    
        models = list(get_model_registry().urls.keys())
        self.model = None  # set default model
        self.model_selector = QComboBox()
        self.model_selector.addItems(models)
        # Create a layout and add the title label and combo box
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.model_selector)
        
        # Set layout on the model widget
        model_widget.setLayout(layout)
        return model_widget

    # def on_load_model(self):
    #     # Open file dialog to select a model
    #     file_dialog = QFileDialog(self)
    #     file_dialog.setFileMode(QFileDialog.ExistingFiles)
    #     file_dialog.setNameFilter("Model (*.pt)")
    #     file_dialog.setViewMode(QFileDialog.List)

    #     if file_dialog.exec_():
    #         file_paths = file_dialog.selectedFiles()
    #         if file_paths:
    #             # Assuming you load a single model path here
    #             model_path = file_paths[0]
    #             self.load_model(model_path)

    # def load_model(self, model_path):
    #     print("model path type and value", type(model_path), model_path)
    #     # Load the model from the selected path
    #     model = get_model(model_path)
    #     self.model = model

    def on_predict(self):
        # Get the model and postprocessing settings.
        model_key = self.model_selector.currentText()
        if model_key == "- choose -":
            show_info("Please choose a model.")
            return
        

        model_registry = get_model_registry()
        model_path = model_registry.fetch(model_key)

        if self.image is None:
            show_info("Please choose an image.")
            return
        
        # get tile shape and halo from the viewer
        tile_shape = (self.tile_x_param.value(), self.tile_y_param.value())
        halo = (self.halo_x_param.value(), self.halo_y_param.value())
        tiling = {
            "tile": {
                "x": self.tile_x_param.value(),
                "y": self.tile_y_param.value(),
                "z": 1
                },
            "halo": {
                "x": self.halo_x_param.value(),
                "y": self.halo_y_param.value(),
                "z": 1
                }
            }
        segmentation = segment_vesicles(self.image, model_path=model_path)  #tiling=tiling
        # Add predictions to Napari as separate layers
        # for i, pred in enumerate(segmentation):
        #     layer_name = f"Prediction {i+1}"
        #     self.viewer.add_image(pred, name=layer_name, colormap="inferno", blending="additive")
        layer_kwargs = {"colormap": "inferno", "blending": "additive"}
        return segmentation, layer_kwargs

    def _create_settings_widget(self):
        setting_values = QWidget()
        # setting_values.setToolTip(get_tooltip("embedding", "settings"))
        setting_values.setLayout(QVBoxLayout())

        # Create UI for the device.
        self.device = "auto"
        device_options = ["auto"] + _available_devices()

        self.device_dropdown, layout = self._add_choice_param("device", self.device, device_options)
                                                            #   tooltip=get_tooltip("embedding", "device"))
        setting_values.layout().addLayout(layout)

        # Create UI for the tile shape.
        self.tile_x, self.tile_y = 256, 256  # defaults
        self.tile_x_param, self.tile_y_param, layout = self._add_shape_param(
            ("tile_x", "tile_y"), (self.tile_x, self.tile_y), min_val=0, max_val=2048, step=16,
            # tooltip=get_tooltip("embedding", "tiling")
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the halo.
        self.halo_x, self.halo_y = 32, 32  # defaults
        self.halo_x_param, self.halo_y_param, layout = self._add_shape_param(
            ("halo_x", "halo_y"), (self.halo_x, self.halo_y), min_val=0, max_val=512,
            # tooltip=get_tooltip("embedding", "halo")
        )
        setting_values.layout().addLayout(layout)
        
        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings


def segmentation_widget():
    return SegmentationWidget()
