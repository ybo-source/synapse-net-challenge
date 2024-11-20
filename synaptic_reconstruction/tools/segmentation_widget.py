import napari
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox

from .base_widget import BaseWidget
from .util import (run_segmentation, get_model, get_model_registry, _available_devices, get_device, 
                   get_current_tiling, compute_scale_from_voxel_size)
from synaptic_reconstruction.inference.util import get_default_tiling
import copy


class SegmentationWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()
        self.tiling = {}

        # Create the image selection dropdown.
        self.image_selector_name = "Image data"
        self.image_selector_widget = self._create_layer_selector(self.image_selector_name, layer_type="Image")

        # Create buttons and widgets.
        self.predict_button = QPushButton("Run Segmentation")
        self.predict_button.clicked.connect(self.on_predict)
        self.model_selector_widget = self.load_model_widget()
        self.settings = self._create_settings_widget()

        # Add the widgets to the layout.
        layout.addWidget(self.image_selector_widget)
        layout.addWidget(self.model_selector_widget)
        layout.addWidget(self.settings)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)

    def load_model_widget(self):
        model_widget = QWidget()
        title_label = QLabel("Select Model:")

        models = ["- choose -"] + list(get_model_registry().urls.keys())
        self.model_selector = QComboBox()
        self.model_selector.addItems(models)
        # Create a layout and add the title label and combo box
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.model_selector)

        # Set layout on the model widget
        model_widget.setLayout(layout)
        return model_widget

    def on_predict(self):
        # Get the model and postprocessing settings.
        model_type = self.model_selector.currentText()
        if model_type == "- choose -":
            show_info("Please choose a model.")
            return

        # Load the model.
        device = get_device(self.device_dropdown.currentText())
        model = get_model(model_type, device)

        # Get the image data.
        image = self._get_layer_selector_data(self.image_selector_name)
        if image is None:
            show_info("Please choose an image.")
            return

        # load current tiling
        self.tiling = get_current_tiling(self.tiling, self.default_tiling, image.shape)
        
        # TODO: Use scale derived from the image resolution.
        # get avg image shape from training of the selected model
        # wichmann data avg voxel size = 17.53

        metadata = self._get_layer_selector_data(self.image_selector_name, return_metadata=True)
        voxel_size = metadata.get("voxel_size", None)

        if self.scale_param.value() != 1.0:  # changed from default
            scale = []
            for k in range(len(image.shape)):
                scale.append(self.scale_param.value())
        elif voxel_size:
            # calculate scale so voxel_size is the same as in training
            scale = compute_scale_from_voxel_size(voxel_size, model_type)
        else:
            scale = None
        print(f"Rescaled the image by {scale} to optimize for the selected model.")
        
        segmentation = run_segmentation(
            image, model=model, model_type=model_type, tiling=self.tiling, scale=scale
        )

        # Add the segmentation layer
        self.viewer.add_labels(segmentation, name=f"{model_type}-segmentation")
        show_info(f"Segmentation of {model_type} added to layers.")

    def _create_settings_widget(self):
        setting_values = QWidget()
        # setting_values.setToolTip(get_tooltip("embedding", "settings"))
        setting_values.setLayout(QVBoxLayout())

        # Create UI for the device.
        device = "auto"
        device_options = ["auto"] + _available_devices()

        self.device_dropdown, layout = self._add_choice_param("device", device, device_options)
        setting_values.layout().addLayout(layout)

        # Create UI for the tile shape.
        self.default_tiling = get_default_tiling()
        self.tiling = copy.deepcopy(self.default_tiling)
        self.tiling["tile"]["x"], self.tiling["tile"]["y"], self.tiling["tile"]["z"], layout = self._add_shape_param(
            ("tile_x", "tile_y", "tile_z"),
            (self.default_tiling["tile"]["x"], self.default_tiling["tile"]["y"], self.default_tiling["tile"]["z"]),
            min_val=0, max_val=2048, step=16,
            # tooltip=get_tooltip("embedding", "tiling")
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the halo.

        self.tiling["halo"]["x"], self.tiling["halo"]["y"], self.tiling["halo"]["z"], layout = self._add_shape_param(
            ("halo_x", "halo_y", "halo_z"),
            (self.default_tiling["halo"]["x"], self.default_tiling["halo"]["y"], self.default_tiling["halo"]["z"]),
            min_val=0, max_val=512,
            # tooltip=get_tooltip("embedding", "halo")
        )
        setting_values.layout().addLayout(layout)

        
        # calculate scale: read voxcel size from layer metadata
        self.viewer
        self.scale_param, layout = self._add_float_param(
            "scale", 1.0, min_val=0.0, max_val=8.0,
        )
        setting_values.layout().addLayout(layout)

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings
