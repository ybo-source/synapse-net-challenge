import napari
import napari.layers
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox

from .base_widget import BaseWidget
from .util import run_segmentation, get_model, get_model_registry, _available_devices


class SegmentationWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

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
        model = get_model(model_type, self.device)

        # Get the image data.
        image = self._get_layer_selector_data(self.image_selector_name)
        if image is None:
            show_info("Please choose an image.")
            return

        # get tile shape and halo from the viewer
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

        # TODO: Use scale derived from the image resolution.
        scale = [self.scale_param.value()]
        segmentation = run_segmentation(
            image, model=model, model_type=model_type, tiling=tiling, scale=scale
        )

        # Add the segmentation layer
        self.viewer.add_labels(segmentation, name=f"{model_type}-segmentation")
        show_info(f"Segmentation of {model_type} added to layers.")

    def _create_settings_widget(self):
        setting_values = QWidget()
        # setting_values.setToolTip(get_tooltip("embedding", "settings"))
        setting_values.setLayout(QVBoxLayout())

        # Create UI for the device.
        self.device = "auto"
        device_options = ["auto"] + _available_devices()

        self.device_dropdown, layout = self._add_choice_param("device", self.device, device_options)
        setting_values.layout().addLayout(layout)

        # Create UI for the tile shape.
        # TODO: make the tiling 3d and get the default values from 'inference'
        self.tile_x, self.tile_y = 512, 512  # defaults
        self.tile_x_param, self.tile_y_param, layout = self._add_shape_param(
            ("tile_x", "tile_y"), (self.tile_x, self.tile_y), min_val=0, max_val=2048, step=16,
            # tooltip=get_tooltip("embedding", "tiling")
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the halo.
        self.halo_x, self.halo_y = 64, 64  # defaults
        self.halo_x_param, self.halo_y_param, layout = self._add_shape_param(
            ("halo_x", "halo_y"), (self.halo_x, self.halo_y), min_val=0, max_val=512,
            # tooltip=get_tooltip("embedding", "halo")
        )
        setting_values.layout().addLayout(layout)

        self.scale_param, layout = self._add_float_param(
            "scale", 0.5, min_val=0.0, max_val=8.0,
        )
        setting_values.layout().addLayout(layout)

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings
