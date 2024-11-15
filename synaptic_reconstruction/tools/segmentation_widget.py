import napari
import napari.layers
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox

from .base_widget import BaseWidget
from .util import run_segmentation, get_model_registry, _available_devices


class SegmentationWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.model = None
        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        # Create the image selection dropdown
        self.image_selector_name = "Image data"
        self.image_selector_widget = self._create_layer_selector(self.image_selector_name, layer_type="Image")

        # create buttons
        self.predict_button = QPushButton("Run Segmentation")

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

    def on_predict(self):
        # Get the model and postprocessing settings.
        model_key = self.model_selector.currentText()
        if model_key == "- choose -":
            show_info("Please choose a model.")
            return

        # loading model
        model_registry = get_model_registry()
        model_key = self.model_selector.currentText()
        # model_path = "/home/freckmann15/.cache/synapse-net/models/vesicles"  #
        model_path = model_registry.fetch(model_key)
        # get image data
        image = self._get_layer_selector_data(self.image_selector_name)
        if image is None:
            show_info("Please choose an image.")
            return

        # FIXME: don't hard-code tiling here, but figure it out centrally in the prediction function.
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
        tile_shape = (self.tile_x_param.value(), self.tile_y_param.value())
        halo = (self.halo_x_param.value(), self.halo_y_param.value())
        use_custom_tiling = False
        for ts, h in zip(tile_shape, halo):
            if ts != 0 or h != 0:  # if anything changed from default
                use_custom_tiling = True
        if use_custom_tiling:
            segmentation = run_segmentation(
                image, model_path=model_path, model_key=model_key,
                tiling=tiling, scale=self.scale_param.value()
                )
        else:
            segmentation = run_segmentation(
                image, model_path=model_path, model_key=model_key,
                scale=self.scale_param.value()
                )

        # Add the segmentation layer
        self.viewer.add_labels(segmentation, name=f"{model_key}-segmentation")
        show_info(f"Segmentation of {model_key} added to layers.")

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
