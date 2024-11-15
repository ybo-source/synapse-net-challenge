import napari
import napari.layers
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox

from .base_widget import BaseWidget

# Custom imports for model and prediction utilities
from synaptic_reconstruction import distance_measurements
from ..util import save_to_csv


class DistanceMeasureWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.image_selector_name1 = "Segmentation 1"
        self.image_selector_name2 = "Segmentation 2"
        # Create the image selection dropdown
        self.segmentation1_selector_widget = self._create_layer_selector(self.image_selector_name1, layer_type="Labels")
        self.segmentation2_selector_widget = self._create_layer_selector(self.image_selector_name2, layer_type="Labels")

        # create save path
        self.settings = self._create_settings_widget()

        # create buttons
        self.measure_pairwise_button = QPushButton('Measure Distance Pairwise')
        self.measure_segmentation_to_object_button = QPushButton('Measure Distance Segmentation to Object')

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
        # get save_path

        (distances,
         endpoints1,
         endpoints2,
         seg_ids) = distance_measurements.measure_segmentation_to_object_distances(
            segmentation=segmentation1_data,
            segmented_object=segmentation2_data,
            distance_type="boundary",
            # save_path=self.save_path
        )
        if self.save_path.text() != "":
            # save to csv
            header = "distances endpoints1 endpoints2 seg_ids"
            header_list = header.split(" ")
            file_path = save_to_csv(
                self.save_path.text(),
                data=(distances, endpoints1, endpoints2, seg_ids),
                header=header_list
                )
            show_info(f"Measurements saved to {file_path}")
        lines, properties = distance_measurements.create_object_distance_lines(
            distances=distances,
            endpoints1=endpoints1,
            endpoints2=endpoints2,
            seg_ids=seg_ids
        )

        # Add the lines layer
        self.viewer.add_shapes(
            lines,
            name="Distance Lines",
            shape_type="line",  # Specify the shape type as 'line'
            edge_width=2,
            edge_color="red",
            blending="additive",  # Use 'additive' for blending if needed
        )
        if self.save_path.text() != "":
            show_info(f"Added distance lines and saved file to {file_path}.")
        else:
            show_info("Added distance lines.")
        return

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

        self.save_path, layout = self._add_path_param(
            name="Save Directory", select_type="directory", value=""
        )
        setting_values.layout().addLayout(layout)

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings


def get_distance_measure_widget():
    return DistanceMeasureWidget()
