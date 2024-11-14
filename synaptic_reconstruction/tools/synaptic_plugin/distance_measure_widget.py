import napari
import napari.layers
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox

from .base_widget import BaseWidget
from synaptic_reconstruction.training.supervised_training import get_2d_model

# Custom imports for model and prediction utilities
from synaptic_reconstruction import distance_measurements
from ..util import save_to_csv


class DistanceMeasureWidget(BaseWidget):
    def __init__(self):
        super().__init__()
        
        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()
        
        self.selectors = {}
        self.image_selector_name1 = "Segmentation 1"
        self.image_selector_name2 = "Segmentation 2"
        # Create the image selection dropdown
        self.segmentation1_selector_widget = self.create_image_selector(selector_name=self.image_selector_name1)
        self.segmentation2_selector_widget = self.create_image_selector(selector_name=self.image_selector_name2)
        
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
    
    def get_selected_layer_data(self, selector_name):
        """Return the data for the layer currently selected in a given selector."""
        if selector_name in self.selectors:
            selected_layer_name = self.selectors[selector_name].currentText()
            if selected_layer_name in self.viewer.layers:
                return self.viewer.layers[selected_layer_name].data
        return None  # Return None if layer not found

    def on_measure_segmentation_to_object(self):
        segmentation1_data = self.get_selected_layer_data(self.image_selector_name1)
        segmentation2_data = self.get_selected_layer_data(self.image_selector_name2)
        if segmentation1_data is None or segmentation2_data is None:
            show_info("Please choose both segmentation layers.")
            return
        # get save_path
        
        distances, endpoints1, endpoints2, seg_ids, object_ids = distance_measurements.measure_segmentation_to_object_distances(
            segmentation=segmentation1_data,
            segmented_object=segmentation2_data,
            distance_type="boundary",
            #save_path=self.save_path
        )
        if self.save_path is not None:
            file_path = save_to_csv(self.save_path, data=(distances, endpoints1, endpoints2, seg_ids, object_ids))
            show_info(f"Measurements saved to {file_path}")
        
        show_info("Not implemented yet.")
        return

    def on_measure_pairwise(self):
        if self.image is None:
            show_info("Please choose a segmentation.")
            return
        if self.save_path is None:  
            show_info("Please choose a save path.")
            return
        # get segmentation
        segmentation = self.image
        # run measurements
        show_info("Not implemented yet.")
        return 
        distance_measurements.measure_pairwise_object_distances(
            segmentation=segmentation, distance_type="boundary",
            save_path=self.save_path
            )
        lines, properties = distance_measurements.create_distance_lines(
            measurement_path=self.save_path
            )
            
        # Add the lines layer
        self.viewer.add_lines(lines, name="Distance Lines", visible=True, edge_width=2, edge_color="red", edge_blend="additive")
        # Add the segmentation layer
        # self.viewer.add_image(segmentation, name="Segmentation", colormap="inferno", blending="additive")

        # layer_kwargs = {"colormap": "inferno", "blending": "additive"}
        # return segmentation, layer_kwargs

    def create_image_selector(self, selector_name):
        attribute_dict = {}
        viewer = self.viewer
        """Create an image selector widget for a specific layer attribute."""
        selector_widget = QWidget()
        image_selector = QComboBox()
        title_label = QLabel(f"Select Layer for {selector_name}:")

        # Populate initial options
        self.update_selector(viewer, image_selector)
        
        # Connect selection change to update image data in attribute_dict
        image_selector.currentIndexChanged.connect(
            lambda: self.update_image_data(viewer, image_selector, attribute_dict, selector_name)
        )

        # Update selector on layer events
        viewer.layers.events.inserted.connect(lambda event: self.update_selector(viewer, image_selector))
        viewer.layers.events.removed.connect(lambda event: self.update_selector(viewer, image_selector))

        # Store this combo box in the selectors dictionary
        self.selectors[selector_name] = image_selector

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(image_selector)
        selector_widget.setLayout(layout)

        return selector_widget

    def update_selector(self, viewer, selector):
        """Update a single selector with the current image layers in the viewer."""
        selector.clear()
        image_layers = [layer.name for layer in viewer.layers] #if isinstance(layer, napari.layers.Image)
        selector.addItems(image_layers)

    def update_image_data(self, viewer, selector, attribute_dict, attribute_name):
        """Update the specified attribute in the attribute_dict with selected layer data."""
        selected_layer_name = selector.currentText()
        if selected_layer_name in viewer.layers:
            attribute_dict[attribute_name] = viewer.layers[selected_layer_name].data
        else:
            attribute_dict[attribute_name] = None  # Reset if no valid selection

    def _create_settings_widget(self):
        setting_values = QWidget()
        # setting_values.setToolTip(get_tooltip("embedding", "settings"))
        setting_values.setLayout(QVBoxLayout())

        self.save_path, layout = self._add_path_param(
            name="Save Directory", select_type="directory", value=None
        )
        setting_values.layout().addLayout(layout)
        
        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings

    # def create_image_selector(self):
    #     selector_widget = QWidget()
    #     self.image_selector = QComboBox()
        
    #     title_label = QLabel("Select Image Layer:")

    #     # Populate initial options
    #     self.update_image_selector()
        
    #     # Connect selection change to update self.image
    #     self.image_selector.currentIndexChanged.connect(self.update_image_data)

    #     # Connect to Napari layer events to update the list
    #     self.viewer.layers.events.inserted.connect(self.update_image_selector)
    #     self.viewer.layers.events.removed.connect(self.update_image_selector)

    #     layout = QVBoxLayout()
    #     layout.addWidget(title_label)
    #     layout.addWidget(self.image_selector)
    #     selector_widget.setLayout(layout)
    #     return selector_widget

    # def update_image_selector(self, event=None):
    #     """Update dropdown options with current image layers in the viewer."""
    #     self.image_selector.clear()

    #     # Add each image layer's name to the dropdown
    #     image_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
    #     self.image_selector.addItems(image_layers)

    # def update_image_data(self):
    #     """Update the self.image attribute with data from the selected layer."""
    #     selected_layer_name = self.image_selector.currentText()
    #     if selected_layer_name in self.viewer.layers:
    #         self.image = self.viewer.layers[selected_layer_name].data
    #     else:
    #         self.image = None  # Reset if no valid selection


def get_distance_measure_widget():
    return DistanceMeasureWidget()
