import os

import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .base_widget import BaseWidget
from synaptic_reconstruction.imod.to_imod import convert_segmentation_to_spheres
from synaptic_reconstruction.morphology import compute_object_morphology
from synaptic_reconstruction.tools.util import _save_table

try:
    from napari_skimage_regionprops import add_table
except ImportError:
    add_table = None


class MorphologyWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.image_selector_name1 = "Segmentation"
        # Create the image selection dropdown.
        self.segmentation1_selector_widget = self._create_layer_selector(self.image_selector_name1, layer_type="Labels")

        # Create advanced settings.
        self.settings = self._create_settings_widget()

        # Create and connect buttons.
        self.measure_button1 = QPushButton("Measure Vesicle Morphology")
        self.measure_button1.clicked.connect(self.on_measure_vesicle_morphology)

        self.measure_button2 = QPushButton("Measure Structure Morphology")
        self.measure_button2.clicked.connect(self.on_measure_structure_morphology)

        # Add the widgets to the layout.
        layout.addWidget(self.segmentation1_selector_widget)
        layout.addWidget(self.settings)
        layout.addWidget(self.measure_button1)
        layout.addWidget(self.measure_button2)

        self.setLayout(layout)

    def _create_shapes_layer(self, coords, radii, name="Shapes Layer"):
        """
        Create a Shapes layer with properties for IDs, coordinates, and radii.

        Args:
            coords (np.ndarray): Array of 2D or 3D coordinates.
            radii (np.ndarray): Array of radii corresponding to the coordinates.
            name (str): Name of the layer.

        Returns:
            line_layer: The created Shapes layer.
        """
        assert len(coords) == len(radii), f"Shape mismatch: {coords.shape}, {radii.shape}"
        
        # Define the shape outlines (e.g., circles or lines). 
        # For circles, the `coords` are the centers, and `radii` define the size.
        if coords.shape[1] == 2:
            # For 2D data, represent circles as lines approximating the circumference
            lines = [
                np.column_stack((
                    coords[i, 0] + radii[i] * np.cos(np.linspace(0, 2 * np.pi, 100)),
                    coords[i, 1] + radii[i] * np.sin(np.linspace(0, 2 * np.pi, 100))
                )) for i in range(len(coords))
            ]
        else:
            raise NotImplementedError("3D shapes not yet implemented.")  # Handle 3D if needed

        # Properties for the table
        properties = {
            "index": np.arange(len(coords)),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "radii": radii,
        }

        # Add the shapes layer
        line_layer = self.viewer.add_shapes(
            lines,
            name=name,
            shape_type="polygon",  # Use "polygon" for closed shapes like circles
            edge_width=2,
            edge_color="red",
            face_color="transparent",
            blending="additive",
            properties=properties,  # Attach the properties here
        )
        return line_layer

    def _to_table_data(self, coords, radii):
        assert len(coords) == len(radii), f"Shape mismatch: {coords.shape}, {radii.shape}"
        # Handle both 2D and 3D coordinates
        if coords.ndim == 2:
            col_names = ['x', 'y'] if coords.shape[1] == 2 else ['x', 'y', 'z']
            table_data = {
                'index': np.arange(len(coords)),
                **{col: coords[:, i] for i, col in enumerate(col_names)},
                'radii': radii
            }
        else:
            # Fallback for 1D label and radii
            table_data = {"label": coords, "distance": radii}
        
        return pd.DataFrame(table_data)

    def _add_table(self, coords, radii, name: str):
        layer = self._create_shapes_layer(coords, radii)

        # Add a table layer to the Napari viewer
        if add_table is not None:
            add_table(layer, self.viewer)

        # Save table to file if save path is provided
        if self.save_path.text() != "":
            file_path = _save_table(self.save_path.text(), self._to_table_data(coords, radii))
            show_info(f"INFO: Added table and saved file to {file_path}.")
        else:
            show_info("INFO: Table added to viewer.")

    def on_measure_vesicle_morphology(self):
        segmentation = self._get_layer_selector_data(self.image_selector_name1)
        if segmentation is None:
            show_info("INFO: Please choose a segmentation.")
            return

        # get metadata from layer if available
        metadata = self._get_layer_selector_data(self.image_selector_name1, return_metadata=True)
        resolution = metadata.get("voxel_size", None)
        if resolution is not None:
            resolution = [v for v in resolution.values()]
        # if user input is present override metadata
        if self.voxel_size_param.value() != 0.0:  # changed from default
            resolution = segmentation.ndim * [self.voxel_size_param.value()]

        coords, radii = convert_segmentation_to_spheres(
            segmentation=segmentation,
            resolution=resolution
        )
        print("coords", coords.shape, "radii", radii.shape)
        # table_data = self._to_table_data(coords, radii)
        self._add_table(coords, radii, name="Vesicles")

    def on_measure_structure_morphology(self):
        return None
        # segmentation = self._get_layer_selector_data(self.image_selector_name1)
        # if segmentation is None:
        #     show_info("Please choose a segmentation.")
        #     return
        # # get metadata from layer if available
        # metadata = self._get_layer_selector_data(self.image_selector_name1, return_metadata=True)
        # resolution = metadata.get("voxel_size", None)
        # if resolution is not None:
        #     resolution = [v for v in resolution.values()]
        # # if user input is present override metadata
        # if self.voxel_size_param.value() != 0.0:  # changed from default
        #     resolution = segmentation.ndim * [self.voxel_size_param.value()]

        # (distances,
        #  endpoints1,
        #  endpoints2,
        #  seg_ids) = distance_measurements.measure_pairwise_object_distances(
        #     segmentation=segmentation, distance_type="boundary", resolution=resolution
        # )
        # lines, properties = distance_measurements.create_pairwise_distance_lines(
        #     distances=distances, endpoints1=endpoints1, endpoints2=endpoints2, seg_ids=seg_ids.tolist(),
        # )
        # table_data = self._to_table_data(
        #     distances=properties["distance"],
        #     seg_ids=np.concatenate([properties["id_a"][:, None], properties["id_b"][:, None]], axis=1)
        # )
        # self._add_lines_and_table(lines, properties, table_data, name="pairwise-distances")

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
