import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .base_widget import BaseWidget

try:
    from napari_skimage_regionprops import add_table
except ImportError:
    add_table = None


class VesiclePoolWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.image_selector_name = "Distances to Structure"
        self.image_selector_name1 = "Vesicles Segmentation"
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
            tooltip="Enter a comma separated query string (e.g., 'radius > 15, distance > 250') "
            "Possible filters: radius, distance, area, intensity_max, intensity_mean, intensity_min, intensity_std"
        )

        # Create advanced settings.
        self.settings = self._create_settings_widget()

        # Create and connect buttons.
        self.measure_button1 = QPushButton("Create Vesicle Pool")
        self.measure_button1.clicked.connect(self.on_pool_vesicles)

        # Add the widgets to the layout.
        layout.addWidget(self.image_selector_widget)
        layout.addWidget(self.segmentation1_selector_widget)
        layout.addLayout(query_layout)
        layout.addLayout(new_layer_name_layout)
        layout.addLayout(pooled_group_name_layout)
        layout.addWidget(self.measure_button1)

        self.setLayout(layout)

    def on_pool_vesicles(self):
        distances_layer = self._get_layer_selector_layer(self.image_selector_name)
        distances = distances_layer.properties
        segmentation = self._get_layer_selector_data(self.image_selector_name1)
        morphology_layer = self._get_layer_selector_layer(self.image_selector_name1)
        morphology = morphology_layer.properties

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

        if distances is None:
            show_info("INFO: Distances layer could not be found or has no values.")
            return
        self._compute_vesicle_pool(segmentation, distances, morphology, new_layer_name, pooled_group_name, query)

    def _compute_vesicle_pool(self, segmentation, distances, morphology, new_layer_name, pooled_group_name, query):
        """
        Compute a vesicle pool based on the provided query parameters.

        Args:
            segmentation (array): Segmentation data (e.g., labeled regions).
            distances (dict): Properties from the distances layer.
            morphology (dict): Properties from the morphology layer.
            new_layer_name (str): Name for the new layer to be created.
            pooled_group_name (str): Name for the pooled group to be assigned.
            query (dict): Query parameters.
        """

        distances_ids = distances.get("id", [])
        morphology_ids = morphology.get("label_id", [])

        # Check if IDs are identical
        if set(distances_ids) != set(morphology_ids):
            show_info("ERROR: The IDs in distances and morphology are not identical.")
            return

        distances = pd.DataFrame(distances)
        morphology = pd.DataFrame(morphology)

        # Merge dataframes on the 'id' column
        merged_df = morphology.merge(distances, left_on="label_id", right_on="id", suffixes=("_morph", "_dist"))

        # Apply the query string to filter the data
        filtered_df = self._parse_query(query, merged_df)

        # Extract valid vesicle IDs
        valid_vesicle_ids = filtered_df["label_id"].tolist()

        new_layer_data = np.zeros(segmentation.shape, dtype=np.uint8)
        pool_id = 1
        layer = None

        # check if group already exists 
        if new_layer_name in self.viewer.layers:
            layer = self.viewer.layers[new_layer_name]
            if pooled_group_name not in layer.properties["pool"]:
                new_layer_data = layer.data
                pool_id = len(np.unique(layer.properties["pool"])) + 1
        # compute vesicles with new pool_id and properties
        for vesicle_id in valid_vesicle_ids:
            new_layer_data[segmentation == vesicle_id] = pool_id
        new_properties = {
            "id": valid_vesicle_ids,
            "radius": filtered_df["radius"].tolist(),
            "distance": filtered_df["distance"].tolist(),
            "pool": [pooled_group_name] * len(valid_vesicle_ids)
        }
        if new_layer_name in self.viewer.layers:
            layer = self.viewer.layers[new_layer_name]
            # override current vesicles with new pooled vesicles
            if pooled_group_name in layer.properties["pool"]:
                layer.data = new_layer_data
                layer.properties = new_properties
                show_info(f"Vesicle pool '{pooled_group_name}' overriden with {len(valid_vesicle_ids)} vesicles.")
            else:
                # add new vesicles and pool to existing layer
                current_properties = pd.DataFrame(layer.properties)
                new_properties = pd.DataFrame(new_properties)
                merged = pd.concat([current_properties, new_properties], ignore_index=True)
                layer.data = new_layer_data
                layer.properties = merged
                show_info(f"Vesicle pool '{pooled_group_name}' updated with {len(valid_vesicle_ids)} vesicles.")
        else:
            # Create a new layer in the viewer
            self.viewer.add_labels(
                new_layer_data,
                name=new_layer_name,
                properties=new_properties
            )
            show_info(
                f"Added new layer '{new_layer_name}' with {len(valid_vesicle_ids)} "
                f"vesicles in group '{pooled_group_name}'."
            )
        if add_table is not None:
            add_table(self.viewer.layers[new_layer_name], self.viewer)
        return {
                "id": valid_vesicle_ids,
                "radius": filtered_df["radius"].tolist(),
                "distance": filtered_df["distance"].tolist(),
            }

    def _parse_query(self, query, data):
        """
        Parse and apply a query string to filter data.

        Args:
            query (str): Comma-separated query string (e.g., "radius > 15, distance > 250").
            data (pd.DataFrame): DataFrame containing the data to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        filters = query.split(",")  # Split the query into individual conditions
        filters = [f.strip() for f in filters]  # Remove extra spaces
        for condition in filters:
            try:
                # Apply each condition to filter the DataFrame
                data = data.query(condition)
            except Exception as e:
                print(f"Failed to apply condition '{condition}': {e}")
                continue
        return data

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
