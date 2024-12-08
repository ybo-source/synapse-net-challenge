import napari
import pandas as pd

from synapse_net.file_utils import read_mrc
from synapse_net.sample_data import get_sample_data
from synapse_net.inference import compute_scale_from_voxel_size, get_model, run_segmentation


def segment_structures(tomogram, voxel_size):
    # Segment the synaptic vesicles. The data will automatically be resized
    # to match the average voxel size of the training data.
    model_name = "vesicles_3d"  # This is the name for the vesicle model for EM tomography.
    model = get_model(model_name)  # Load the corresponding model.
    # Compute the scale to match the tomogram voxel size to the training data.
    scale = compute_scale_from_voxel_size(voxel_size, model_name)
    vesicles = run_segmentation(tomogram, model, model_name, scale=scale)

    # Segment the active zone.
    model_name = "active_zone"
    model = get_model(model_name)
    scale = compute_scale_from_voxel_size(voxel_size, model_name)
    active_zone = run_segmentation(tomogram, model, model_name, scale=scale)

    # Segment the synaptic compartments.
    model_name = "compartments"
    model = get_model(model_name)
    scale = compute_scale_from_voxel_size(voxel_size, model_name)
    compartments = run_segmentation(tomogram, model, model_name, scale=scale)

    return {"vesicles": vesicles, "active_zone": active_zone, "compartments": compartments}


def postprocess_segmentation(segmentations):
    pass


def measure_distances(segmentations):
    pass


def assign_vesicle_pools(distances):
    pass


def visualize_results(tomogram, segmentations, vesicle_pools):
    # TODO vesicle pool visualization
    viewer = napari.Viewer()
    viewer.add_image(tomogram)
    for name, segmentation in segmentations.items():
        viewer.add_labels(segmentation, name=name)
    napari.run()


def save_analysis(segmentations, distances, vesicle_pools, save_path):
    pass


def main():
    """This script implements an example analysis pipeline with SynapseNet and applies it to a tomogram.
    Here, we analyze docked and non-attached vesicles in a sample tomogram."""

    # Load the tomogram for our sample data.
    mrc_path = get_sample_data("tem_tomo")
    tomogram, voxel_size = read_mrc(mrc_path)

    # Segment synaptic vesicles, the active zone, and the synaptic compartment.
    segmentations = segment_structures(tomogram, voxel_size)
    import h5py
    with h5py.File("seg.h5", "r") as f:
        for name, seg in segmentations.items():
            f.create_dataset(name, data=seg, compression="gzip")

    # Post-process the segmentations, to find the presynaptic terminal,
    # filter out vesicles not in the terminal, and to 'snape' the AZ to the presynaptic boundary.
    segmentations = postprocess_segmentation(segmentations)

    # Measure the distances between the AZ and vesicles.
    distances = measure_distances(segmentations)

    # Assign the vesicle pools, 'docked' and 'non-attached' vesicles, based on the distances.
    vesicle_pools = assign_vesicle_pools(distances)

    # Visualize the results.
    visualize_results(tomogram, segmentations, vesicle_pools)

    # Compute the vesicle radii and combine and save all measurements.
    save_path = "analysis_results.xlsx"
    save_analysis(segmentations, distances, vesicle_pools, save_path)


if __name__ == "__main__":
    main()
