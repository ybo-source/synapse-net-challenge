import argparse
import os
import imageio.v3 as imageio

import mrcfile
import pandas

from elf.io import open_file

from synaptic_reconstruction.distance_measurements import (
    measure_pairwise_object_distances,
    measure_segmentation_to_object_distances,
    create_distance_lines,
    create_object_distance_lines,
)


def compute_all_distances(vesicles, ribbon, pd, boundaries, resolution, save_folder):
    # Compute the distance of the vesicles to the ribbon.
    ribbon_save = os.path.join(save_folder, "ribbon_dist.npz")
    if not os.path.exists(ribbon_save):
        measure_segmentation_to_object_distances(vesicles, ribbon, save_path=ribbon_save, resolution=resolution)

    # Compute the distance of the vesicles to the PD.
    pd_save = os.path.join(save_folder, "pd_dist.npz")
    if not os.path.exists(pd_save):
        measure_segmentation_to_object_distances(vesicles, pd, save_path=pd_save, resolution=resolution)

    # Compute the distance of the vesicle to the boundaries.
    boundary_save = os.path.join(save_folder, "boundary_dist.npz")
    if not os.path.exists(boundary_save):
        measure_segmentation_to_object_distances(vesicles, boundaries, save_path=boundary_save, resolution=resolution)

    # Compute the distance between all the vesicles.
    ves_save = os.path.join(save_folder, "ves_dist.npz")
    if not os.path.exists(ves_save):
        measure_pairwise_object_distances(vesicles, save_path=ves_save, n_threads=2, resolution=resolution)

    return ves_save, ribbon_save, pd_save, boundary_save


def visualize_distances(
    tomo, vesicles, ribbon, pd, boundaries,
    distance_path_vesicles, distance_path_ribbon, distance_path_pd, distance_path_boundaries,
    scale=2, show=True,
):
    import napari
    from synaptic_reconstruction.tools.distance_measurement import _downsample

    tomo = _downsample(tomo, scale=scale)
    vesicles = _downsample(vesicles, is_seg=True, scale=scale)
    ribbon = _downsample(ribbon, is_seg=True, scale=scale)
    pd = _downsample(pd, is_seg=True, scale=scale)
    boundaries = _downsample(boundaries, is_seg=True, scale=scale)

    # TODO compute the pairs to filter in the vesicles here
    ves_lines, ves_props = create_distance_lines(distance_path_vesicles, n_neighbors=3, scale=scale)
    ribbon_lines, ribbon_props = create_object_distance_lines(distance_path_ribbon, max_distance=50, scale=scale)
    pd_lines, pd_props = create_object_distance_lines(distance_path_pd, max_distance=50, scale=scale)
    boundary_lines, boundary_props = create_object_distance_lines(
        distance_path_boundaries, max_distance=50, scale=scale
    )

    ves_dist = pandas.DataFrame(ves_props)
    ribbon_dist = pandas.DataFrame(ribbon_props)
    pd_dist = pandas.DataFrame(pd_props)
    boundary_dist = pandas.DataFrame(boundary_props)
    if not show:
        return ves_dist, ribbon_dist, pd_dist, boundary_dist

    v = napari.Viewer()
    v.add_image(tomo)
    v.add_labels(vesicles)
    v.add_labels(ribbon)
    v.add_labels(pd, name="presynaptic-density")
    v.add_labels(boundaries)

    v.add_shapes(ves_lines, shape_type="line", name="vesicle-distances", visible=False)
    v.add_shapes(ribbon_lines, shape_type="line", name="ribbon-distances", visible=False)
    v.add_shapes(pd_lines, shape_type="line", name="presynaptic-distances", visible=False)
    v.add_shapes(boundary_lines, shape_type="line", name="boundary-distances", visible=False)

    napari.run()

    return ves_dist, ribbon_dist, pd_dist, boundary_dist


def to_excel(vesicle_distances, ribbon_distances, pd_distances, boundary_distances, result_path):
    vesicle_distances.to_excel(result_path, sheet_name="vesicle-vesicle-distances", index=False)
    with pandas.ExcelWriter(result_path, engine="openpyxl", mode="a") as writer:
        ribbon_distances.to_excel(writer, sheet_name="vesicle-ribbon-distances", index=False)
        pd_distances.to_excel(writer, sheet_name="vesicle-pd-distances", index=False)
        boundary_distances.to_excel(writer, sheet_name="vesicle-boundary-distances", index=False)


def process_distance_measurements(tomo_path, show):
    print("Processing distances for", tomo_path)

    with mrcfile.open(tomo_path, "r") as f:
        resolution = f.voxel_size.tolist()
    # bring to nanometer
    resolution = [res / 10 for res in resolution]
    print("The resolution is", resolution)

    vesicle_path = tomo_path.replace(".rec", "_vesicles.tif")
    ribbon_path = tomo_path.replace(".rec", "_ribbon.tif")
    pd_path = tomo_path.replace(".rec", "_pd.tif")
    boundary_path = tomo_path.replace(".rec", "_boundary.tif")
    assert os.path.exists(vesicle_path), vesicle_path
    assert os.path.exists(ribbon_path), ribbon_path
    assert os.path.exists(pd_path), pd_path
    assert os.path.exists(boundary_path), boundary_path

    vesicles = imageio.imread(vesicle_path)
    ribbon = imageio.imread(ribbon_path)
    pd = imageio.imread(pd_path)
    boundaries = imageio.imread(boundary_path)
    if pd.sum() == 0:
        # Empty postsynaptic density is not yet supported
        raise NotImplementedError

    distance_save_folder = os.path.join(os.path.split(tomo_path)[0], "distances")
    os.makedirs(distance_save_folder, exist_ok=True)
    ves_dist, ribbon_dist, pd_dist, boundary_dist = compute_all_distances(
        vesicles, ribbon, pd, boundaries, resolution, distance_save_folder
    )

    with open_file(tomo_path, "r") as f:
        tomo = f["data"][:]
    ves_dist, ribbon_dist, pd_dist, boundary_dist = visualize_distances(
        tomo, vesicles, ribbon, pd, boundaries, ves_dist, ribbon_dist, boundary_dist, pd_dist, show=show
    )

    out_path = os.path.join(distance_save_folder, "distances.xlsx")
    to_excel(ves_dist, ribbon_dist, pd_dist, boundary_dist, result_path=out_path)


def main():
    # path on laptop
    # tomo_path = "/home/pape/Work/data/moser/em-susi/04_wild_type_strong_stimulation/NichtAnnotiert/M1aModiolar/2/Emb71M1aGridA4sec4mod6.rec"
    parser = argparse.ArgumentParser()
    parser.add_argument("tomo_path")
    parser.add_argument("-s", "--show", action="store_true")

    args = parser.parse_args()
    process_distance_measurements(args.tomo_path, show=args.show)


if __name__ == "__main__":
    main()
