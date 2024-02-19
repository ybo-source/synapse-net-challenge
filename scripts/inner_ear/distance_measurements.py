import argparse
import os
import imageio.v3 as imageio

import mrcfile
import numpy as np
import pandas

from elf.io import open_file
from skimage.measure import regionprops

from synaptic_reconstruction.distance_measurements import (
    measure_segmentation_to_object_distances,
    create_object_distance_lines,
    filter_blocked_segmentation_to_object_distances,
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
        measure_segmentation_to_object_distances(
            vesicles, boundaries, save_path=boundary_save, resolution=resolution
        )

    # NOTE: we do not need the inter-vesicle distances
    # # Compute the distance between all the vesicles.
    # ves_save = os.path.join(save_folder, "ves_dist.npz")
    # if not os.path.exists(ves_save):
    #     measure_pairwise_object_distances(vesicles, save_path=ves_save, n_threads=2, resolution=resolution)

    # return ves_save, ribbon_save, pd_save, boundary_save
    return ribbon_save, pd_save, boundary_save


def _get_data(tomo_path):
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
    return vesicles, ribbon, pd, boundaries


def precompute_all_distances(data_root):
    for root, dirs, files in os.walk(data_root):
        dirs.sort()

        for ff in files:
            path = os.path.join(root, ff)
            if not path.endswith(".rec"):
                continue
            with mrcfile.open(path, "r") as f:
                resolution = f.voxel_size.tolist()
            vesicles, ribbon, pd, boundaries = _get_data(path)
            save_folder = os.path.join(root, "distances")
            os.makedirs(save_folder, exist_ok=True)
            compute_all_distances(vesicles, ribbon, pd, boundaries, resolution, save_folder)


def assign_vesicles_to_pools(
    vesicles,
    distance_path_ribbon,
    distance_path_pd,
    distance_path_boundaries,
    scale=None
):

    def load_dist(measurement_path, seg_ids=None):
        auto_dists = np.load(measurement_path)
        distances, this_seg_ids = auto_dists["distances"], auto_dists["seg_ids"]
        if seg_ids is not None:
            assert np.array_equal(seg_ids, this_seg_ids)
        return distances, this_seg_ids

    ribbon_distances, seg_ids = load_dist(distance_path_ribbon)
    pd_distances, _ = load_dist(distance_path_pd, seg_ids=seg_ids)
    bd_distances, _ = load_dist(distance_path_boundaries, seg_ids=seg_ids)

    # Find the vesicles that are ribbon associated (RA-V).
    # Criterion: vesicles are closer than 80 nm to ribbon and they are in the first row
    # (i.e. not blocked by another vesicle).
    rav_ribbon_distance = 80  # nm
    rav_ids = seg_ids[ribbon_distances < rav_ribbon_distance]
    # Filter out the blocked vesicles.
    rav_ids = filter_blocked_segmentation_to_object_distances(
        vesicles, distance_path_ribbon, seg_ids=rav_ids, scale=scale,
    )

    # Find the vesicles that are membrane proximal (MP-V).
    # Criterion: vesicles are closer than 50 nm to the membrane and closer than 100 nm to the PD.
    mpv_pd_distance = 100  # nm
    mpv_bd_distance = 50  # nm
    mpv_ids = seg_ids[np.logical_and(pd_distances < mpv_pd_distance, bd_distances < mpv_bd_distance)]

    # Find the vesicles that are membrane docked (Docked-V).
    # Criterion: vesicles are closer than 2 nm to the membrane and closer than 100 nm to the PD.
    docked_pd_distance = 100  # nm
    docked_bd_distance = 2  # nm
    docked_ids = seg_ids[np.logical_and(pd_distances < docked_pd_distance, bd_distances < docked_bd_distance)]

    # Keep only the vesicle ids that are in one of the three categories.
    vesicle_ids = np.unique(np.concatenate([rav_ids, mpv_ids, docked_ids]))

    # Create a dictionary to map vesicle ids to their corresponding pool.
    # (RA-V get's over-written by MP-V, which is correct).
    pool_assignments = {vid: "RA-V" for vid in rav_ids}
    pool_assignments.update({vid: "MP-V" for vid in mpv_ids})
    pool_assignments.update({vid: "Docked-V" for vid in docked_ids})

    return vesicle_ids, pool_assignments


def compute_radii(vesicles, vesicle_ids, resolution):
    props = regionprops(vesicles)
    radii = {
        prop.label: resolution[0] * (prop.axis_minor_length + prop.axis_major_length) / 2
        for prop in props
    }
    radii = [radii[ves_id] for ves_id in vesicle_ids]
    return radii


def visualize_distances(
    tomo, vesicles, ribbon, pd, boundaries,
    distance_path_ribbon, distance_path_pd, distance_path_boundaries,
    resolution, scale=2, show=True,
):
    import napari
    from synaptic_reconstruction.tools.distance_measurement import _downsample

    tomo = _downsample(tomo, scale=scale)
    vesicles = _downsample(vesicles, is_seg=True, scale=scale)
    ribbon = _downsample(ribbon, is_seg=True, scale=scale)
    pd = _downsample(pd, is_seg=True, scale=scale)
    boundaries = _downsample(boundaries, is_seg=True, scale=scale)

    # Associate the vesicles with vesicle pools.
    vesicle_ids, pool_assignments = assign_vesicles_to_pools(
        vesicles, distance_path_ribbon, distance_path_pd, distance_path_boundaries, scale=scale
    )
    #
    # ves_lines, ves_props = create_distance_lines(distance_path_vesicles, n_neighbors=3, scale=scale)
    vesicle_radii = compute_radii(vesicles, vesicle_ids, resolution)

    # Only the RA-V distances
    ribbon_lines, ribbon_props = create_object_distance_lines(
        distance_path_ribbon, seg_ids=vesicle_ids, scale=scale
    )

    # Only the MP-V and docked distances
    pd_lines, pd_props = create_object_distance_lines(distance_path_pd, seg_ids=vesicle_ids, scale=scale)
    boundary_lines, boundary_props = create_object_distance_lines(
        distance_path_boundaries, seg_ids=vesicle_ids, scale=scale
    )

    # ves_dist = pandas.DataFrame(ves_props)
    ribbon_dist = pandas.DataFrame(ribbon_props)
    pd_dist = pandas.DataFrame(pd_props)
    boundary_dist = pandas.DataFrame(boundary_props)
    ves_assignments = pandas.DataFrame.from_dict(
        {
            "id": list(pool_assignments.keys()),
            "pool": list(pool_assignments.values()),
            "radius [nm]": vesicle_radii,
        }
    )
    if not show:
        return ves_assignments, ribbon_dist, pd_dist, boundary_dist

    vesicle_pools = np.zeros_like(vesicles)
    for pool_id, pool_name in enumerate(("RA-V", "MP-V", "Docked-V"), 1):
        ves_ids_pool = [vid for vid, pname in pool_assignments.items() if pname == pool_name]
        vesicle_pools[np.isin(vesicles, ves_ids_pool)] = pool_id

    v = napari.Viewer()
    v.add_image(tomo)
    v.add_labels(vesicles, visible=False)
    v.add_labels(vesicle_pools)
    v.add_labels(ribbon)
    v.add_labels(pd, name="presynaptic-density")
    v.add_labels(boundaries)

    # v.add_shapes(ves_lines, shape_type="line", name="vesicle-distances", visible=False)
    v.add_shapes(ribbon_lines, shape_type="line", name="ribbon-distances", visible=False)
    v.add_shapes(pd_lines, shape_type="line", name="presynaptic-distances", visible=False)
    v.add_shapes(boundary_lines, shape_type="line", name="boundary-distances", visible=False)

    napari.run()

    return ves_assignments, ribbon_dist, pd_dist, boundary_dist


def to_excel(
    ves_assignments,
    ribbon_distances,
    pd_distances,
    boundary_distances,
    morphology_measurements,
    result_path
):
    ves_assignments.to_excel(result_path, sheet_name="vesicle-pools", index=False)
    with pandas.ExcelWriter(result_path, engine="openpyxl", mode="a") as writer:
        ribbon_distances.to_excel(writer, sheet_name="vesicle-ribbon-distances", index=False)
        pd_distances.to_excel(writer, sheet_name="vesicle-pd-distances", index=False)
        boundary_distances.to_excel(writer, sheet_name="vesicle-boundary-distances", index=False)
        morphology_measurements.to_excel(writer, sheet_name="morphology", index=False)


# Any more morphology?
# Surface?
def compute_morphology(ribbon, pd):
    ribbon_size = ribbon.sum()  # in pixels
    pd_size = pd.sum()  # in pixels
    morphology = pandas.DataFrame({
        "structure": ["ribbon", "presynaptic-density"],
        "size [pixel]": [ribbon_size, pd_size]
    })
    return morphology


def process_distance_measurements(tomo_path, show):
    print("Processing distances for", tomo_path)

    with mrcfile.open(tomo_path, "r") as f:
        resolution = f.voxel_size.tolist()
    # bring to nanometer
    resolution = [res / 10 for res in resolution]
    print("The resolution is", resolution)

    vesicles, ribbon, pd, boundaries = _get_data(tomo_path)
    if pd.sum() == 0:
        # Empty postsynaptic density is not yet supported
        raise NotImplementedError

    distance_save_folder = os.path.join(os.path.split(tomo_path)[0], "distances")
    os.makedirs(distance_save_folder, exist_ok=True)
    ribbon_dist, pd_dist, boundary_dist = compute_all_distances(
        vesicles, ribbon, pd, boundaries, resolution, distance_save_folder
    )

    # Compute additional morphology for the PD and Ribbon
    morphology_measurements = compute_morphology(ribbon, pd)

    with open_file(tomo_path, "r") as f:
        tomo = f["data"][:]
    ves_assignments, ribbon_dist, pd_dist, boundary_dist = visualize_distances(
        tomo, vesicles, ribbon, pd, boundaries, ribbon_dist, pd_dist, boundary_dist,
        resolution=resolution, show=show
    )

    out_path = os.path.join(distance_save_folder, "measurements.xlsx")
    to_excel(
        ves_assignments, ribbon_dist, pd_dist, boundary_dist,
        morphology_measurements=morphology_measurements,
        result_path=out_path,
    )


def main():
    # path on laptop
    parser = argparse.ArgumentParser()
    parser.add_argument("tomo_path")
    parser.add_argument("-s", "--show", action="store_true")
    parser.add_argument("-a", "--compute_all_distances", action="store_true")

    args = parser.parse_args()
    if args.compute_all_distances:
        precompute_all_distances(args.tomo_path)
    else:
        process_distance_measurements(args.tomo_path, show=args.show)


if __name__ == "__main__":
    main()
