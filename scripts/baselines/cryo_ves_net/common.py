import os
import tempfile
from glob import glob
from pathlib import Path

import h5py
import mrcfile
import cryovesnet


# additional parameters?
def _segment_vesicles(directory):
    pl = cryovesnet.Pipeline(directory, pattern="*.mrc")
    pl.setup_cryovesnet_dir(make_masks=False)

    pl.run_deep()
    pl.rescale()
    pl.label_vesicles(within_segmentation_region=False)
    pl.label_vesicles_adaptive(separating=True)
    pl.make_spheres()
    pl.repair_spheres()


def _prepare_input(path, output_folder, input_key, resolution):
    fname = Path(path).stem
    sub_folder = os.path.join(output_folder, fname)

    os.makedirs(sub_folder, exist_ok=True)
    out_path = os.path.join(sub_folder, f"{fname}.mrc")

    if path.endswith(".h5"):
        assert resolution is not None
        with h5py.File(path, "r") as f:
            vol = f[input_key][:]

        mrcfile.new(out_path, data=vol)
        with mrcfile.open(out_path, mode="r+") as f:
            f.voxel_size = resolution
            f.update_header_from_data()

    # TODO copy the file and ensure the resolution
    elif path.endswith(".mrc"):
        pass

    return out_path, sub_folder


def _process_output(tmp, tmp_file, output_folder, output_key):
    fname = Path(tmp_file).stem
    seg_path = os.path.join(tmp, "cryovesnet", f"{fname}_convex_labels.mrc")
    with mrcfile.open(seg_path, "r") as f:
        seg = f.data[:]

    out_path = os.path.join(output_folder, f"{fname}.h5")
    with h5py.File(out_path, "a") as f:
        f.create_dataset(output_key, data=seg, compression="gzip")


# TODO support nested
# TODO support mask
def apply_cryo_vesnet(
    input_folder, output_folder, pattern, input_key,
    resolution=None, output_key="prediction/vesicles/cryovesnet"
):
    os.makedirs(output_folder, exist_ok=True)
    files = sorted(glob(os.path.join(input_folder, pattern)))

    with tempfile.TemporaryDirectory() as tmp:

        for file in files:

            # Get the resolution info for this file.
            if resolution is None:
                res = None
            else:
                fname = Path(file).stem
                res = resolution[fname] if isinstance(resolution, dict) else resolution

            # Prepare the input files by copying them over or resaving them (if h5).
            tmp_file, sub_folder = _prepare_input(file, tmp, input_key, res)

            # Segment the vesicles in the file.
            _segment_vesicles(sub_folder)

            # Write the output file.
            _process_output(sub_folder, tmp_file, output_folder, output_key)
