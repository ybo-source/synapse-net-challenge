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
    pl.label_vesicles(within_segmentation_region=False)
    pl.label_vesicles_adaptive(separating=True)
    pl.make_spheres()
    pl.repair_spheres()


def _prepare_input(path, output_folder, input_key, resolution):
    out_path = os.path.join(output_folder, f"{Path(path).stem}.mrc")

    if path.endswith(".h5"):
        assert resolution is not None
        with h5py.File(path, "r") as f:
            vol = f[input_key][:]

        mrcfile.new(out_path, data=vol)
        with mrcfile.open(out_path, mode="r+") as f:
            f.header.cella.x = resolution[0]
            f.header.cella.y = resolution[1]
            f.header.cella.z = resolution[2]

    # TODO just copy the file
    elif path.endswith(".mrc"):
        pass


# TODO support nested
def apply_cryo_vesnet(
    input_folder, output_folder, pattern, input_key,
    resolution=None, output_key="prediction/vesicles/cryovesnet"
):
    files = sorted(glob(os.path.join(input_folder, pattern)))
    with tempfile.TemporaryDirectory() as tmp:

        # Prepare the input files by copying them over or resaving them (if h5).
        for file in files:
            if resolution is None:
                res = None
            else:
                fname = Path(file).stem
                res = resolution[fname] if isinstance(resolution, dict) else resolution
            _prepare_input(file, tmp, input_key, res)

        # Segment the vesicles in all files.
        _segment_vesicles(tmp)
        breakpoint()

        # TODO
        # Re-save the segmentations to the output folder.
