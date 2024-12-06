import os
from glob import glob

import h5py

from synapse_net.imod.to_imod import write_segmentation_to_imod_as_points


def export_all_to_imod(check_input=True, check_export=True):
    files = sorted(glob("./proofread_az/**/*.h5", recursive=True))
    mrc_root = "./mrc_files"
    output_folder = "./vesicle_export"

    for ff in files:
        ds, fname = os.path.split(ff)
        ds = os.path.basename(ds)
        out_folder = os.path.join(output_folder, ds)
        out_path = os.path.join(out_folder, fname.replace(".h5", ".mod"))
        if os.path.exists(out_path):
            continue

        os.makedirs(out_folder, exist_ok=True)
        mrc_path = os.path.join(mrc_root, ds, fname.replace(".h5", ".rec"))
        assert os.path.exists(mrc_path), mrc_path

        with h5py.File(ff, "r") as f:
            seg = f["vesicles"][:]

        write_segmentation_to_imod_as_points(mrc_path, seg, out_path, min_radius=7, radius_factor=0.7)


def main():
    export_all_to_imod()


if __name__ == "__main__":
    main()
