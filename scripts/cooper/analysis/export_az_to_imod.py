import os
import tempfile
from glob import glob
from subprocess import run
from shutil import copyfile

import h5py
import pandas as pd

from synaptic_reconstruction.imod.to_imod import write_segmentation_to_imod
from scipy.ndimage import binary_dilation, binary_closing


def check_imod(tomo_path, mod_path):
    run(["imod", tomo_path, mod_path])


def export_all_to_imod(check_input=True, check_export=True):
    files = sorted(glob("./az_segmentation/**/*.h5", recursive=True))
    mrc_root = "./mrc_files"
    output_folder = "./az_export/initial_model"

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
            seg = f["thin_az"][:]

        seg = binary_dilation(seg, iterations=2)
        seg = binary_closing(seg, iterations=2)

        write_segmentation_to_imod(mrc_path, seg, out_path)

        if check_input:
            import napari
            from elf.io import open_file
            with open_file(mrc_path, "r") as f:
                raw = f["data"][:]
            v = napari.Viewer()
            v.add_image(raw)
            v.add_labels(seg)
            napari.run()

        if check_export:
            check_imod(mrc_path, out_path)


# https://bio3d.colorado.edu/imod/doc/man/reducecont.html
def reduce_all_contours():
    pass


# https://bio3d.colorado.edu/imod/doc/man/smoothsurf.html#TOP
def smooth_all_surfaces(check_output=True):
    input_files = sorted(glob("./az_export/initial_model/**/*.mod", recursive=True))

    mrc_root = "./mrc_files"
    output_folder = "./az_export/smoothed_model"
    for ff in input_files:
        ds, fname = os.path.split(ff)
        ds = os.path.basename(ds)
        out_folder = os.path.join(output_folder, ds)
        out_file = os.path.join(out_folder, fname)
        if os.path.exists(out_file):
            continue

        os.makedirs(out_folder, exist_ok=True)
        run(["smoothsurf", ff, out_file])
        if check_output:
            mrc_path = os.path.join(mrc_root, ds, fname.replace(".mod", ".rec"))
            assert os.path.exists(mrc_path), mrc_path
            check_imod(mrc_path, out_file)


def measure_surfaces():
    input_files = sorted(glob("./az_export/smoothed_model/**/*.mod", recursive=True))

    result = {
        "Dataset": [],
        "Tomogram": [],
        "AZ Surface": [],
    }
    for ff in input_files:
        ds, fname = os.path.split(ff)
        ds = os.path.basename(ds)
        fname = os.path.splitext(fname)[0]

        with tempfile.NamedTemporaryFile() as f_mesh, tempfile.NamedTemporaryFile() as f_mod:
            tmp_path_mesh = f_mesh.name
            tmp_path_mod = f_mod.name
            copyfile(ff, tmp_path_mesh)
            run(["imodmesh", tmp_path_mesh])
            run(["imodinfo", "-f", tmp_path_mod, tmp_path_mesh])
            area = None
            with open(tmp_path_mod, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line.startswith("Total mesh surface area"):
                        area = float(line.split(" ")[-1])
            assert area is not None
            area /= 2

            result["Dataset"].append(ds)
            result["Tomogram"].append(fname)
            result["AZ Surface"].append(area)

    result = pd.DataFrame(result)
    result.to_excel("./az_measurements_all.xlsx", index=False)


def filter_surfaces():
    all_results = pd.read_excel("./az_measurements_all.xlsx")
    man_tomos = pd.read_csv("./man_tomos.tsv")

    man_results = all_results.merge(man_tomos[["Dataset", "Tomogram"]], on=["Dataset", "Tomogram"], how="inner")
    man_results.to_excel("./az_measuerements_manual.xlsx", index=False)


def main():
    export_all_to_imod(False, False)
    smooth_all_surfaces(False)
    # measure_surfaces()
    filter_surfaces()


if __name__ == "__main__":
    main()
