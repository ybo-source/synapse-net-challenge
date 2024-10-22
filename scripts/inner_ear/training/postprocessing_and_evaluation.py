import os
from glob import glob

import h5py

from synaptic_reconstruction.inference.vesicles import segment_vesicles
from synaptic_reconstruction.inference.postprocessing.ribbon import segment_ribbon
from synaptic_reconstruction.inference.postprocessing.presynaptic_density import segment_presynaptic_density
from torch_em.util import load_model
from tqdm import tqdm

from train_structure_segmentation import get_train_val_test_split

# ROOT = "/home/pape/Work/data/synaptic_reconstruction/moser"
ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser"
MODEL_PATH = "/mnt/lustre-emmy-hdd/projects/nim00007/models/synaptic-reconstruction/vesicle-DA-inner_ear-v2"
OUTPUT_ROOT = "./predictions"


def run_vesicle_segmentation(input_paths, model_path, name, is_nested=False):
    output_root = os.path.join(OUTPUT_ROOT, name)
    model = None

    for path in input_paths:
        root, fname = os.path.split(path)
        if is_nested:
            folder_name = os.path.split(root)[1]
            output_folder = os.path.join(output_root, folder_name)
        else:
            output_folder = output_root

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, fname)

        if os.path.exists(output_path):
            with h5py.File(output_path, "r") as f:
                if "vesicles" in f:
                    continue

        if model is None:
            model = load_model(model_path)

        with h5py.File(path, "r") as f:
            tomogram = f["raw"][:]

        seg = segment_vesicles(input_volume=tomogram, model=model)
        with h5py.File(output_path, "a") as f:
            f.create_dataset("vesicles", data=seg, compression="gzip")


def postprocess_structures(paths, name, prefix=None, is_nested=False):
    output_root = os.path.join(OUTPUT_ROOT, name)

    for path in tqdm(paths):
        root, fname = os.path.split(path)
        if is_nested:
            folder_name = os.path.split(root)[1]
            output_folder = os.path.join(output_root, folder_name)
        else:
            output_folder = output_root
        output_path = os.path.join(output_folder, fname)

        with h5py.File(output_path, "r") as f:
            if prefix is None and "segmentation" in f:
                continue
            elif prefix is not None and f"{prefix}/segmentation" in f:
                continue

            vesicles = f["vesicles"][:]
            if prefix is None:
                ribbon_pred = f["ribbon"][:]
                presyn_pred = f["PD"][:]
            else:
                ribbon_pred = f[f"{prefix}/ribbon"][:]
                presyn_pred = f[f"{prefix}/PD"][:]

        ribbon = segment_ribbon(ribbon_pred, vesicles, n_slices_exclude=15, n_ribbons=1)
        presyn = segment_presynaptic_density(presyn_pred, ribbon, n_slices_exclude=15)

        with h5py.File(output_path, "a") as f:
            if prefix is None:
                f.create_dataset("segmentation/ribbon", data=ribbon, compression="gzip")
                f.create_dataset("segmentation/PD", data=presyn, compression="gzip")
            else:
                f.create_dataset(f"{prefix}/segmentation/ribbon", data=ribbon, compression="gzip")
                f.create_dataset(f"{prefix}/segmentation/PD", data=presyn, compression="gzip")


def segment_train_domain():
    _, _, paths = get_train_val_test_split(os.path.join(ROOT, "inner_ear_data"))
    print("Run evaluation on", len(paths), "tomos")
    name = "train_domain"
    run_vesicle_segmentation(paths, MODEL_PATH, name, is_nested=True)
    postprocess_structures(paths, name, is_nested=True)


def segment_vesicle_pools():
    paths = sorted(glob(os.path.join(ROOT, "other_tomograms/01_vesicle_pools", "*.h5")))
    run_vesicle_segmentation(paths, MODEL_PATH, "vesicle_pools")


def segment_rat():
    paths = sorted(glob(os.path.join(ROOT, "other_tomograms/03_ratten_tomos", "*.h5")))
    run_vesicle_segmentation(paths, MODEL_PATH, "rat")


def main():
    segment_train_domain()
    # segment_vesicle_pools()
    # segment_rat()


if __name__ == "__main__":
    main()
