import os
from glob import glob

import h5py
import pandas as pd
from elf.evaluation.dice import dice_score
from synaptic_reconstruction.inference.ribbon_synapse import segment_ribbon_synapse_structures
from torch_em.util import load_model
from tqdm import tqdm

from train_structure_segmentation import get_train_val_test_split
from train_structure_segmentation import noop  # noqa

OUTPUT_ROOT = "./predictions"


def run_prediction(input_paths, model_path, name, is_nested=False, prefix=None):
    output_root = os.path.join(OUTPUT_ROOT, name)
    model = load_model(model_path)

    for path in input_paths:
        root, fname = os.path.split(path)
        if is_nested:
            folder_name = os.path.split(root)[1]
            output_folder = os.path.join(output_root, folder_name)
        else:
            output_folder = output_root

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, fname)

        if os.path.exists(output_path) and prefix is None:
            continue
        elif os.path.exists(output_path):
            with h5py.File(output_path, "r") as f:
                if prefix in f:
                    continue

        with h5py.File(path, "r") as f:
            tomogram = f["raw"][:]

        prediction = segment_ribbon_synapse_structures(
            input_volume=tomogram,
            model=model,
            verbose=True,
            threshold=0.5,
        )
        with h5py.File(output_path, "a") as f:
            for name, pred in prediction.items():
                ds_name = name if prefix is None else f"{prefix}/{name}"
                f.create_dataset(ds_name, data=pred, compression="gzip")


def visualize():
    pass


def evaluate(input_paths, name, is_nested=False, prefix=None, save_path=None, label_names=None):
    if save_path is not None and os.path.exists(save_path):
        return pd.read_csv(save_path)

    structure_names = ["ribbon", "PD", "membrane"]
    if label_names is None:
        label_names = structure_names
    output_root = os.path.join(OUTPUT_ROOT, name)

    results = {
        "method": [],
        "file_name": [],
    }
    results.update({nn: [] for nn in structure_names})
    for path in tqdm(input_paths, desc="Run evaluation"):
        root, fname = os.path.split(path)
        if is_nested:
            folder_name = os.path.split(root)[1]
            output_folder = os.path.join(output_root, folder_name)
        else:
            output_folder = output_root
        output_path = os.path.join(output_folder, fname)

        results["method"].append("Src" if prefix is None else prefix)
        results["file_name"].append(f"{folder_name}/{fname}" if is_nested else fname)

        with h5py.File(path, "r") as f_in, h5py.File(output_path, "r") as f_out:
            for sname, label_name in zip(structure_names, label_names):
                gt = f_in[f"labels/{label_name}"][:]
                pred = f_out[sname if prefix is None else f"{prefix}/{sname}"][:]
                score = dice_score(pred, gt)
                results[sname].append(score)

    results = pd.DataFrame(results)
    if save_path is not None:
        results.to_csv(save_path, index=False)
    return results


def predict_and_evaluate_train_domain():
    _, _, paths = get_train_val_test_split()
    print("Run evaluation on", len(paths), "tomos")

    name = "train_domain"
    model_path = "./checkpoints/inner_ear_structure_model"

    run_prediction(paths, model_path, name, is_nested=True)
    evaluate(paths, name, is_nested=True, save_path="./results/train_domain.csv")

    # TODO
    # visualize()


def predict_and_evaluate_target_domain(paths, name, adapted_model_path):
    print("Run evaluation on", len(paths), "tomos")

    src_model_path = "./checkpoints/inner_ear_structure_model"
    save_path = f"./results/{name}.csv"

    label_names = ["ribbons", "presynapse", "membrane"]

    if not os.path.exists(save_path):
        # Run prediction and evaluation for the source model
        run_prediction(paths, src_model_path, name, prefix="Src")
        results_src = evaluate(paths, name, prefix="Src", label_names=label_names)

        # Run prediction and evaluation for the adapted model
        run_prediction(paths, adapted_model_path, name, prefix="Adapted")
        results_adapted = evaluate(paths, name, prefix="Adapted", label_names=label_names)

        # Join and save the results
        results = pd.concat([results_src, results_adapted])
        results.to_csv(save_path, index=False)

    # TODO
    # visualize()


def predict_and_evaluate_vesicle_pools():
    paths = sorted(glob(
        "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/other_tomograms/01_vesicle_pools/*.h5"  # noqa
    ))
    adapted_model_path = "./checkpoints/structure-model-adapt-vesicle_pools"
    predict_and_evaluate_target_domain(paths, "vesicle_pools", adapted_model_path)


def predict_and_evaluate_rat():
    paths = sorted(glob(
        "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/other_tomograms/03_ratten_tomos/*.h5"
    ))
    adapted_model_path = "./checkpoints/structure-model-adapt-rat"
    predict_and_evaluate_target_domain(paths, "rat", adapted_model_path)


def main():
    # predict_and_evaluate_train_domain()
    # predict_and_evaluate_vesicle_pools()
    predict_and_evaluate_rat()


if __name__ == "__main__":
    main()
