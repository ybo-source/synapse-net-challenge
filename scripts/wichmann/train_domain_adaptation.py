import os
import argparse
from glob import glob
import json

from sklearn.model_selection import train_test_split
from synaptic_reconstruction.training.domain_adaptation import mean_teacher_adaptation

TRAIN_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held/"
OUTPUT_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/DA_training_endbulb_v2"

def _require_train_val_test_split(datasets):
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

    def _train_val_test_split(names):
        train, test = train_test_split(names, test_size=1 - train_ratio, shuffle=True)
        _ratio = test_ratio / (test_ratio + val_ratio)
        val, test = train_test_split(test, test_size=_ratio)
        return train, val, test

    for ds in datasets:
        print(ds)
        split_path = os.path.join(OUTPUT_ROOT, f"split-{ds}.json")
        if os.path.exists(split_path):
            continue

        file_paths = sorted(glob(os.path.join(TRAIN_ROOT, ds, "*.h5")))
        file_names = [os.path.basename(path) for path in file_paths]

        # Check if there is only one file
        if len(file_names) == 1:
            print(f"Warning: Dataset '{ds}' contains only one file. No validation split will be created for this dataset.")
            train = file_names  # Assign the single file to the train split
            val = []  # No validation files
        else:
            train, val, test = _train_val_test_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val, "test": test}, f)

def _require_train_val_split(datasets):
    train_ratio, val_ratio = 0.8, 0.2

    # Define the list of files to skip
    testset = {
        "1Otof_AVCN03_429A_WT_M.Stim_D3_2_35461.h5",
        "1Otof_AVCN03_429D_WT_Rest_H5_4_35461.h5",
        "1Otof_AVCN07_451A_WT_Rest_B3_10_35932.h5",
        "Otof_AVCN07_449S_KO_Rest_G5_7_35934.h5",
        "Otof_AVCN07_449T_KO_M.Stim_M1_6_35934.h5"
    }

    def _train_val_split(names):
        train, val = train_test_split(names, test_size=1 - train_ratio, shuffle=True)
        return train, val

    for ds in datasets:
        print(ds)
        split_path = os.path.join(OUTPUT_ROOT, f"split-{ds}.json")
        if os.path.exists(split_path):
            continue

        file_paths = sorted(glob(os.path.join(TRAIN_ROOT, ds, "*.h5")))
        file_names = [os.path.basename(path) for path in file_paths]

        # Filter out files that should be skipped
        file_names = [name for name in file_names if name not in testset]

        # Check if there are no files left after filtering
        if len(file_names) == 0:
            print(f"Warning: Dataset '{ds}' has no files left after filtering. Skipping this dataset.")
            continue
        
        # Check if there is only one file
        if len(file_names) == 1:
            print(f"Warning: Dataset '{ds}' contains only one file. No validation split will be created for this dataset.")
            train = file_names  # Assign the single file to the train split
            val = []  # No validation files
        else:
            train, val = _train_val_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val}, f)

def get_paths(split, datasets, testset=True):
    if testset:
        _require_train_val_test_split(datasets)
    else:
        _require_train_val_split(datasets)

    paths = []
    for ds in datasets:
        split_path = os.path.join(OUTPUT_ROOT, f"split-{ds}.json")
        with open(split_path) as f:
            names = json.load(f)[split]
        ds_paths = [os.path.join(TRAIN_ROOT, ds, name) for name in names]
        assert all(os.path.exists(path) for path in ds_paths)
        paths.extend(ds_paths)

    return paths

def vesicle_domain_adaptation(teacher_model, testset = True):
    datasets = [
    "Adult_KO_MStim",
    "Adult_KO_Rest",
    "Adult_WT_MStim",
    "Adult_WT_Rest",
    "Old_KO_MStim",
    "Old_WT_Rest",
    "Young_KO_MStim",
    "Young_KO_Rest",
    "Young_WT_MStim",
    "Young_WT_Rest"
]
    train_paths = get_paths("train", datasets=datasets, testset=testset)
    val_paths = get_paths("val", datasets=datasets, testset=testset)
    
    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    #adjustable parameters
    patch_shape = [48, 256, 256]
    model_name = "vesicle-DA-endbulb-v2"
    
    model_root = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/models_v2/checkpoints/"
    checkpoint_path = os.path.join(model_root, teacher_model)

    mean_teacher_adaptation(
        name=model_name,
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        raw_key="raw",
        patch_shape=patch_shape,
        save_root="/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/DA_models",
        source_checkpoint=checkpoint_path,
        confidence_threshold=0.75,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--teacher_model", required=True, help="Name of teacher model")
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()
    
    vesicle_domain_adaptation(args.teacher_model, args.testset)


if __name__ == "__main__":
    main()
