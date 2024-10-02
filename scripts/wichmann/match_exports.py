import os
import h5py
import numpy as np
import argparse

def get_h5_file_content(file_path):
    """Read the content of the 'raw' key from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        try:
            return f["raw"][:]
        except:
            return None

def combine_h5_files(file1_path, vesicle_file2_name):
    """Combine two HDF5 files, keeping file1 as is and add key "vesicles" from vesicle file to it"""
    with h5py.File(vesicle_file2_name) as f:
        labels = f["labels"]
        vesicles = labels["vesicles"][:]
    
    with h5py.File(file1_path, "a") as f:
        f.create_dataset("labels/vesicles", data=vesicles, compression="gzip")


def find_matching_files(folder1, vesicle_folder2):
    """Find matching files between two folders based on the 'raw' content."""
    folder1_files = {f: os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.h5')}
    vesicles_folder2_files = {f: os.path.join(vesicle_folder2, f) for f in os.listdir(vesicle_folder2) if f.endswith('.h5')}

    matches = []

    # Compare 'raw' content between files
    for file1_name, file1_path in folder1_files.items():
        raw_data1 = get_h5_file_content(file1_path)
        if raw_data1 is None:
            continue

        for file2_name, file2_path in vesicles_folder2_files.items():
            raw_data2 = get_h5_file_content(file2_path)
            if raw_data2 is None:
                continue

            # Check if 'raw' datasets match
            if np.array_equal(raw_data1, raw_data2):
                matches.append((file1_path, file2_path))
                break

    return matches

def combine_matching_files(folder1, vesicle_folder2):
    """Combine matching files from two folders."""

    matches = find_matching_files(folder1, vesicle_folder2)

    for file1, vesicle_file2 in matches:
        file1_name = os.path.basename(file1)
        vesicle_file2_name = os.path.basename(vesicle_folder2)

        combine_h5_files(file1, vesicle_file2)
        print(f"Combined: {file1_name} and {vesicle_file2_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder1")
    parser.add_argument("vesicle_folder2")
    args = parser.parse_args()

    combine_matching_files(args.folder1, args.vesicle_folder2)




if __name__ == "__main__":
    main()

