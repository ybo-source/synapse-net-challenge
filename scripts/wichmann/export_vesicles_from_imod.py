import os
from glob import glob
from pathlib import Path
import argparse
import sys
sys.path.append('/home/smuth/Documents/PhD/code/synaptic-reconstruction')
from synapse_net.ground_truth import extract_vesicle_training_data

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/original_imod_data/endbulb_of_held/Adult"  # noqa
OUT_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held/Adult"  # noqa

def extract(ROOT=ROOT,OUT_ROOT=OUT_ROOT, name = "KO_MStim"):
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem
        imod_path = os.path.join(ROOT, name, f"{fname}_model.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, visualize=False,
        exclude_label_patterns=["Endbulb", "Active Zone"]
    ) #, exclude=["1Otof_AVCN07_449T_KO_M.Stim_M4_4_35934.rec"]
    return output_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imod_path")
    parser.add_argument("-o", "--out_path")
    parser.add_argument("-n", "--name")
    args = parser.parse_args()
    extract(args.imod_path, args.out_path, args.name)



if __name__ == "__main__":
    main()
