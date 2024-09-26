import os
import shutil
import json

def copy_test_set(json_file, input_path, output_path):
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Copy files in the test set to the output directory
    for file_name in data['test']:
        src_path = os.path.join(input_path, file_name)
        dst_path = os.path.join(output_path, file_name)
        shutil.copy(src_path, dst_path)
        print(f"Copied {file_name} to {output_path}")


json_file = "/scratch-emmy/usr/nimsmuth/synapse_seg/data/training/split-12_chemical_fix_cryopreparation.json"
input_path = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/extracted/20240909_cp_datatransfer/12_chemical_fix_cryopreparation"
output_path = "/scratch-emmy/usr/nimsmuth/synapse_seg/data/training/testsets/12_chemical_fix_cryopreparation"
os.makedirs(output_path, exist_ok=True)

copy_test_set(json_file, input_path, output_path)

# 01_hoi_maus_2020_incomplete  03_hog_cs1sy7         05_stem750_sv_training  07_hoi_s1sy7_tem250_ihgp       09_stem750_66k         11_tem_multiple_release
# 02_hcc_nanogold              04_hoi_stem_examples  06_hoi_wt_stem750_fm    08_hcc_wt_lumsyt1nb_tem250_kw  10_tem_single_release  12_chemical_fix_cryopreparation