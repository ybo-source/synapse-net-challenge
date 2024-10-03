import shutil
import tempfile
from subprocess import run

import imageio.v3 as imageio
from elf.io import open_file
import argparse
import mrcfile
import os
import numpy as np
import sys
from scipy.ndimage import binary_closing

def transpose_tomo(tomogram):
        data0 = np.swapaxes(tomogram, 0, -1)
        data1 = np.fliplr(data0)
        transposed_data = np.swapaxes(data1, 0, -1)
        return transposed_data


def get_label_names(imod_path,  return_types=False):
    cmd = "imodinfo"

    label_names, label_types = {}, {}

    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name
        run([cmd, "-f", tmp_path, imod_path])

        object_id = None
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines:

                if line.startswith("OBJECT"):
                    object_id = int(line.rstrip("\n").strip().split()[-1])

                if line.startswith("NAME"):
                    name = ' '.join(line.rstrip("\n").strip().split()[1:]) 
                    assert object_id is not None
                    label_names[object_id] = name

                if "object uses" in line:
                    type_ = " ".join(line.rstrip("\n").strip().split()[2:]).rstrip(".")
                    label_types[object_id] = type_

    if return_types:
        return label_names, label_types
    return label_names

def pick_cristae(imod_path):
    object_list=get_label_names(imod_path)
    cristae_id_list  = []
    for object_id, name in object_list.items():
        # Check if 'cristae' is present in the name (case insensitive)
        if 'cristae' in name.lower():
            cristae_id_list.append(object_id)

    print(object_list)
    print(cristae_id_list)

    if not cristae_id_list:
        print("No cristae objects found.")
        return None

    cristae_ids_string = ','.join(map(str, cristae_id_list))
    print(cristae_ids_string)
    
    return cristae_ids_string

def pick_mito(imod_path):
    object_list=get_label_names(imod_path)
    mito_id_list = []
    for object_id, name in object_list.items():
        # Check if 'mito' is present in the name (case insensitive) and 'cristae' is not present
        if 'mito' in name.lower() and 'cristae' not in name.lower():
            mito_id_list.append(object_id)

    print(object_list)
    print(mito_id_list)

    if not mito_id_list:
        print("No cristae objects found.")
        return None
    
    cristae_ids_string = ','.join(map(str, mito_id_list))
    print(cristae_ids_string)
    
    return cristae_ids_string

def pick_endbulb(imod_path):
    object_list=get_label_names(imod_path)
    endbulb_id_list = []
    for object_id, name in object_list.items():
        # Check if 'endbulb' is present in the name (case insensitive)
        if 'endbulb' in name.lower():
            endbulb_id_list.append(object_id)

    print(object_list)
    print(endbulb_id_list)

    if not endbulb_id_list:
        print("No cristae objects found.")
        return None
    
    endbulb_ids_string = ','.join(map(str, endbulb_id_list))
    print(endbulb_ids_string)
    
    return endbulb_ids_string

def pick_AZ(imod_path):
    object_list=get_label_names(imod_path)
    AZ_id_list = []
    for object_id, name in object_list.items():
        # Check if 'AZ' is present in the name (case insensitive)
        if 'az' in name.lower() or 'active zone' in name.lower():
            AZ_id_list.append(object_id)

    print(object_list)
    print(AZ_id_list)

    if not AZ_id_list:
        print("No cristae objects found.")
        return None
    
    AZ_ids_string = ','.join(map(str, AZ_id_list))
    print(AZ_ids_string)
    
    return AZ_ids_string

def export_segmentation(imod_path, mrc_path, object_ids_string, object_id=None, output_path=None, require_object=True):
    cmd = "imodmop"
    cmd_path = shutil.which(cmd)
    assert cmd_path is not None, f"Could not find the {cmd} imod command."

    #sys.exit()
    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name

        if object_id is None: 
            cmd = [cmd, "-l", object_ids_string, "-o", object_ids_string, imod_path, mrc_path, tmp_path] #"-ma", "1",
        else:
            cmd = [cmd, "-ma", "1", "-o", str(object_id), imod_path, mrc_path, tmp_path]

        run(cmd)
        #print(mrc_path)
        with open_file(tmp_path, ext=".mrc", mode="r") as f:
            data = f["data"][:]
    unique_values = np.unique(data)
    num_unique_values = len(unique_values)
    print(f"num unique data: {num_unique_values}")
    segmentation = data #== 1
    if require_object and segmentation.sum() == 0:
        id_str = "" if object_id is None else f"for object {object_id}"
        raise RuntimeError(f"Segmentation extracted from {imod_path} {id_str} is empty.")

    combined_segmentation = segmentation.copy()

    for segmentation_id in (id for id in unique_values if id != 0):
        binarized = segmentation == segmentation_id
        structuring_element = np.zeros((3, 1, 1))
        structuring_element[:, 0, 0] = 1
        closed_segmentation = binary_closing(binarized, iterations=2, structure=structuring_element)

        combined_segmentation[closed_segmentation] = segmentation_id 

    if output_path is None:
        return combined_segmentation 

    imageio.imwrite(output_path, segmentation.astype("uint8"), compression="zlib")

def save_segmentation( out_path, mrc_path, segmentation_mito=None, segmentation_cristae=None, segmentation_endbulb=None, segmentation_AZ=None):
    with mrcfile.open(mrc_path) as mrc:
        data = mrc.data
    
    data = transpose_tomo(data)

    file_name_with_extension = os.path.basename(mrc_path)
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    out_path = os.path.join(out_path, file_name_without_extension + ".h5")

    with open_file(out_path, "a") as f:
        f.create_dataset("raw", data=data, compression="gzip")
        if segmentation_mito is not None:
            f.create_dataset("labels/mitochondria", data=segmentation_mito, compression="gzip")
        if segmentation_cristae is not None:
            f.create_dataset("labels/cristae", data=segmentation_cristae, compression="gzip")
        if segmentation_endbulb is not None:
            f.create_dataset("labels/endbulb", data=segmentation_endbulb, compression="gzip")
        if segmentation_AZ is not None:
            f.create_dataset("labels/AZ", data=segmentation_AZ, compression="gzip")

def convert_file(imod_path, mrc_path, out_path):

    print(get_label_names(imod_path))
    cristae_ids_string = pick_cristae(imod_path)
    segmentation_cristae = export_segmentation(imod_path, mrc_path, object_ids_string=cristae_ids_string) if cristae_ids_string is not None else None
    mito_ids_string = pick_mito(imod_path)
    segmentation_mito = export_segmentation(imod_path, mrc_path, object_ids_string=mito_ids_string) if mito_ids_string is not None else None
    endbulb_ids_string = pick_endbulb(imod_path)
    segmentation_endbulb = export_segmentation(imod_path, mrc_path, object_ids_string=endbulb_ids_string) if endbulb_ids_string is not None else None
    AZ_ids_string = pick_AZ(imod_path)
    segmentation_AZ = export_segmentation(imod_path, mrc_path, object_ids_string=AZ_ids_string) if AZ_ids_string is not None else None
    save_segmentation(out_path, mrc_path, segmentation_mito, segmentation_cristae, segmentation_endbulb, segmentation_AZ)


def convert_folder(imod_path, mrc_path, out_path):
    print("Converting folder")
    imod_files = sorted([f for f in os.listdir(imod_path) if f.endswith('.mod')])
    mrc_files = sorted([f for f in os.listdir(mrc_path) if f.endswith('.rec')])

    for imod_file, mrc_file in zip(imod_files, mrc_files):
        print("you joking?")
        convert_file(os.path.join(imod_path, imod_file), os.path.join(mrc_path, mrc_file), out_path)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imod_path", required=True)
    parser.add_argument("-m", "--mrc_path", required=True)
    parser.add_argument("-o", "--out_path", required=True)
    args = parser.parse_args()
    imod_path =args.imod_path

    if os.path.isdir(imod_path):
        convert_folder(imod_path, args.mrc_path, args.out_path)
    else:
        convert_file(imod_path, args.mrc_path, args.out_path)



if __name__ == "__main__":
    main()
