import os
from glob import glob

from elf.io import open_file
from micro_sam.precompute_state import precompute_state
from skimage.transform import rescale

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer"  # noqa
ROOT_CRYO = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/fernandez-busnadiego/vesicle_gt/v1"  # noqa


def preprocess_tomogram(dataset, tomogram):
    scale = (0.25, 0.25, 0.25)

    output_root = f"./output/{dataset}"
    output_tomos = os.path.join(output_root, "tomograms")
    output_embed = os.path.join(output_root, "embeddings")
    os.makedirs(output_tomos, exist_ok=True)
    os.makedirs(output_embed, exist_ok=True)

    fname = os.path.splitext(tomogram)[0]
    input_path = os.path.join(output_tomos, f"{fname}.h5")
    output_path = os.path.join(output_embed, f"{fname}.zarr")
    if os.path.exists(output_path):
        return

    tomogram_path = os.path.join(ROOT, dataset, tomogram)
    with open_file(tomogram_path, "r") as f:
        tomo = f["data"][:]

    print("Resizing tomogram ...")
    tomo = rescale(tomo, scale, preserve_range=True).astype(tomo.dtype)

    with open_file(input_path, "a") as f:
        f.create_dataset("data", data=tomo, compression="gzip")

    print("Precompute state ...")
    precompute_state(
        input_path=input_path,
        output_path=output_path,
        model_type="vit_b",
        key="data",
        checkpoint_path="./checkpoints/compartment_model_v2/best.pt",
        ndim=3,
        precompute_amg_state=True,
    )


def preprocess_cryo_tomogram(fname):
    scale = (0.5, 0.5, 0.5)

    dataset = "cryo"
    output_root = f"./output/{dataset}"
    output_tomos = os.path.join(output_root, "tomograms")
    output_embed = os.path.join(output_root, "embeddings")
    os.makedirs(output_tomos, exist_ok=True)
    os.makedirs(output_embed, exist_ok=True)

    tomogram = os.path.join(ROOT_CRYO, f"{fname}.h5")

    input_path = os.path.join(output_tomos, f"{fname}.h5")
    output_path = os.path.join(output_embed, f"{fname}.zarr")
    if os.path.exists(output_path):
        return

    tomogram_path = os.path.join(ROOT_CRYO, dataset, tomogram)
    with open_file(tomogram_path, "r") as f:
        tomo = f["raw"][:]

    print("Resizing tomogram ...")
    tomo = rescale(tomo, scale, preserve_range=True).astype(tomo.dtype)

    with open_file(input_path, "a") as f:
        f.create_dataset("data", data=tomo, compression="gzip")

    print("Precompute state ...")
    precompute_state(
        input_path=input_path,
        output_path=output_path,
        model_type="vit_b",
        key="data",
        checkpoint_path="./checkpoints/compartment_model_v2/best.pt",
        ndim=3,
    )


def preprocess_05():
    dataset = "05_stem750_sv_training"
    tomograms = sorted(glob(os.path.join(ROOT, dataset, "*.mrc")))
    for tomo in tomograms:
        preprocess_tomogram(dataset, os.path.basename(tomo))


def preprocess_06():
    dataset = "06_hoi_wt_stem750_fm"
    tomograms = sorted(glob(os.path.join(ROOT, dataset, "*.mrc")))
    for tomo in tomograms:
        preprocess_tomogram(dataset, os.path.basename(tomo))


def preprocess_09():
    dataset = "09_stem750_66k"
    tomograms = sorted(glob(os.path.join(ROOT, dataset, "*.mrc")))
    for tomo in tomograms:
        preprocess_tomogram(dataset, os.path.basename(tomo))


def preprocess_cryo():
    fname = "vesicles-33K-L1"
    preprocess_cryo_tomogram(fname)

    fname = "vesicles-64K-LAM12"
    preprocess_cryo_tomogram(fname)


def main():
    preprocess_cryo()
    preprocess_05()
    preprocess_06()
    preprocess_09()


if __name__ == "__main__":
    main()
