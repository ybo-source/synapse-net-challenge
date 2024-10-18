import os
from glob import glob

import imageio.v3 as imageio

from elf.io import open_file
from skimage.transform import rescale
from tqdm import tqdm

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer"  # noqa
ROOT_CRYO = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/fernandez-busnadiego/vesicle_gt/v1"  # noqa
OUTPUT_IMAGES = "./output/images"


def process_tomogram(tomo_path, scale, tomo_key="data"):
    with open_file(tomo_path, "r") as f:
        tomo = f[tomo_key][:]

    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    offset = len(glob(os.path.join(OUTPUT_IMAGES, "*.tif")))

    print("Resizing tomogram ...")
    tomo = rescale(tomo, scale, preserve_range=True).astype(tomo.dtype)

    z_max = tomo.shape[0]
    slices = [z_max // 2, z_max // 4, 3 * z_max // 4]

    for i, z in enumerate(slices):
        im = tomo[z]
        idx = i + offset
        out_path = os.path.join(OUTPUT_IMAGES, f"image_{idx:05}.tif")
        imageio.imwrite(out_path, im, compression="zlib")


def preprocess_05():
    scale = (0.25, 0.25, 0.25)
    dataset = "05_stem750_sv_training"
    tomograms = sorted(glob(os.path.join(ROOT, dataset, "*.mrc")))
    for tomo in tqdm(tomograms):
        process_tomogram(tomo, scale)


def preprocess_06():
    scale = (0.25, 0.25, 0.25)
    dataset = "06_hoi_wt_stem750_fm"
    tomograms = sorted(glob(os.path.join(ROOT, dataset, "*.mrc")))
    for tomo in tqdm(tomograms):
        process_tomogram(tomo, scale)


def preprocess_09():
    scale = (0.25, 0.25, 0.25)
    dataset = "09_stem750_66k"
    tomograms = sorted(glob(os.path.join(ROOT, dataset, "*.mrc")))
    for tomo in tqdm(tomograms):
        process_tomogram(tomo, scale)


def preprocess_cryo():
    scale = (0.5, 0.5, 0.5)
    tomograms = sorted(glob(os.path.join(ROOT_CRYO, "*.h5")))
    for tomo in tqdm(tomograms):
        process_tomogram(tomo, scale, tomo_key="raw")


def precompute_state():
    from micro_sam.util import get_sam_model
    from micro_sam.precompute_state import _precompute_state_for_files

    images = sorted(glob(os.path.join(OUTPUT_IMAGES, "*.tif")))
    embedding_path = "./output/embeddings"

    predictor = get_sam_model(model_type="vit_b", checkpoint_path="./checkpoints/compartment_model/best.pt")
    precompute_amg_state = False
    decoder = None

    _precompute_state_for_files(
        predictor, images, embedding_path, ndim=2, tile_shape=None, halo=None,
        precompute_amg_state=precompute_amg_state, decoder=decoder,
    )


def main():
    # preprocess_05()
    # preprocess_06()
    # preprocess_09()
    # preprocess_cryo()
    precompute_state()


if __name__ == "__main__":
    main()
