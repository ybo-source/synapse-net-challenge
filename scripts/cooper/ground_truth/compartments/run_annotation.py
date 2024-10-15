import os

from elf.io import open_file
from micro_sam.sam_annotator import annotator_3d, image_folder_annotator


def run_volume_annotation(ds, name):
    checkpoint_path = "./checkpoints/compartment_model/best.pt"

    tomogram_path = f"./output/{ds}/tomograms/{name}.h5"
    embedding_path = f"./output/{ds}/embeddings/{name}.zarr"
    assert os.path.exists(embedding_path), embedding_path

    with open_file(tomogram_path, "r") as f:
        tomogram = f["data"][:]
    annotator_3d(tomogram, embedding_path=embedding_path, model_type="vit_b", checkpoint_path=checkpoint_path)


def run_image_annotation():
    checkpoint_path = "./checkpoints/compartment_model/best.pt"
    image_folder_annotator(
        input_folder="output/images", output_folder="output/annotations", pattern="*.tif",
        embedding_path="output/embeddings", model_type="vit_b",
        checkpoint_path=checkpoint_path
    )


def main():
    run_image_annotation()

    # ds = "09_stem750_66k"
    # name = "36859_J1_66K_TS_PS_05_rec_2kb1dawbp_crop"

    # ds = "cryo"
    # name = "vesicles-64K-LAM12"

    # run_annotation(ds, name)


if __name__ == "__main__":
    main()
