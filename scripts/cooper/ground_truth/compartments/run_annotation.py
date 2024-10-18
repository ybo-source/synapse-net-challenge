import os
from glob import glob
from pathlib import Path

from elf.io import open_file
from micro_sam.sam_annotator import annotator_3d, image_folder_annotator


def run_volume_annotation(ds, name):
    checkpoint_path = "./checkpoints/compartment_model_v2/best.pt"

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


def annotate_cryo():
    ds = "cryo"
    name = "vesicles-33K-L1"
    run_volume_annotation(ds, name)


def _series_annotation(ds):
    # name = "upSTEM750_36859_J2_TS_SP_001_rec_2kb1dawbp_crop"
    images = glob(f"./output/{ds}/tomograms/*.h5")
    for image in images:
        name = Path(image).stem
        seg_path = f"./output/{ds}/segmentations/{name}.tif"
        print("Run segmentation for:", ds, name)
        if os.path.exists(seg_path):
            print("Skipping", ds, name, "because it is already segmented.")
            continue
        run_volume_annotation(ds, name)


def annotate_05():
    ds = "05_stem750_sv_training"
    _series_annotation(ds)


def annotate_06():
    ds = "06_hoi_wt_stem750_fm"
    _series_annotation(ds)


def main():
    # run_image_annotation()
    # annotate_cryo()
    # annotate_05()
    annotate_06()


if __name__ == "__main__":
    main()
