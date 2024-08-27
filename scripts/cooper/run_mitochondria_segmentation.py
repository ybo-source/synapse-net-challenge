import os
from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm
from torch_em.transform.raw import standardize
from synaptic_reconstruction.inference.mitochondria import segment_mitochondria
import imageio.v3 as iio
import mrcfile


def run_mitochondria_segmentation(args):
    if os.path.exists(args.input_path) and args.input_path != "":
        img_paths = sorted(glob(os.path.join(args.input_path, "**", "*.mrc"), recursive=True))
    elif os.path.exists(args.single_image_path) and args.single_image_path != "":
        img_paths = [args.single_image_path]
    else:
        raise Exception(f"Input path not found {args.input_path}")
    # check if model path exists and remove best.pt if present
    if not os.path.exists(args.model_path):
        raise Exception(f"Model path not found {args.model_path}")
    if "best.pt" in args.model_path:
        model_path = args.model_path.replace("best.pt", "")
    else:
        model_path = args.model_path

    print(f"Processing {len(img_paths)} files")

    # get output path corresponding to input path, if not given
    if args.output_path == "":
        output_path = args.input_path
    elif not args.single_image_path == "":
        output_path = Path(args.single_image_path).parent
    else:
        output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)

    for img_path in tqdm(img_paths):
        output_path = os.path.join(output_path, os.path.splitext(os.path.basename(img_path))[0] + "_prediction.tif")
        # load img volume
        with mrcfile.open(img_path, "r") as f:
            img = f.data
        img = standardize(img)
        seg = segment_mitochondria(img, model_path)
        # save tif with imageio
        iio.imwrite(output_path, seg, compression="zlib")
        print(f"Saved segmentation to {output_path}.")


def main():
    parser = argparse.ArgumentParser(description="Segment mitochodria")
    parser.add_argument(
        "--input_path", "-i", default="",
        help="The filepath to directory containing the mitochodria data."
    )
    parser.add_argument(
        "--output_path", "-o", default="",
        help="The filepath to directory where the segmented images will be saved."
    )
    parser.add_argument(
        "--single_image_path", "-s", default="",
        help="The filepath to a single image to be segmented."
    )
    parser.add_argument(
        "--model_path", "-m", default="",
        help="The filepath to the mitochondria model."
    )

    args = parser.parse_args()

    run_mitochondria_segmentation(args)


if __name__ == "__main__":
    main()
