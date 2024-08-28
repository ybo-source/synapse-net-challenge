import argparse
from functools import partial

from synaptic_reconstruction.inference.vesicles import segment_vesicles
from synaptic_reconstruction.inference.util import inference_helper


def run_vesicle_segmentation(args):
    segmentation_function = partial(segment_vesicles, model_path=args.model_path, verbose=False)
    inference_helper(args.input_path, args.output_path, segmentation_function, force=args.force)


def main():
    parser = argparse.ArgumentParser(description="Segment vesicles")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmented images will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True,
        help="The filepath to the vesicle model."
    )
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    run_vesicle_segmentation(args)


if __name__ == "__main__":
    main()
