import argparse
from functools import partial

from synaptic_reconstruction.inference.cristae import segment_cristae
from synaptic_reconstruction.inference.util import inference_helper


def run_cristae_segmentation(args):
    segmentation_function = partial(segment_cristae, model_path=args.model_path)
    inference_helper(
        args.input_path,
        args.output_path,
        segmentation_function,
        extra_input_path=args.second_input_path
        )


def main():
    parser = argparse.ArgumentParser(description="Segment mitochodria")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to mrc file or directory containing the mitochodria data."
    )
    parser.add_argument(
        "--second_input_path", "-s", required=True,
        help=""
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmented images will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True,
        help="The filepath to the mitochondria model."
    )

    args = parser.parse_args()
    run_cristae_segmentation(args)


if __name__ == "__main__":
    main()
