import argparse
from functools import partial

from synaptic_reconstruction.inference.cristae import segment_cristae
from synaptic_reconstruction.inference.util import inference_helper, parse_tiling


def run_cristae_segmentation(args):
    tiling = parse_tiling(args.tile_shape, args.halo)
    segmentation_function = partial(segment_cristae, model_path=args.model_path, verbose=False, tiling=tiling)
    inference_helper(
        args.input_path,
        args.output_path,
        segmentation_function,
        extra_input_path=args.mito_segmentation_path,
        force=args.force,
    )


def main():
    parser = argparse.ArgumentParser(description="Segment cristae in EM Tomograms.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to mrc file or directory containing the tomogram data."
    )
    parser.add_argument(
        "--mito_segmentation_path", "-s", required=True,
        help="The filepath to the tif file or directory containing the mito segmentation."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentation will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True, help="The filepath to the cristae model."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Whether to over-write already present segmentation results."
    )
    parser.add_argument(
        "--tile_shape", type=int, nargs=3,
        help="The tile shape for prediction. Lower the tile shape if GPU memory is insufficient."
    )
    parser.add_argument(
        "--halo", type=int, nargs=3,
        help="The halo for prediction. Increase the halo to minimize boundary artifacts."
    )

    args = parser.parse_args()
    run_cristae_segmentation(args)


if __name__ == "__main__":
    main()
