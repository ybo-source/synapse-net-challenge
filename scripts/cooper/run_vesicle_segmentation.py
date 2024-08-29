import argparse
from functools import partial

from synaptic_reconstruction.inference.vesicles import segment_vesicles
from synaptic_reconstruction.inference.util import inference_helper, parse_tiling


def run_vesicle_segmentation(args):
    tiling = parse_tiling(args.tile_shape, args.halo)
    segmentation_function = partial(
        segment_vesicles, model_path=args.model_path, verbose=False, tiling=tiling,
        exclude_boundary=not args.include_boundary
    )
    inference_helper(args.input_path, args.output_path, segmentation_function, force=args.force, data_ext=args.data_ext)


def main():
    parser = argparse.ArgumentParser(description="Segment vesicles in EM tomograms.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True, help="The filepath to the vesicle model."
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
    parser.add_argument(
        "--data_ext", default=".mrc", help="The extension of the tomogram data. By default .mrc."
    )
    parser.add_argument(
        "--include_boundary", action="store_true",
        help="Include vesicles that touch the top / bottom of the tomogram. By default these are excluded."
    )

    args = parser.parse_args()
    run_vesicle_segmentation(args)


if __name__ == "__main__":
    main()
