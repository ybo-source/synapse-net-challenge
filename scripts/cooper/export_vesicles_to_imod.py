import argparse
from functools import partial

from synapse_net.imod.to_imod import export_helper, write_segmentation_to_imod_as_points


def export_vesicles_to_imod(args):
    export_function = partial(
        write_segmentation_to_imod_as_points, min_radius=args.min_radius, radius_factor=args.increase_radius
    )
    export_helper(args.input_path, args.segmentation_path, args.output_path, export_function, force=args.force)


def main():
    parser = argparse.ArgumentParser(description="Export vesicle segmentation to IMOD.")
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="The filepath to the mrc file or the directory containing the data."
    )
    parser.add_argument(
        "-s", "--segmentation_path", required=True,
        help="The filepath to the tif file or the directory containing the corresponding vesicle segmentations."
    )
    parser.add_argument(
        "-o", "--output_path", required=True,
        help="The filepath to the directory where the exported mod files will be saved."
    )
    parser.add_argument(
        "--min_radius", type=float, default=10,
        help="The minimum export radius in nm."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Whether to over-write already present export results."
    )
    parser.add_argument(
        "--increase_radius", type=float, default=1.3,
        help="The factor to increase the radius of the exported vesicles.",
    )
    args = parser.parse_args()
    export_vesicles_to_imod(args)


if __name__ == "__main__":
    main()
