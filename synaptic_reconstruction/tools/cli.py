import argparse
from functools import partial

from .util import run_segmentation, get_model
from ..imod.to_imod import export_helper, write_segmentation_to_imod_as_points, write_segmentation_to_imod
from ..inference.util import inference_helper, parse_tiling


def imod_point_cli():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--segmentation_path", "-s", required=True,
        help="The filepath to the tif file or the directory containing the segmentations."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--segmentation_key", "-k", help=""
    )
    parser.add_argument(
        "--min_radius", type=float, default=10.0, help=""
    )
    parser.add_argument(
        "--radius_factor", type=float, default=1.0, help="",
    )
    parser.add_argument(
        "--force", action="store_true", help="",
    )
    args = parser.parse_args()

    export_function = partial(
        write_segmentation_to_imod_as_points,
        min_radius=args.min_radius,
        radius_factor=args.radius_factor,
    )

    export_helper(
        input_path=args.input_path,
        segmentation_path=args.segmentation_path,
        output_root=args.output_path,
        export_function=export_function,
        force=args.force,
        segmentation_key=args.segmentation_key,
    )


def imod_object_cli():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--segmentation_path", "-s", required=True,
        help="The filepath to the tif file or the directory containing the segmentations."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--segmentation_key", "-k", help=""
    )
    parser.add_argument(
        "--force", action="store_true", help="",
    )
    args = parser.parse_args()
    export_helper(
        input_path=args.input_path,
        segmentation_path=args.segmentation_path,
        output_root=args.output_path,
        export_function=write_segmentation_to_imod,
        force=args.force,
        segmentation_key=args.segmentation_key,
    )


# TODO: handle kwargs
# TODO: add custom model path
# TODO: enable autoscaling from input resolution
def segmentation_cli():
    parser = argparse.ArgumentParser(description="Run segmentation.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    # TODO: list the availabel models here by parsing the keys of the model registry
    parser.add_argument(
        "--model", "-m", required=True, help="The model type."
    )
    parser.add_argument(
        "--mask_path", help="The filepath to a tif file with a mask that will be used to restrict the segmentation."
        "Can also be a directory with tifs if the filestructure matches input_path."
    )
    parser.add_argument("--input_key", "-k", required=False)
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
        "--segmentation_key", "-s", help=""
    )
    # TODO enable autoscaling
    parser.add_argument(
        "--scale", type=float, default=None, help=""
    )
    args = parser.parse_args()

    model = get_model(args.model)
    tiling = parse_tiling(args.tile_shape, args.halo)
    scale = None if args.scale is None else 3 * (args.scale,)

    segmentation_function = partial(
        run_segmentation, model=model, model_type=args.model, verbose=False, tiling=tiling, scale=scale
    )
    inference_helper(
        args.input_path, args.output_path, segmentation_function,
        mask_input_path=args.mask_path, force=args.force, data_ext=args.data_ext,
        output_key=args.segmentation_key,
    )
