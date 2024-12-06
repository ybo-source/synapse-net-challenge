from elf.io import open_file


def test_export():
    from synapse_net.imod.to_imod import write_segmentation_to_imod_as_points
    from subprocess import run

    mrc_path = "20241108_3D_Imig_DATA_2014/!_M13DKO_TOMO_DATA_Imig2014_mrc-mod-FM/A_M13DKO_080212_CTRL4.8_crop/A_M13DKO_080212_CTRL4.8_crop.mrc"  # noqa
    seg_path = "imig_data/Munc13DKO/A_M13DKO_080212_CTRL4.8_crop.h5"
    out_path = "exported_vesicles.mod"

    with open_file(seg_path, "r") as f:
        seg = f["vesicles/segment_from_combined_vesicles"][:]

    # !!!! 0.7
    write_segmentation_to_imod_as_points(
        mrc_path, seg, out_path, min_radius=10, radius_factor=0.7
    )
    run(["imod", mrc_path, out_path])


test_export()
