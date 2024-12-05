import os

import h5py
import napari

from tqdm import tqdm


def check_proofread(raw_path, seg_path):
    with h5py.File(seg_path, "r") as f:
        seg1 = f["labels_pp/thin_az"][:]
        seg2 = f["labels_pp/filtered_az"][:]
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:]

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg1)
    v.add_labels(seg2)
    napari.run()


def main():
    # FIXME something wrong in the zenodo upload
    root_raw = "/home/pape/Work/my_projects/synaptic-reconstruction/scripts/data_summary/for_zenodo/synapse-net/active_zones/train"  # noqa
    root_seg = "./postprocessed_AZ"

    test_tomograms = {
        "01": [
            "WT_MF_DIV28_01_MS_09204_F1.h5", "WT_MF_DIV14_01_MS_B2_09175_CA3.h5", "M13_CTRL_22723_O2_05_DIV29_5.2.h5", "WT_Unt_SC_09175_D4_05_DIV14_mtk_05.h5",  # noqa
            "20190805_09002_B4_SC_11_SP.h5", "20190807_23032_D4_SC_01_SP.h5", "M13_DKO_22723_A1_03_DIV29_03_MS.h5", "WT_MF_DIV28_05_MS_09204_F1.h5", "M13_CTRL_09201_S2_06_DIV31_06_MS.h5", # noqa
            "WT_MF_DIV28_1.2_MS_09002_B1.h5", "WT_Unt_SC_09175_C4_04_DIV15_mtk_04.h5",   "M13_DKO_22723_A4_10_DIV29_10_MS.h5",  "WT_MF_DIV14_3.2_MS_D2_09175_CA3.h5",  # noqa
               "20190805_09002_B4_SC_10_SP.h5", "M13_CTRL_09201_S2_02_DIV31_02_MS.h5", "WT_MF_DIV14_04_MS_E1_09175_CA3.h5", "WT_MF_DIV28_10_MS_09002_B3.h5",   "WT_Unt_SC_05646_D4_02_DIV16_mtk_02.h5",   "M13_DKO_22723_A4_08_DIV29_08_MS.h5",  "WT_MF_DIV28_04_MS_09204_M1.h5",   "WT_MF_DIV28_03_MS_09204_F1.h5",   "M13_DKO_22723_A1_05_DIV29_05_MS.h5",  # noqa
            "WT_Unt_SC_09175_C4_06_DIV15_mtk_06.h5",  "WT_MF_DIV28_09_MS_09002_B3.h5", "20190524_09204_F4_SC_07_SP.h5",
               "WT_MF_DIV14_02_MS_C2_09175_CA3.h5",    "M13_DKO_23037_K1_01_DIV29_01_MS.h5",  "WT_Unt_SC_09175_E2_01_DIV14_mtk_01.h5", "20190807_23032_D4_SC_05_SP.h5",   "WT_MF_DIV14_01_MS_E2_09175_CA3.h5",   "WT_MF_DIV14_03_MS_B2_09175_CA3.h5",   "M13_DKO_09201_O1_01_DIV31_01_MS.h5",  "M13_DKO_09201_U1_04_DIV31_04_MS.h5",  # noqa
            "WT_MF_DIV14_04_MS_E2_09175_CA3_2.h5",   "WT_Unt_SC_09175_D5_01_DIV14_mtk_01.h5",
            "M13_CTRL_22723_O2_05_DIV29_05_MS_.h5",  "WT_MF_DIV14_02_MS_B2_09175_CA3.h5", "WT_MF_DIV14_01.2_MS_D1_09175_CA3.h5",  # noqa
        ],
        "12": ["20180305_09_MS.h5", "20180305_04_MS.h5", "20180305_08_MS.h5",
               "20171113_04_MS.h5", "20171006_05_MS.h5", "20180305_01_MS.h5"],
    }

    for ds, test_tomos in test_tomograms.items():
        ds_name_raw = "single_axis_tem" if ds == "01" else "chemical-fixation"
        ds_name_seg = "01_hoi_maus_2020_incomplete" if ds == "01" else "12_chemical_fix_cryopreparation"
        for tomo in tqdm(test_tomos, desc=f"Proofread {ds}"):
            raw_path = os.path.join(root_raw, ds_name_raw, tomo)
            seg_path = os.path.join(root_seg, ds_name_seg, tomo)
            check_proofread(raw_path, seg_path)


main()
