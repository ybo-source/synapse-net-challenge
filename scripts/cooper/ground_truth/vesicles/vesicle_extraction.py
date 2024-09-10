import os
from glob import glob
from pathlib import Path

from synaptic_reconstruction.ground_truth import extract_vesicle_training_data

ROOT = "/projects/extern/nhr/nhr_ni/nim00007/dir.project/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer"  # noqa
OUT_ROOT = "/projects/extern/nhr/nhr_ni/nim00007/dir.project/data/synaptic-reconstruction/cooper/extracted/20240909_cp_datatransfer"  # noqa


def extract_01():
    name = "01_hoi_maus_2020_incomplete"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem

        # this filepattern matches the files in A_WT_SC_DIV_14
        imod_name = fname.rstrip("_SP").split("_")
        imod_name = "_".join(imod_name[:-1] + ["mtk"] + imod_name[-1:])
        imod_path = os.path.join(gt_folder, folder, f"{imod_name}.mod")

        # try matching the closest filename
        if not os.path.exists(imod_path):

            if str(folder).startswith("E"):
                match_name = "_".join(fname.replace("_MS", "").split("_")[:6])
            elif fname.startswith("WT_Unt_05646_D1_3.1_DIV16") and str(folder).startswith("B"):
                match_name = "WT_Unt_05646_D1_3.1_DIV16"
            else:
                match_name = fname

            imod_file_names = [os.path.basename(xx) for xx in sorted(glob(os.path.join(gt_folder, folder, "*.mod")))]
            matching_name = [name for name in imod_file_names if (name.startswith(match_name) and "AZ" not in name)]
            # matching_name = [name for name in imod_file_names if name.startswith(match_name)]
            if len(matching_name) != 1:
                if len(matching_name) == 2:
                    if fname.endswith("05_MS"):
                        assert matching_name[0] == "M13_CTRL_22723_O2_05_DIV29_mtk_05.mod"
                        imod_path = os.path.join(gt_folder, folder, matching_name[0])
                        return imod_path
                    elif fname.endswith("5.2_MS"):
                        assert matching_name[1] == "M13_CTRL_22723_O2_05_DIV29_mtk_5.2.mod"
                        imod_path = os.path.join(gt_folder, folder, matching_name[1])
                        return imod_path
                    elif fname.startswith("WT_MF_DIV28_06_MS_09204_F1"):  # use just one of the AZs
                        imod_path = os.path.join(gt_folder, folder, matching_name[0])
                        return imod_path
                raise RuntimeError(
                    f"Did not find a matching annotation for {relative_file_path}, matching with {match_name}. "
                    f"Found: {matching_name}"
                )
            imod_path = os.path.join(gt_folder, folder, matching_name[0])

        assert os.path.exists(imod_path), imod_path
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, skip_no_labels=False,
        exclude_label_patterns=["wimp", "contact_sites", "Exo/endo", "ExoEndo"]
    )


def main():
    extract_01()


if __name__ == "__main__":
    main()
