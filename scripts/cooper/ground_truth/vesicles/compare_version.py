import os
from glob import glob


def main():
    root_old = "/scratch-grete/projects/nim00007/data/synapse_net/train_data_cooper"
    root_new = "/projects/extern/nhr/nhr_ni/nim00007/dir.project/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer"  # noqa

    old_folders = sorted(glob(os.path.join(root_old, "0*"))) + sorted(glob(os.path.join(root_old, "1*")))
    for folder in old_folders:
        name = os.path.basename(folder)
        print(name)

        if name == "08_hcc_wt_lumsyt1nb_tem250_k2":
            name = "08_hcc_wt_lumsyt1nb_tem250_kw"
        if name == "11_tem_multi_release":
            name = "11_tem_multiple_release"
        new_folder = os.path.join(root_new, name)

        if not os.path.exists(new_folder):
            print("This data is missing for the new GT")
            print()
            continue

        n_files_old = len(glob(os.path.join(folder, "*.h5")))

        n_files_new = len(glob(os.path.join(new_folder, "**/*.mrc"), recursive=True))
        n_files_new += len(glob(os.path.join(new_folder, "**/*.rec"), recursive=True))

        if n_files_new != n_files_old:
            # print("The number of files disagrees:")
            print("Old:", n_files_old)
            print("New:", n_files_new)

        print()


if __name__ == "__main__":
    main()
