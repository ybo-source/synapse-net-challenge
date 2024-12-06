import h5py
import napari


def check_gt(path):

    with h5py.File(path, "r") as f:
        tomo = f["raw"][:]
        vesicles = f["labels/vesicles"][:]
        mask = f["labels/mask"][:]

    v = napari.Viewer()
    v.add_image(tomo)
    v.add_labels(vesicles)
    v.add_labels(mask)
    napari.run()


def main():
    gt_path1 = "/home/pape/Work/data/fernandez-busnadiego/vesicle_gt/v2/vesicles-33K-L1.h5"
    check_gt(gt_path1)

    gt_path2 = "/home/pape/Work/data/fernandez-busnadiego/vesicle_gt/v2/vesicles-64K-LAM12.h5"
    check_gt(gt_path2)


if __name__ == "__main__":
    main()
