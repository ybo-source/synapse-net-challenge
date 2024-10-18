from common import apply_cryo_vesnet


def main():
    input_folder = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/fernandez-busnadiego/vesicle_gt/v2"  # noqa
    output_folder = "./cryo-vesnet-test"

    # Resolution in Angstrom in XYZ
    # The two tomograms have a different resolution.
    resolution = {
        "vesicles-33K-L1": (14.6, 14.6, 14.6),
        "vesicles-64K-LAM12": (7.56, 7.56, 7.56),
    }
    apply_cryo_vesnet(
        input_folder, output_folder,
        pattern="*.h5", input_key="raw",
        mask_folder=input_folder, mask_key="/labels/mask",
        resolution=resolution
    )


if __name__ == "__main__":
    main()
