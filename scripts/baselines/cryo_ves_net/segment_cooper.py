from common import apply_cryo_vesnet


def main():
    input_folder = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets"  # noqa
    output_folder = "./cryo-vesnet-test2"

    # TODO determine the correct resolution (in angstrom) for each dataset
    resolution = (10, 10, 10)
    apply_cryo_vesnet(input_folder, output_folder, pattern="*.h5", input_key="raw", resolution=resolution, nested=True)


if __name__ == "__main__":
    main()
