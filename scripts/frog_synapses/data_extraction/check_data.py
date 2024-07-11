import napari

from common import read_volume, read_labels


def check_data(folder):
    # visualize_labels(folder)

    stack = read_volume(folder)
    vesicles = read_labels(folder, stack.shape, ["vesicles", "labeled_vesicles"])
    membrane = read_labels(folder, stack.shape, "membrane")

    v = napari.Viewer()
    v.add_image(stack)
    v.add_labels(vesicles)
    v.add_labels(membrane)
    napari.run()


def main():
    folder = "/home/pape/Work/data/silvio/frog-em/block10U3A/one"
    check_data(folder)


if __name__ == "__main__":
    main()
