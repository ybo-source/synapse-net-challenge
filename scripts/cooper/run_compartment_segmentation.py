import h5py
import napari

from synaptic_reconstruction.inference.compartments import segment_compartments


def check_2d():

    input_path = "synapse-examples/36859_J1_66K_TS_CA3_PS_26_rec_2Kb1dawbp_crop.h5"
    model_path = "synapse-examples/compartment_model.pt"

    with h5py.File(input_path, "r") as f:
        input_ = f["raw"][279]

    seg, pred = segment_compartments(
        input_, model_path=model_path, verbose=True, return_predictions=True, scale=(0.25, 0.25),
        tiling={"tile": {"x": 512, "y": 512, "z": 1}, "halo": {"x": 32, "y": 32, "z": 1}}
    )

    v = napari.Viewer()
    v.add_image(input_)
    v.add_image(pred)
    v.add_labels(seg)
    napari.run()


def check_3d():

    input_path = "synapse-examples/36859_J1_66K_TS_CA3_PS_26_rec_2Kb1dawbp_crop.h5"
    model_path = "synapse-examples/compartment_model.pt"

    with h5py.File(input_path, "r") as f:
        input_ = f["raw"][:]

    seg, pred = segment_compartments(
        input_, model_path=model_path, verbose=True, return_predictions=True, scale=(0.25, 0.25, 0.25),
    )
    with h5py.File() as f:
        pass


def main():
    # check_2d()
    check_3d()


if __name__ == "__main__":
    main()
