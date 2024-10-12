from micro_sam.precompute_state import precompute_state


def preprocess_tomogram():

    precompute_state(
        input_path="",
        output_path="",
        model_type="vit_b",
        key="",
        checkpoint_path="",
        ndim=3,
    )


def main():
    preprocess_tomogram()


if __name__ == "__main__":
    main()
