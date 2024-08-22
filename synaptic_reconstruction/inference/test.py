import os
import argparse
import h5py
from mitochondria import segment_mitochondria


def main(args):
    # load test pred and run it through segmentation funciton

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--data_dir", type=str, default="", help="Path to the data directory")
    args = parser.parse_args()

    main(args)