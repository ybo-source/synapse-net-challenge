import os
from glob import glob

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/inner_ear_data"


def main():
    tomograms = glob(os.path.join(ROOT, "**/*.h5"), recursive=True)
    print("Number of tomograms:")
    print(len(tomograms))


main()
