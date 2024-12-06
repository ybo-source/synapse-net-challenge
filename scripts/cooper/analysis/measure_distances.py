import os
from glob import glob

import h5py
import pandas as pd
from tqdm import tqdm

from synapse_net.distance_measurements import measure_segmentation_to_object_distances


RESOLUTION = (1.554,) * 3


def measure_distances(path, ds, fname):
    with h5py.File(path, "r") as f:
        if "thin_az_corrected" in f:
            print("Loading corrected az!")
            az = f["thin_az_corrected"][:]
        else:
            az = f["/az_thin_proofread"][:]
        vesicles = f["vesicles"][:]
    distances, _, _, _ = measure_segmentation_to_object_distances(vesicles, az, resolution=RESOLUTION)
    return distances


def main():
    ratings = pd.read_excel("quality_ratings/az_quality_clean_FM.xlsx")

    dataset_results = {
        ds: {"CTRL": {}, "DKO": {}} for ds in pd.unique(ratings["Dataset"])
    }

    restrict_to_good_azs = False

    paths = sorted(glob("proofread_az/**/*.h5", recursive=True))
    for path in tqdm(paths):

        ds, fname = os.path.split(path)
        ds = os.path.split(ds)[1]
        fname = os.path.splitext(fname)[0]

        category = "CTRL" if "CTRL" in fname else "DKO"

        if restrict_to_good_azs:
            rating = ratings[
                (ratings["Dataset"] == ds) & (ratings["Tomogram"] == fname)
            ]["Rating"].values[0]
            if rating != "Good":
                continue

        distances = measure_distances(path, ds, fname)
        dataset_results[ds][category][fname] = distances

    for ds, categories in dataset_results.items():
        for category, tomogram_data in categories.items():
            sorted_data = dict(sorted(tomogram_data.items()))  # Sort by tomogram names
            result_df = pd.DataFrame.from_dict(sorted_data, orient='index').transpose()

            os.makedirs("./results/distances", exist_ok=True)
            output_path = os.path.join("./results/distances", f"distance_analysis_for_{ds}_{category}.csv")

            # Save the DataFrame to CSV
            result_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
