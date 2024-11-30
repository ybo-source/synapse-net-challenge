import os
from glob import glob

import numpy as np
import pandas as pd


def filter_sizes_by_distance(size_table, distance_table, out_dir, max_distance=100):
    fname = os.path.basename(size_table)
    print("Filtering vesicles for", fname)

    size_table = pd.read_csv(size_table)
    distance_table = pd.read_csv(distance_table)
    assert (size_table.columns == distance_table.columns).all()
    out_columns = {}
    n_tot, n_filtered = 0, 0
    all_values = []
    for col_name in size_table.columns:
        size_values = size_table[col_name].values
        distance_values = distance_table[col_name].values
        size_values, distance_values = (
            size_values[np.isfinite(size_values)],
            distance_values[np.isfinite(distance_values)]
        )
        assert len(size_values) == len(distance_values)
        n_tot += len(size_values)
        mask = distance_values < max_distance
        out_columns[col_name] = size_values[mask]
        n_filtered += mask.sum()
        all_values.extend(size_values[mask].tolist())

    print("Total number of vesicles:", n_tot)
    print("Number of vesicles after filtering:", n_filtered)
    print("Average diameter:", np.mean(all_values))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    filtered_sizes = pd.DataFrame.from_dict(out_columns, orient='index').transpose()
    filtered_sizes.to_csv(out_path, index=False)


def main():
    size_tables = sorted(glob("./results/diameters/*.csv"))
    distance_tables = sorted(glob("./results/distances/*.csv"))
    for size_tab, distance_tab in zip(size_tables, distance_tables):
        filter_sizes_by_distance(size_tab, distance_tab, "./results/filtered_diameters")


if __name__ == "__main__":
    main()
