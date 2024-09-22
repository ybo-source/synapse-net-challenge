import argparse
import json
import os
from glob import glob

import numpy as np
from synaptic_reconstruction.imod import get_label_names
from tqdm import tqdm

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer"  # noqa


# Filter label names that just contain numbers or contain 'wimp'.
# These are from mtk for automatic distance measurements and not relevant for us.
def _filter_names(label_names):

    def _filter(name):
        try:
            int(name)
            return True
        except ValueError:
            pass

        if "wimp" in name.lower():
            return True

        return False

    label_names = [name for name in label_names if not _filter(name)]
    return label_names


def extract_annotations_for_dataset(dataset_folder):
    out_path = os.path.join(dataset_folder, "imod_annotations.json", )
    if os.path.exists(out_path):
        with open(out_path) as f:
            annotations = json.load(f)
        return annotations

    annotations = {}
    mod_files = glob(os.path.join(dataset_folder, "**/*.mod"), recursive=True)
    for f in mod_files:
        label_names = get_label_names(f)
        label_names = list(label_names.values())
        # Filter out irrelevant label names.
        label_names = _filter_names(label_names)
        fname = os.path.relpath(f, dataset_folder)
        annotations[fname] = label_names

    with open(out_path, "w") as f:
        json.dump(annotations, f)
    return annotations


def extract_all_annotations():
    all_annotations = {}
    dataset_folders = sorted(glob(os.path.join(ROOT, "*")))
    for ds_folder in tqdm(dataset_folders, desc="Extract annotations from IMOD"):
        annotations = extract_annotations_for_dataset(ds_folder)
        ds_name = os.path.basename(ds_folder)
        all_annotations[ds_name] = annotations
    return all_annotations


def find_queries(all_annotations, queries, exclude):
    queries_ = [q.lower() for q in queries]

    matches = []
    print()
    print("The following matches were found for the queries", queries)
    for dataset, annotations in all_annotations.items():
        n_tomograms = len(annotations)
        hits = []
        for tomo_name, imod_names in annotations.items():
            this_matches = [
                name for name in imod_names if any(q in name.lower() for q in queries_)
            ]
            if exclude is not None:
                [name for name in this_matches if name not in exclude]
            if this_matches:
                hits.append(tomo_name)
                matches.extend(this_matches)

        print("Dataset", dataset, ":")
        print("Found", len(hits), "/", n_tomograms, "tomograms with a match.")

    print()
    print("Found the following matching terms (with number of occurences)")
    unique_matches, number_of_matches = np.unique(matches, return_counts=True)
    for match, count in zip(unique_matches, number_of_matches):
        print(match, ":", count)


def print_all_annotations(all_annotations):
    all_names = []
    for this_annotations in all_annotations.values():
        all_names.extend([name.lower() for names in this_annotations.values() for name in names])
    all_names = np.unique(all_names)
    print("The following unique annotation names are in the IMOD files:")
    for name in all_names:
        print(name)


def main():
    all_annotations = extract_all_annotations()

    # TODO implement additional command line arguments for
    # saving extract tomogram lists that contain queries
    parser = argparse.ArgumentParser(
        description="List annotations present in IMOD annotations shared by the Cooper lab."
    )
    parser.add_argument(
        "-a", "--all", action="store_true",
        help="Print all available annotation names in the IMOD fails."
    )
    parser.add_argument(
        "-q", "--queries", nargs="+",
        help="Pass a query of annotation names and list the datasets containing corresponding annotations."
    )
    parser.add_argument(
        "-x", "--exclude", nargs="+",
        help="Pass a list of names to exclude from the query."
    )
    args = parser.parse_args()

    if args.all:
        print_all_annotations(all_annotations)

    if not args.queries:
        return "No queries were passed, doing nothing."
    find_queries(all_annotations, args.queries, args.exclude)


if __name__ == "__main__":
    main()
