import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import get_all_measurements, get_measurements_with_annotation


def for_tomos_with_annotation():
    manual_assignments, automatic_assignments = get_measurements_with_annotation()
    breakpoint()


# def for_all_tomos():
#     automatic_assignments = get_all_measurements()


def main():
    for_tomos_with_annotation()
    # for_all_tomos()


if __name__ == "__main__":
    main()
