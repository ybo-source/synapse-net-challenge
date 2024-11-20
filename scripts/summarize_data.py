import numpy as np
import pandas as pd


az_train = pd.read_excel("data_summary/active_zone_training_data.xlsx")
compartment_train = pd.read_excel("data_summary/compartment_training_data.xlsx")
vesicle_train = pd.read_excel("data_summary/vesicle_training_data.xlsx")
vesicle_da = pd.read_excel("data_summary/vesicle_domain_adaptation_data.xlsx", sheet_name="cryo")


def training_resolutions():
    res_az = np.round(az_train["resolution"].mean(), 2)
    res_compartment = np.round(compartment_train["resolution"].mean(), 2)
    res_cryo = np.round(vesicle_da["resolution"].mean(), 2)
    res_vesicles = np.round(vesicle_train["resolution"].mean(), 2)

    print("Training resolutions for models:")
    print("active_zone:", res_az)
    print("compartments:", res_compartment)
    # TODO
    print("mitochondria:", 1.0)
    print("vesicles_2d:", res_vesicles)
    print("vesicles_3d:", res_vesicles)
    print("vesicles_cryo:", res_cryo)


training_resolutions()
