"""Parses Labview Waveform Data from CSV format and saves as .npy.
Saves channels 1, 2, and 3.
Also includes functions that deal with the slicing and stuff
"""

import csv
import numpy as np


def parse_filename(name):
    split = name.split("-")
    gas = split[0]
    power = int(split[1][:-1])
    ice = "No" not in split[2]
    run = int(split[3])

    return gas, power, ice, run


def open_np_data(save_folder, filename):
    save_path = save_folder / f"{filename}.npy"
    return np.load(save_path)


def load_numpy_data(save_folder, trigger=True):
    i_diff = open_np_data(save_folder, "currents_diff")
    i_opa = open_np_data(save_folder, "currents_opa")
    v_diff = open_np_data(save_folder, "voltages_diff")
    v_opa = open_np_data(save_folder, "voltages_opa")
    v_opa = np.array([v_opa]*len(i_opa))
    times = open_np_data(save_folder, "x_axis")
    if not trigger:
        return i_diff, i_opa, v_diff, v_opa, times
    else:
        trigger = open_np_data(save_folder, "trigger")
        return i_diff, i_opa, v_diff, v_opa, times, trigger


def initialize_indices(folder_path, entries, type):
    indices_path = folder_path / f"indices_{type}.npy"
    if  indices_path.is_file():
        indices = np.load(indices_path)
        print(indices)
        return indices
    else:
        new_arr = np.zeros((entries, 4), dtype=int)
        np.save(indices_path, new_arr)
        return new_arr


def save_params_as_csv(folderpath, description, data, type, num):
    filepath = folderpath / f"params{type}{num}.csv"
    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        #[descriptor, Te, Te_r2, Vp, Vp_r2, n0, Vf, i_sat]
        writer.writerow([description, "Te", "r^2 (Te)", "Vp", "RMSE (Vp)", "n0", "Vf", "i_sat"])
        writer.writerows(data)
