"""
Parse_Data.py

Utilities for loading saved plasma diagnostic data from disk,
managing per-folder index arrays used to track fit regions, and exporting
fitted plasma parameters to CSV.

Provides:
    - open_np_data: loads a single named .npy array from a save folder.
    - load_numpy_data: loads the standard set of differential/OPA
      current, voltage, and time arrays (and optionally a trigger array)
      for a given save folder.
    - initialize_indices: loads a saved per-folder index array (used to
      track fit index ranges, e.g. i_Te/i_Vp), or creates and saves a
      zero-initialized one if none exists yet.
    - save_params_as_csv: writes fitted plasma parameters (Te, Vp, n0,
      Vf, i_sat, and fit-quality metrics) to a CSV file with a fixed
      header row.
"""

import csv
import numpy as np
from pathlib import Path


def open_np_data(save_folder: Path, filename: str) -> np.ndarray:
    """
    Load a single .npy file from a save folder by base filename.

    Args:
        save_folder (pathlib.Path): Directory containing the saved arrays.
        filename (str): Base filename (without ".npy" extension).

    Returns:
        arr: The loaded array.
    """
    save_path = save_folder / f"{filename}.npy"
    return np.load(save_path)


def load_numpy_data(save_folder: Path, trigger: bool = True) -> tuple[np.ndarray, ...]:
    """
    Load the standard set of saved diagnostic arrays for a data folder.

    Loads differential and OPA current/voltage arrays, a shared time axis,
    and (optionally) a trigger array, from fixed filenames within
    `save_folder`. The single `v_opa` array is broadcast/tiled to match
    the number of rows in `i_opa`.

    Args:
        save_folder (pathlib.Path): Directory containing the saved .npy
            files (expects "currents_diff", "currents_opa",
            "voltages_diff", "voltages_opa", "x_axis", and optionally
            "trigger").
        trigger (bool): If True, also load and return the trigger array.

    Returns:
        i_diff (2D np.ndarray): all currents measured by the Differential circuit. 
            Rows correspond to time, while columns correspond to voltage.
        i_opa (2D np.ndarray): all currents measured by the Op-Amp circuit. Rows 
            correspond to time, while columns correspond to voltage.
        v_diff (2D np.ndarray): the y-axis bias voltage values, each row is the 
            corresponding voltage of a row in `i_diff`.
        v_opa (2D np.ndarray): the y-axis bias voltage values, each row is the 
            corresponding voltage of a row in `i_opa`.
        times (1D np.ndarray): the x-axis time values, each entry is the timestamp 
            of one row in `i_diff` and `i_opa.
        trigger_data (2D np.ndarray): trigger signals for the data. Only returned 
            if `trigger` is true.

    """
    i_diff = open_np_data(save_folder, "currents_diff")
    i_opa = open_np_data(save_folder, "currents_opa")
    v_diff = open_np_data(save_folder, "voltages_diff")
    v_opa = open_np_data(save_folder, "voltages_opa")
    v_opa = np.array([v_opa] * len(i_opa))
    times = open_np_data(save_folder, "x_axis")
    if not trigger:
        return i_diff, i_opa, v_diff, v_opa, times
    else:
        trigger = open_np_data(save_folder, "trigger")
        return i_diff, i_opa, v_diff, v_opa, times, trigger


def initialize_indices(folder_path: Path, entries: int, type: str) -> np.ndarray:
    """
    Load a saved index array for a data type, or create a zero-initialized
    one if none exists yet.

    Args:
        folder_path (pathlib.Path): Directory to look for/save the index
            file in.
        entries (int): Number of rows for a newly created array (ignored
            if a file already exists on disk).
        type (str): Label identifying which index file to load/create
            (used to build the filename "indices_{type}.npy").

    Returns:
        np.ndarray: An (entries, 4) int array of indices, either loaded
            from disk or newly created and saved.
    """
    indices_path = folder_path / f"indices_{type}.npy"
    if indices_path.is_file():
        indices = np.load(indices_path)
        return indices
    else:
        new_arr = np.zeros((entries, 4), dtype=int)
        np.save(indices_path, new_arr)
        return new_arr


def save_params_as_csv(
    folderpath: Path,
    description: str,
    data: list[list],
    type: str,
    num: int | str,
) -> None:
    """
    Write fitted plasma parameters to a CSV file with a fixed header row.

    Args:
        folderpath (pathlib.Path): Directory to save the CSV into.
        description (str): Label for the first header column (identifies
            what `data`'s rows represent).
        data (ist[list]): Rows of parameter values to write,
            expected to match the header
            [description, Te, r^2 (Te), Vp, RMSE (Vp), n0, Vf, i_sat].
        type (str): Label used in the output filename
            ("params{type}{num}.csv").
        num (int or str): Identifier used in the output filename.

    Returns:
        None
    """
    filepath = folderpath / f"params{type}{num}.csv"
    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [description, "Te", "r^2 (Te)", "Vp", "RMSE (Vp)", "n0", "Vf", "i_sat"]
        )
        writer.writerows(data)
