"""
Plot_Curves.py

Plotting utilities for I_V curve analysis and experiment folder-name parsing.

Provides:
    - graph_electrontemp: plots ln(current) vs. bias voltage with electron
      temperature (Te) and plasma potential (Vp) fit overlays.
    - graph_raw_curve: plots raw current vs. bias voltage with an ion
      saturation current reference and floating potential (Vf) marker.
    - render_initial_IV_curve / render_initial_raw_curve: construct
      placeholder figures (dummy data) for these two plot types before
      real measurement data is loaded, prompting the user to choose a
      data folder.
    - parse_folder_name: parses a hyphen-delimited experiment folder name
      into a human-readable label, with multiple output formats.

Module-level constants:
    MKSTYLE (str): Marker style string ("." ) used across plotting
        functions for consistent point styling.
"""
import numpy as np
import matplotlib.pyplot as plt

MKSTYLE = "."
 

def render_initial_IV_curve() -> tuple[plt.Figure, plt.Axes, list[plt.Artist]]:
    """
    Initialize a placeholder IV (current-voltage) curve plot before real
    data is loaded.

    Creates a 3x3 inch matplotlib figure and draws a curve via
    graph_electrontemp using dummy data The title prompts
    the user to select a data folder.

    Returns:
        fig (matplotlib.figure.Figure): The created figure.
        ax (matplotlib.axes.Axes): The axes containing the plot.
        artists (list): Artist objects for the plotted curve(s) and title,
            grouped so they can be removed or updated together later.
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    t = np.arange(0, 3, .01)
    artists = graph_electrontemp(t, t, 2, t, t, (0, 1), (0, 1))

    ax.set_xlabel("Bias Voltage (V)")
    ax.set_ylabel("Current (ln[A])")
    title_artist = ax.set_title("Choose Data Folder", fontsize="small")
    artists.append(title_artist)
    plt.tight_layout()

    return fig, ax, artists


def render_initial_raw_curve()-> tuple[plt.Figure, plt.Axes, list[plt.Artist]]:
    """
    Initialize a placeholder raw current-voltage plot before real data is
    loaded.

    Creates a 3x3 inch matplotlib figure with a horizontal reference line
    at y=0, then draws a curve via graph_raw_curve using dummy
    voltage/current data. The title prompts the user to select a data folder.

    Returns:
        fig (matplotlib.figure.Figure): The created figure.
        ax (matplotlib.axes.Axes): The axes containing the plot.
        artists (list): Artist objects for the plotted curve(s) and title,
            grouped so they can be removed or updated together later.
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axhline(0, color="gray", linewidth=1)

    t = np.arange(0, 3, .01)
    artists = graph_raw_curve(t, t, 2, 1)

    ax.set_xlabel("Bias Voltage (V)")
    ax.set_ylabel("Current (mA)")
    title_artist = ax.set_title("Choose Data Folder", fontsize="small")
    artists.append(title_artist)
    
    plt.tight_layout()

    return fig, ax, artists



def graph_electrontemp(
    x_voltage: np.ndarray,
    ln_current: np.ndarray,
    Vp: float,
    Te_fit: np.ndarray,
    Vp_fit: np.ndarray,
    i_Te: tuple[int, int],
    i_Vp: tuple[int, int],
    ) -> list[plt.Line2D]:
    """
    Plot a Langmuir-probe-style ln(current) vs. bias voltage curve, annotated
    with electron-temperature (Te) and plasma-potential (Vp) fit regions.

    Plots the raw ln(current) curve, overlays two fit lines (Te_fit, Vp_fit)
    across the full voltage range, highlights the data subranges used for
    each fit as scatter points (indexed by i_Te and i_Vp), and marks the
    plasma potential point Vp on the Te_fit line.

    Args:
        x_voltage (1D array-like): Bias voltage values (x-axis).
        ln_current (1D array-like): Natural log of measured current (y-axis).
        Vp (float): Plasma potential.
        Te_fit (1D array-like): Fitted curve values used for the electron
            temperature fit (same length as x_voltage).
        Vp_fit (1D array-like): Fitted curve values used for the plasma
            potential fit (same length as x_voltage).
        i_Te (tuple[int, int]): Start/end indices into x_voltage/ln_current,
            marking the data range used for the Te fit.
        i_Vp (tuple[int, int]): Start/end indices into x_voltage/ln_current,
            marking the data range used for the Vp fit.

    Returns:
        list: The six matplotlib Line2D artists created, in order:
            [I-V curve, Te fit line, I-V curve points included in Te fit, Vp fit
            line, I-V curve points included in Vp fit, point marker for Vp].
    """

    (curve,) = plt.plot(x_voltage, ln_current, label="ln(current)", marker=MKSTYLE)

    (Te_fit_line,) = plt.plot(x_voltage, Te_fit, label="Te fit", color="red")

    Te_scatter, = plt.plot(
        x_voltage[i_Te[0] : i_Te[1]],
        ln_current[i_Te[0] : i_Te[1]],
        zorder=2,
        color="red",
        marker=MKSTYLE,
        linestyle='None'
    )

    (Vp_fit_line,) = plt.plot(x_voltage, Vp_fit, label="Vp fit", color="limegreen")
    Vp_scatter, = plt.plot(
        x_voltage[i_Vp[0] : i_Vp[1]],
        ln_current[i_Vp[0] : i_Vp[1]],
        zorder=2,
        color="limegreen",
        marker=MKSTYLE,
        linestyle='None'
    )

    Vp_point, = plt.plot(
        Vp,
        Te_fit[np.where(x_voltage == Vp)[0]],
        zorder=2,
        color="black",
        label=f"Vp={Vp:.02f} V",
        marker=MKSTYLE,
        linestyle='None'
    )

    return [curve, Te_fit_line, Te_scatter, Vp_fit_line, Vp_scatter, Vp_point]



def graph_raw_curve(
    x_voltage: np.ndarray,
    current: np.ndarray,
    i_sat: float | np.ndarray,
    Vf_idx: int,
) -> list[plt.Line2D]:
    """
    Plot raw current vs. bias voltage, with an ion-saturation-current
    reference line and the floating potential (Vf) marked as a point.

    Args:
        x_voltage (1D array-like): Bias voltage values (x-axis).
        current (1D array-like): Measured current values (y-axis).
        i_sat (float or 1D array-like): Ion saturation current. If scalar,
            broadcast to a constant array matching len(current) so it can
            be plotted as a flat reference line; if array-like, plotted
            as-is (allows a non-constant i_sat curve).
        Vf_idx (int): Index into x_voltage/current identifying the
            floating potential (Vf) point to mark.

    Returns:
        list: The three matplotlib Line2D artists created, in order:
            [I-V curve, i_sat, Vf marker].
    """

    if np.ndim(i_sat) == 0:
        i_sat = np.full(len(current), i_sat)
    (curve,) = plt.plot(x_voltage, current, label="current", marker=MKSTYLE)
    isat_artist, = plt.plot(x_voltage, i_sat, label="i_sat", color="blue")
    Vf_artist, = plt.plot(x_voltage[Vf_idx], current[Vf_idx], label="Vf", marker=MKSTYLE, color="black")

    return [curve, isat_artist, Vf_artist]



def parse_folder_name(name, form=1, type=None):
    """
    Parse a structured experiment folder name into a human-readable label.

    Expects `name` to be a hyphen-delimited string with exactly four fields:
    "<gas>-<power>-<ice>-<pressure>" (e.g. "Argon-20W-Ice-1"). Pressure
    codes "1" and "2" are mapped to "400" and "700" (mTorr); any other
    value passes through unchanged. Ice codes "Ice"/"NoIce" are mapped to
    "With Ice"/"No Ice"; any other value passes through unchanged.

    Args:
        name (str): Folder name in "<gas>-<power>-<ice>-<pressure>" format.
        form (int): Selects the output format/verbosity:
            1 -> "{power}\\n{pressure} mTorr\\n{ice}" (multi-line, no gas)
            2 -> "{power} {pressure} mTorr {gas} Plasma, {ice}{type}"
            3 -> "{power} {pressure} mTorr {ice}"
            other -> returns `name` unchanged
        type (str, optional): If given, appended as " (type)" — only used
            in form 2.

    Returns:
        str: The formatted label, or the original `name` if `form` doesn't
            match a known case.
    """
        
    gas, power, ice, pressure = name.split("-")

    pressure_map = {"1": "400", "2": "700"}
    ice_map = {"Ice": "With Ice", "NoIce": "No Ice"}

    pressure_out = pressure_map.get(pressure, pressure)
    ice_out = ice_map.get(ice, ice)

    type = f" ({type})" if type is not None else ""

    match form:
        case 1:
            return f"{power}\n{pressure_out} mTorr\n{ice_out}"
        case 2:
            return f"{power} {pressure_out} mTorr {gas} Plasma, {ice_out}{type}"
        case 3:
            return f"{power} {pressure_out} mTorr {ice_out}"
        case _:
            return name