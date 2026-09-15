import numpy as np
import matplotlib.pyplot as plt
 

MKSTYLE = "."


def render_initial_IV_curve():
    fig, ax = plt.subplots(figsize=(3, 3))
    t = np.arange(0, 3, .01)
    artists = graph_electrontemp(t, t, 2, t, t, [0, 1], [0, 1])

    ax.set_xlabel("Bias Voltage (V)")
    ax.set_ylabel("Current (ln[A])")
    title_artist = ax.set_title("Choose Data Folder", fontsize="small")
    artists.append(title_artist)
    plt.tight_layout()

    return fig, ax, artists


def render_initial_raw_curve():
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



def graph_electrontemp(x_voltage, ln_current, Vp, Te_fit, Vp_fit, i_Te, i_Vp):

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



def graph_raw_curve(x_voltage, current, i_sat, Vf_idx):
    if np.ndim(i_sat) == 0:
        i_sat = np.full(len(current), i_sat)
    (curve,) = plt.plot(x_voltage, current, label="current", marker=MKSTYLE)
    isat_artist, = plt.plot(x_voltage, i_sat, label="i_sat", color="blue")
    Vf_artist, = plt.plot(x_voltage[Vf_idx], current[Vf_idx], label="Vf", marker=MKSTYLE, color="black")

    return [curve, isat_artist, Vf_artist]



def parse_folder_name(name, form=1, type=None):
    """
    Parses folder names of the form:
        [Gas]-[power]W-[Ice]-[pressure]

    Returns a formatted label:
        "[power]W [pressure] mTorr\n[Ice]"

    with substitutions:
        pressure: "1" -> "400", "2" -> "700"
        ice:      "Ice" -> "With Ice", "NoIce" -> "No Ice"
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
            return f"{power} {pressure_out} mTorr Argon Plasma, {ice_out}{type}"
        case 3:
            return f"{power} {pressure_out} mTorr {ice_out}"
        case _:
            return name