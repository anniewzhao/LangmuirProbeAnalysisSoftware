import tkinter as tk
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import KeyEvent
from typing import Callable

import Parse_Data
import Langmuir_Analysis
import Plot_Curves

currents, voltages, x_axis, indices = [], [], [], []
curr_frame = 0
frame_rate = 20
max_idx = 10
probe_w = 2.5 * 10 ** (-4)
diff_probe_l = 7.57 * 10 ** (-3)
opa_probe_l = 5.29 * 10 ** (-3)

starting_frame = False
attributes_arr = [None]


fol = "TestData"


def round_to_multiple(number: float, multiple: float) -> float:
    """
    Round `number` to the nearest multiple of `multiple`, unless
    `starting_frame` is set, in which case `number` is returned unchanged.

    Depends on the global `starting_frame` flag rather than a parameter,
    so its behavior at any given call site depends on external state set
    elsewhere (e.g. in start_semiauto_analysis).

    Args:
        number (float or int): Value to round.
        multiple (float or int): Multiple to round to.

    Returns:
        float or int: The rounded value, or `number` unchanged if
            `starting_frame` is true.
    """
    global starting_frame
    if not starting_frame:
        return round(number / multiple) * multiple
    else:
        return number


def open_folder() -> None:
    """
    Prompt the user to select a data folder, load its saved current/
    voltage arrays, initialize frame indices, and start semi-automatic
    analysis.

    Opens a directory picker, loads differential and OPA current/voltage
    arrays via Parse_Data.load_numpy_data (without trigger data), and
    selects between the differential and OPA arrays based on the current
    state of `button_toggle_method`. Reshapes `currents` to 2D if it was
    loaded as a 1D array. If the folder's parent directory is named
    "Afterglow", sets frame_rate=1 and seeks the current frame to the
    afterglow peak (via Langmuir_Analysis.find_afterglow_peak); otherwise
    sets frame_rate=20 and starts at frame 0. Loads/creates the index
    array for the current method, then kicks off
    start_semiauto_analysis.

    Reads/writes numerous globals: fol, frame_rate, currents, voltages,
    x_axis, indices, curr_frame.

    Returns:
        None
    """
    global fol, frame_rate
    # Open the file picker dialog
    folder_path = tk.filedialog.askdirectory(
        title="Folder containing 3D data?", initialdir=fol
    )

    label_openfolder.config(text=folder_path)

    # Ensure the user didn't click cancel
    if folder_path:
        print(f"Selected file: {folder_path}")
        global currents, voltages, x_axis, indices, curr_frame
        fol = Path(folder_path)
        i_diff, i_opa, v_diff, v_opa, x_axis = Parse_Data.load_numpy_data(
            fol, trigger=False
        )

        if button_toggle_method.cget("text") == "Differential":
            currents = i_diff
            voltages = v_diff
        else:
            currents = i_opa
            voltages = v_opa

        if currents.ndim == 1:
            currents = np.array([currents])

        if "Afterglow" in fol.parent.name:
            frame_rate = 1
            idx = Langmuir_Analysis.find_afterglow_peak(currents)
            curr_frame = idx

        else:
            frame_rate = 20
            curr_frame = 0

        indices = Parse_Data.initialize_indices(
            fol, len(x_axis), button_toggle_method.cget("text")
        )

        start_semiauto_analysis(len(x_axis))


def start_semiauto_analysis(num_frames: int) -> None:
    """
    Initialize UI state and data structures for a newly loaded folder,
    then render the first frame.

    Configures the frame slider's range, sets the ln-current axis x-limits
    from the current frame's voltage data, seeds the first frame's indices
    to a default [3, 5, 7, 9] if unset, and (re)initializes
    `attributes_arr` to a zero-filled placeholder for every frame. Sets
    `starting_frame` to True for the initial call to
    update_curve_graphs so that frame-dependent logic there (e.g.
    round_to_multiple, auto-index carry-over) can special-case the first
    render, then resets it to False afterward.

    Args:
        num_frames (int): Number of frames/curves in the loaded dataset,
            used to size `attributes_arr`.

    Reads/writes globals: curr_frame, indices, attributes_arr,
        starting_frame.

    Returns:
        None
    """
    global curr_frame, indices, attributes_arr, starting_frame
    slider_update.config(from_=0, to=len(x_axis) - 1)
    ax_ln.set_xlim(min(voltages[curr_frame]), max(voltages[curr_frame]))
    if indices[0][0] == 0:
        indices[0] = [3, 5, 7, 9]
        starting_frame = True

    attributes_arr = [[0 for _ in range(8)] for _ in range(num_frames)]
    update_curve_graphs(i=curr_frame)
    starting_frame = False


def get_probe_length() -> float:
    """
    Return the probe shaft length corresponding to the currently selected
    acquisition method (Differential vs. Opa).

    Returns:
        float: `diff_probe_l` if `button_toggle_method` currently reads
            "Differential", otherwise `opa_probe_l`.
    """
    if button_toggle_method.cget("text") == "Differential":
        return diff_probe_l
    else:
        return opa_probe_l


def update_IV_graph(
    i_sat: float, _voltages: np.ndarray, ln_current: np.ndarray, i: int
) -> tuple[float, float, float, float, float]:
    """
    Recompute Te/Vp/n0 for frame `i` and update the semi-log IV plot's
    artists in place.

    Builds Te/Vp fit index ranges from the global `indices` array for
    frame `i`, runs Langmuir_Analysis.process_IV_curve, and updates the
    fit-line, scatter-region, and Vp-marker artists accordingly. Falls
    back to a fixed set of zero values (Te, Vp, Te_r2, Vp_rmse, n0 = 0)
    if process_IV_curve raises ValueError, leaving the fit-related
    artists un-updated for this frame. Also updates the raw ln-current
    curve, both fit-region scatter overlays, and the text annotation
    regardless of success/failure, and rescales the y-axis to the data.

    Args:
        i_sat (float): Ion saturation current for this frame.
        _voltages (np.ndarray): Cleaned bias voltage array for this
            frame (post semilog_IV processing).
        ln_current (np.ndarray): Cleaned ln(current) array for this
            frame.
        i (int): Frame index, used to look up fit index ranges from the
            global `indices` array.

    Reads/writes global: indices (read only here).

    Returns:
        tuple[float, float, float, float, float]: (Te, Vp, n0, Te_r2,
            Vp_rmse) for this frame — 0-valued defaults if the fit
            failed.
    """
    global indices

    Te_index = [indices[i][0], indices[i][1]]
    Vp_index = [indices[i][2], indices[i][3]]

    Te, Vp, Te_r2, Vp_rmse, n0 = 0, 0, 0, 0, 0

    try:
        Te, Vp, n0, Te_fit, Vp_fit, Te_r2, Vp_rmse = Langmuir_Analysis.process_IV_curve(
            i_sat,
            _voltages,
            ln_current,
            Te_index,
            Vp_index,
            probe_w,
            get_probe_length(),
        )
        artists_ln[1].set_data(_voltages, Te_fit)
        artists_ln[3].set_data(_voltages, Vp_fit)
        index_of_Vp = np.where(_voltages == Vp)[0]
        artists_ln[5].set_data([Vp, Vp], [Te_fit[index_of_Vp], Te_fit[index_of_Vp]])
    except ValueError:
        print("Value Error")

    artists_ln[0].set_data(_voltages, ln_current)
    artists_ln[2].set_data(
        _voltages[Te_index[0] : Te_index[1]], ln_current[Te_index[0] : Te_index[1]]
    )
    artists_ln[4].set_data(
        _voltages[Vp_index[0] : Vp_index[1]], ln_current[Vp_index[0] : Vp_index[1]]
    )
    artists_ln[6].set_text(
        f"Te: {Te:.02f} (r\u00b2={Te_r2:.4f})   Vp: {Vp:.02f} (RMSE={Vp_rmse:.4f})"
    )

    ax_ln.set_ylim(-15, max(ln_current) + 1)
    canvas_ln.draw_idle()

    return Te, Vp, n0, Te_r2, Vp_rmse


def update_raw_graph(
    current: np.ndarray, voltages: np.ndarray, i_sat: float, vf_idx: int
) -> float:
    """
    Update the raw current-voltage plot's artists (current curve, i_sat
    reference line, Vf marker, and annotation text) for a single frame.

    Converts current to milliamps for display, and rescales the raw plot
    axes to the current frame's data range.

    Args:
        current (np.ndarray): Raw current trace for this frame (amps).
        voltages (np.ndarray): Raw voltage trace for this frame.
        i_sat (float): Ion saturation current (amps) for this frame.
        vf_idx (int): Index into `voltages`/`current` marking the
            floating potential point.

    Returns:
        float: The voltage value at `vf_idx` (i.e. Vf for this frame).
    """

    _current = current * 1e3
    i_sat_arr = np.full(len(_current), i_sat) * 1e3
    artists_raw[0].set_data(voltages, _current)
    artists_raw[1].set_data(voltages, i_sat_arr)
    artists_raw[2].set_data(
        [voltages[vf_idx], voltages[vf_idx]], [_current[vf_idx], _current[vf_idx]]
    )
    artists_raw[3].set_text(
        f"Vf: {voltages[vf_idx]:.02f}      i_sat (mA): {i_sat*1E3:.04f}"
    )
    ax_raw.set_xlim(min(voltages) - 1, max(voltages) + 1)
    ax_raw.set_ylim(min(_current) - 0.03, max(_current) + 0.03)
    canvas_raw.draw_idle()

    return voltages[vf_idx]


def update_curve_graphs(i: int = 0) -> None:
    """
    Recompute and redraw both the semi-log IV and raw current plots for
    frame `i`, updating shared state and persisting index changes to
    disk.

    Sets the current frame and slider position, computes Vf index and
    semi-log IV data for the frame, and — if this is a starting frame or
    the frame's indices are unset, and auto mode is enabled — carries
    over the previous frame's indices and re-derives the Vp fit range via
    Langmuir_Analysis.find_max_Vp_fit (seeding a default Te range on the
    very first frame and toggling auto mode off afterward). Updates the
    index entry boxes to reflect the frame's current indices, updates
    both graphs, records the frame's computed attributes into
    `attributes_arr`, and saves the updated `indices` array to disk.

    Args:
        i (int): Frame index to display.

    Reads/writes globals: x_axis, currents, voltages, indices,
        curr_frame, max_idx, attributes_arr, starting_frame.

    Returns:
        None
    """
    global x_axis, currents, voltages, indices, curr_frame, max_idx, attributes_arr, starting_frame
    curr_frame = i
    slider_update.set(i)

    vf_idx = Langmuir_Analysis.find_vf_idx(currents[i])
    i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV(
        [voltages[i], currents[i]]
    )

    if (starting_frame or indices[i][0] == 0):
        if button_toggle_auto.cget("text") == "Auto":
            indices[i] = indices[i - frame_rate]
            Vp_indices = Langmuir_Analysis.find_max_Vp_fit(_voltages, ln_current)
            indices[i][2], indices[i][3] = Vp_indices[0], Vp_indices[1]
            if starting_frame:
                indices[i][1] = 5
                indices[i][0] = 3
        else:
            indices[i] = [3, 5, 7, 9]

    for box, num in zip(idx_boxes, indices[i]):
        box.delete(0, "end")
        box.insert(0, num)

    max_idx = len(ln_current)

    Te, Vp, n0, Te_r2, Vp_rmse = update_IV_graph(i_sat, _voltages, ln_current, i)
    Vf = update_raw_graph(currents[i], voltages[i], i_sat, vf_idx)

    descriptor = x_axis[i]

    attributes_arr[i] = [descriptor, Te, Te_r2, Vp, Vp_rmse, n0, Vf, i_sat]
    np.save(fol / f"indices_{button_toggle_method.cget("text")}.npy", indices)


def auto_process_data():
    """
    Recompute Te/Vp/n0 for every frame using a fixed voltage-range
    (rather than per-frame index-based) fit selection, then save results
    and updated indices to disk.

    Derives voltage bounds for the Te and Vp fit regions from the
    current frame's indices, then for every frame in the dataset looks
    up the index positions in that frame's own voltage axis nearest to
    those bounds (via np.where), reprocesses the curve with
    Langmuir_Analysis.process_IV_curve using those per-frame indices,
    and accumulates results into local copies (`auto_attr`, `auto_idx`)
    rather than overwriting the shared `attributes_arr`/`indices` until
    the loop completes. Selects a CSV descriptor label based on the
    parent folder name, then saves results via
    Parse_Data.save_params_as_csv and persists `auto_idx` to disk.

    Reads/writes globals: curr_frame, indices, voltages, currents,
        x_axis, attributes_arr, probe_w.

    Returns:
        None
    """
    global curr_frame, indices, voltages, currents, x_axis, attributes_arr

    i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV(
        [voltages[curr_frame], currents[curr_frame]]
    )

    max_voltage = _voltages[indices[curr_frame][1]]
    min_voltage = _voltages[indices[curr_frame][0]]
    max_vp = _voltages[indices[curr_frame][3] - 1]
    min_vp = _voltages[indices[curr_frame][2]]

    auto_attr = attributes_arr.copy()
    auto_idx = indices.copy()

    for i in range(0, len(x_axis) - 1):
        global probe_w

        vf_idx = Langmuir_Analysis.find_vf_idx(currents[i])
        i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV(
            [voltages[i], currents[i]]
        )

        try:
            vp_idx_max = np.where(_voltages >= max_vp)[0][0]
        except IndexError:
            vp_idx_max = len(voltages)

        Te_index = [
            np.where(_voltages >= min_voltage)[0][0],
            np.where(_voltages >= max_voltage)[0][0],
        ]
        Vp_index = [np.where(_voltages >= min_vp)[0][0], vp_idx_max]

        auto_idx[i] = [Te_index[0], Te_index[1], Vp_index[0], Vp_index[1]]

        Te, Vp, n0, Te_fit, Vp_fit, Te_r2, Vp_rmse = Langmuir_Analysis.process_IV_curve(
            i_sat,
            _voltages,
            ln_current,
            Te_index,
            Vp_index,
            probe_w,
            get_probe_length(),
        )
        Vf = voltages[i][vf_idx]
        descriptor = x_axis[i]

        auto_attr[i] = [descriptor, Te, Te_r2, Vp, Vp_rmse, n0, Vf, i_sat]

    match fol.parent.name:
        case "RF":
            desc = "phase"
        case "Afterglow":
            desc = "time"
        case _:
            desc = "descriptor"

    Parse_Data.save_params_as_csv(
        fol, desc, auto_attr, button_toggle_method.cget("text"), ""
    )
    np.save(fol / f"indices_{button_toggle_method.cget("text")}.npy", auto_idx)

    attributes_arr = auto_attr
    indices = auto_idx

    print("done :)")


def update_all_params():
    """
    Recompute Te/Vp/n0 for every frame using each frame's own currently
    stored index values (unlike auto_process_data's fixed-voltage-range
    approach), then save results to disk.

    For every frame, reads that frame's Te/Vp index ranges directly from
    the global `indices` array, reprocesses the curve via
    Langmuir_Analysis.process_IV_curve, and writes results into
    `attributes_arr` in place. Selects a CSV descriptor label based on
    the parent folder name, then saves via Parse_Data.save_params_as_csv
    and re-persists `indices` to disk (unchanged, since this function
    doesn't modify indices).

    Reads/writes globals: indices, voltages, currents, x_axis,
        attributes_arr, probe_w.

    Returns:
        None
    """
    global indices, voltages, currents, x_axis, attributes_arr

    for i in range(0, len(x_axis)):
        global probe_w

        vf_idx = Langmuir_Analysis.find_vf_idx(currents[i])
        i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV(
            [voltages[i], currents[i]]
        )

        Te_index = [indices[i][0], indices[i][1]]
        Vp_index = [indices[i][2], indices[i][3]]

        Te, Vp, n0, Te_fit, Vp_fit, Te_r2, Vp_rmse = Langmuir_Analysis.process_IV_curve(
            i_sat,
            _voltages,
            ln_current,
            Te_index,
            Vp_index,
            probe_w,
            get_probe_length(),
        )
        Vf = voltages[i][vf_idx]
        descriptor = x_axis[i]

        attributes_arr[i] = [descriptor, Te, Te_r2, Vp, Vp_rmse, n0, Vf, i_sat]

    match fol.parent.name:
        case "RF":
            desc = "phase"
        case "Afterglow":
            desc = "time"
        case _:
            desc = "descriptor"

    Parse_Data.save_params_as_csv(
        fol, desc, attributes_arr, button_toggle_method.cget("text"), ""
    )
    np.save(fol / f"indices_{button_toggle_method.cget("text")}.npy", indices)

    print("done :)")


def push_indices():
    """
    Propagate the current frame's Te/Vp fit voltage bounds forward to all
    subsequent frames, without recomputing or saving any fit results.

    Derives voltage bounds for the Te and Vp fit regions from the current
    frame's indices, then for every frame from `curr_frame` onward, finds
    the nearest matching index positions in that frame's own voltage axis
    and overwrites its entry in the global `indices` array. Does not call
    process_IV_curve, update any graphs, or persist `indices` to disk.

    Reads/writes globals: curr_frame, indices, voltages, currents,
        x_axis, attributes_arr (declared global but not written to).

    Returns:
        None
    """

    global curr_frame, indices, voltages, currents, x_axis, attributes_arr

    i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV(
        [voltages[curr_frame], currents[curr_frame]]
    )

    max_voltage = _voltages[indices[curr_frame][1]]
    min_voltage = _voltages[indices[curr_frame][0]]
    max_vp = _voltages[indices[curr_frame][3] - 1]
    min_vp = _voltages[indices[curr_frame][2]]

    for i in range(curr_frame, len(indices) - 1):

        i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV(
            [voltages[i], currents[i]]
        )

        try:
            vp_idx_max = np.where(_voltages >= max_vp)[0][0]
        except IndexError:
            vp_idx_max = len(voltages)
        Te_index = [
            np.where(_voltages >= min_voltage)[0][0],
            np.where(_voltages >= max_voltage)[0][0],
        ]
        Vp_index = [np.where(_voltages >= min_vp)[0][0], vp_idx_max]

        indices[i] = [Te_index[0], Te_index[1], Vp_index[0], Vp_index[1]]

    print("Done ;)")


def prev_frame():
    """
    Step the current frame backward by `frame_rate` (clamped to 0) and
    redraw both graphs.

    Reads/writes global: curr_frame.

    Returns:
        None
    """
    global curr_frame
    curr_frame = curr_frame - frame_rate if curr_frame > frame_rate else 0
    update_curve_graphs(curr_frame)


def next_frame():
    """
    Step the current frame forward by `frame_rate` (clamped to the last
    frame) and redraw both graphs.

    Reads/writes global: curr_frame.

    Returns:
        None
    """
    global curr_frame
    curr_frame = (
        curr_frame + frame_rate
        if curr_frame < len(x_axis) - frame_rate
        else len(x_axis) - 1
    )
    update_curve_graphs(curr_frame)


def change_fit_index(variation: str) -> None:
    """
    Adjust one boundary of the current frame's Te or Vp fit index range
    by one step, respecting minimum-gap and range constraints, then
    redraw both graphs.

    Which index pair is modified (Te's [0,1] or Vp's [2,3] within
    `indices[curr_frame]`) depends on the current state of
    `button_toggle_vp`. `variation` selects which bound moves and in
    which direction; each branch enforces that the lower bound stays
    below the upper bound (with a minimum gap of 2) and within [0,
    max_idx], printing a message and taking no action if the requested
    change would violate this.

    Args:
        variation (str): One of "inc_low", "dec_low", "inc_high",
            "dec_high", selecting which bound to adjust and in which
            direction.

    Reads/writes global: curr_frame (read only here); mutates `indices`
        in place.

    Returns:
        None
    """
    global curr_frame
    idx = [0, 1] if button_toggle_vp.cget("text") == "Te" else [2, 3]
    match (variation):
        case "inc_low":
            if (indices[curr_frame][idx[0]]) > 1:  # ending index not included
                indices[curr_frame][idx[0]] -= 1
                idx_boxes[idx[0]].config(text=f"{indices[curr_frame][idx[0]]}")
            else:
                print("cannot decrease more.")
        case "dec_low":
            if (indices[curr_frame][idx[0]]) + 2 < indices[curr_frame][idx[1]]:
                indices[curr_frame][idx[0]] += 1
                idx_boxes[idx[0]].config(text=f"{indices[curr_frame][idx[0]]}")
            else:
                print("Bottom index must be < top index")
        case "inc_high":
            if (indices[curr_frame][idx[1]]) < max_idx:
                indices[curr_frame][idx[1]] += 1
                idx_boxes[idx[1]].config(text=f"{indices[curr_frame][idx[1]]}")
            else:
                print("cannot increase more.")
        case "dec_high":
            if (indices[curr_frame][idx[1]]) - 2 > indices[curr_frame][idx[0]]:
                indices[curr_frame][idx[1]] -= 1
                idx_boxes[idx[1]].config(text=f"{indices[curr_frame][idx[1]]}")
            else:
                print("top index must be > bottom index")
    update_curve_graphs(curr_frame)


def toggle_fitsetting(event: tk.Event = None) -> None:
    """
    Toggle the fit-selection button between "Te" and "Vp" states,
    updating its label and background color accordingly.

    Args:
        event: Optional Tk event parameter (unused), present to allow
            use as a button/key callback.

    Returns:
        None
    """
    if button_toggle_vp.cget("text") == "Te":
        button_toggle_vp.config(text="Vp", bg="green3")
    else:
        button_toggle_vp.config(text="Te", bg="red")


def toggle_method(event: tk.Event = None) -> None:
    """
    Toggle the acquisition-method button between "Differential" and
    "Opa" states, updating its label and background color accordingly.

    Note: toggling this does not itself reload `currents`/`voltages` for
    the new method — that only happens the next time open_folder runs.

    Args:
        event: Optional Tk event parameter (unused), present to allow
            use as a button/key callback.

    Returns:
        None
    """
    if button_toggle_method.cget("text") == "Differential":
        button_toggle_method.config(text="Opa", bg="aquamarine")
    else:
        button_toggle_method.config(text="Differential", bg="yellow")


def toggle_auto(event: tk.Event = None) -> None:
    """
    Toggle the auto-fit button between "Auto" and "Manual" states,
    updating its label and background color accordingly.

    Args:
        event: Optional Tk event parameter (unused), present to allow
            use as a button/key callback.

    Returns:
        None
    """
    if button_toggle_auto.cget("text") == "Auto":
        button_toggle_auto.config(text="Manual", bg="MediumOrchid1")
    else:
        button_toggle_auto.config(text="Auto", bg="tomato")


def on_key_press(event: KeyEvent) -> None:
    """
    Dispatch a matplotlib key-press event to the corresponding frame-
    navigation or fit-index-adjustment action.

    Maps "a"/"d" to previous/next frame, "w" to toggling Te/Vp fit
    selection, and arrow keys to change_fit_index in the corresponding
    direction. Keys not in this mapping are ignored.

    Args:
        event: Matplotlib key-press event, expected to expose a `.key`
            attribute.

    Returns:
        None
    """

    match (event.key):
        case "a":
            prev_frame()
        case "d":
            next_frame()
        case "w":
            toggle_fitsetting()

        case "up":
            change_fit_index("inc_high")
        case "down":
            change_fit_index("dec_high")
        case "left":
            change_fit_index("inc_low")
        case "right":
            change_fit_index("dec_low")


def save_data():
    """
    Save the current session's accumulated attributes to CSV and close
    the application window.

    Selects a CSV descriptor label based on the parent folder name,
    saves `attributes_arr` via Parse_Data.save_params_as_csv, then
    destroys the root Tk window (ending the application).

    Reads global: attributes_arr.

    Returns:
        None
    """
    global attributes_arr
    match fol.parent.name:
        case "RF":
            desc = "phase"
        case "Afterglow":
            desc = "time"
        case _:
            desc = "descriptor"

    Parse_Data.save_params_as_csv(
        fol, desc, attributes_arr, button_toggle_method.cget("text"), ""
    )

    root.destroy()


def update_slider(i: int | str = 0) -> None:
    """
    Redraw both graphs for the frame corresponding to the slider's
    current value.

    Args:
        i (int or str): Frame index from the slider widget; cast to int
            before use.

    Returns:
        None
    """
    update_curve_graphs(int(i))


def object_with_scale_buttons(
    frame: tk.Frame, object: tk.Widget, prev_function: Callable, next_function: Callable
) -> None:
    """
    Lay out a widget flanked by "Prev" and "Next" buttons in a single
    grid row.

    Args:
        frame (tk.Frame): Parent frame to grid the buttons into (note:
            `object` is gridded into this same frame's grid, so `frame`
            must be `object`'s actual parent for this layout to render
            correctly).
        object (tk widget): The widget to place between the two buttons.
        prev_function (Callable): Command bound to the "Prev" button.
        next_function (Callable): Command bound to the "Next" button.

    Returns:
        None
    """
    button_prev = tk.Button(master=frame, text="Prev", command=prev_function)
    button_next = tk.Button(master=frame, text="Next", command=next_function)

    button_prev.grid(row=0, column=0, sticky="nsew")
    object.grid(row=0, column=1)
    button_next.grid(row=0, column=2, sticky="nsew")


def create_spinbox_grid(parent: tk.Widget) -> tuple[tk.Frame, list[tk.Spinbox]]:
    """
    Build a 2x2 grid of Spinbox widgets inside a new frame.

    Args:
        parent (tk widget): Parent widget for the new frame.

    Returns:
        tuple[tk.Frame, list[tk.Spinbox]]: The container frame and a flat
            list of the four Spinbox widgets, in row-major order.
    """
    frame = tk.Frame(parent)

    spinidx_boxes = []
    for i in range(2):
        for j in range(2):
            sb = tk.Spinbox(frame, from_=0, to=100)
            sb.grid(row=i, column=j, padx=5, pady=5)
            spinidx_boxes.append(sb)

    return frame, spinidx_boxes


def create_graph_canvas(
    fig: plt.Figure,
) -> tuple[FigureCanvasTkAgg, NavigationToolbar2Tk]:
    """
    Embed a matplotlib figure into the Tk root window with a navigation
    toolbar, and wire up key-press/button-release event handling.

    Args:
        fig (matplotlib.figure.Figure): Figure to embed.

    Returns:
        tuple[FigureCanvasTkAgg, NavigationToolbar2Tk]: The created
            canvas and toolbar widgets.
    """
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()

    canvas.mpl_connect("key_press_event", on_key_press)
    canvas.mpl_connect("button_release_event", on_key_press)

    return canvas, toolbar


root = tk.Tk()

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)
root.columnconfigure(2, weight=1)
root.rowconfigure(2, weight=1)  # the row containing the canvas_ln

label = tk.Label(root, text="Semi-Auto Langmuir Analysis GUI")
label.grid(row=0, column=0, columnspan=6)

button_openfolder = tk.Button(master=root, text="Open Folder", command=open_folder)
label_openfolder = tk.Label(root, text="No Folder Open")

button_openfolder.grid(row=1, column=0)
label_openfolder.grid(row=1, column=2, columnspan=4)


fig_ln, ax_ln, artists_ln = Plot_Curves.render_initial_IV_curve()
canvas_ln, toolbar_ln = create_graph_canvas(fig_ln)
canvas_ln.get_tk_widget().grid(row=2, column=0, columnspan=3, sticky="nsew")
toolbar_ln.grid(row=3, column=0, columnspan=3)

fig_raw, ax_raw, artists_raw = Plot_Curves.render_initial_raw_curve()
canvas_raw, toolbar_raw = create_graph_canvas(fig_raw)
canvas_raw.get_tk_widget().grid(row=2, column=3, columnspan=3, sticky="nsew")
toolbar_raw.grid(row=3, column=4, columnspan=2)


button_quit = tk.Button(master=root, text="Quit", command=root.destroy)

spinbox_frame, idx_boxes = create_spinbox_grid(root)
spinbox_frame.grid(row=4, column=0)


frame_buttons = tk.Frame(master=root)
frame_buttons.grid(row=4, column=2, sticky="nsew")

button_toggle_vp = tk.Label(master=frame_buttons, text="Te", bg="red")
button_toggle_vp.bind("<Button-1>", toggle_fitsetting)
button_toggle_vp.pack(fill="both")

button_toggle_method = tk.Label(master=frame_buttons, text="Differential", bg="yellow")
button_toggle_method.bind("<Button-1>", toggle_method)
button_toggle_method.pack(fill="both")

button_toggle_auto = tk.Label(master=frame_buttons, text="Auto", bg="tomato")
button_toggle_auto.bind("<Button-1>", toggle_auto)
button_toggle_auto.pack(fill="both")


button_frame = tk.Frame(master=root)
button_frame.grid(row=3, column=3, rowspan=2, sticky="nsew")

button_auto_full_send = tk.Button(
    master=button_frame, text="AUTO FULL SEND", command=auto_process_data
)
button_auto_full_send.pack(fill="both")

button_push_indices = tk.Button(
    master=button_frame, text="Push Indices", command=push_indices
)
button_push_indices.pack(fill="both")

button_auto_params = tk.Button(
    master=button_frame, text="Auto_params", command=update_all_params
)
button_auto_params.pack(fill="both")

button_finish_curve = tk.Button(master=button_frame, text="Done :)", command=save_data)
button_finish_curve.pack(fill="both")


slider_frame = tk.Frame(root)
slider_frame.grid(row=4, column=4)
slider_update = tk.Scale(
    slider_frame,
    from_=1,
    to=5,
    orient=tk.HORIZONTAL,
    command=update_slider,
    label="Frame",
)
object_with_scale_buttons(slider_frame, slider_update, prev_frame, next_frame)


root.mainloop()
