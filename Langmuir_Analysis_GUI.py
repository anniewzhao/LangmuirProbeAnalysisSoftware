import tkinter as tk
import numpy as np
from pathlib import Path


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import Parse_Data
import Langmuir_Analysis
import Plot_Curves


currents, voltages, x_axis, indices = [], [], [], []
curr_frame=0
frame_rate = 20
max_idx = 10
probe_w = 2.5 * 10 ** (-4)
diff_probe_l = 7.57 * 10 ** (-3)
opa_probe_l = 5.29 * 10 ** (-3)

starting_frame = True


attributes_arr = [None]



fol="TestData"


def round_to_multiple(number, multiple):
    global starting_frame
    if not starting_frame:
        return round(number / multiple) * multiple
    else:
        return number


def open_folder():
    global fol, frame_rate
    # Open the file picker dialog
    folder_path = tk.filedialog.askdirectory(
        title="Folder containing 3D data?",
        initialdir=fol
    )

    label_openfolder.config(text=folder_path)
    
    # Ensure the user didn't click cancel
    if folder_path:
        print(f"Selected file: {folder_path}")
        global currents, voltages, x_axis, indices, curr_frame
        fol = Path(folder_path)
        i_diff, i_opa, v_diff, v_opa, x_axis = Parse_Data.load_numpy_data(fol, trigger=False)

        if button_toggle_method.cget("text") == "Differential":
            currents = i_diff
            voltages = v_diff
        else:
            currents = i_opa
            voltages  = v_opa

        if currents.ndim == 1:
            currents = np.arr([currents])

        if "Afterglow" in fol.parent.name:
            frame_rate = 1
            idx = Langmuir_Analysis.find_afterglow_peak(x_axis, currents)
            print(f"Frame of Peak Current: {idx}")
            curr_frame = idx

        else:
            frame_rate = 20
            curr_frame=0

        indices = Parse_Data.initialize_indices(fol, len(x_axis), button_toggle_method.cget("text"))
        
        start_semiauto_analysis(len(x_axis))



def start_semiauto_analysis(num_frames):
    global curr_frame, indices, attributes_arr, starting_frame
    slider_update.config(from_=0, to=len(x_axis)-1)
    print(voltages.shape)
    ax_ln.set_xlim(min(voltages[curr_frame]), max(voltages[curr_frame]))
    if indices[0][0] == 0:
        indices[0] = [3, 5, 7, 9]

    attributes_arr = [[0 for _ in range(8)] for _ in range(num_frames)]

    starting_frame = True
    update_curve_graphs(i=curr_frame)
    starting_frame = False
    


def get_probe_length():
    if button_toggle_method.cget("text") == "Differential":
        return diff_probe_l
    else:
        return opa_probe_l


def update_IV_graph(i_sat, _voltages, ln_current, i):
    global indices

    Te_index = [indices[i][0], indices[i][1]]
    Vp_index = [indices[i][2], indices[i][3]]

    Te, Vp, Te_r2, Vp_rmse, n0 = 0, 0, 0, 0, 0

    try:
        Te, Vp, n0, Te_fit, Vp_fit, Te_r2, Vp_rmse = Langmuir_Analysis.process_IV_curve(i_sat, _voltages, ln_current, Te_index, Vp_index, probe_w, get_probe_length())
        artists_ln[1].set_data(_voltages, Te_fit)
        artists_ln[3].set_data(_voltages, Vp_fit)
        index_of_Vp = np.where(_voltages == Vp)[0]
        artists_ln[5].set_data([Vp, Vp], [Te_fit[index_of_Vp], Te_fit[index_of_Vp]])
    except ValueError:
        print("Value Error")

    artists_ln[0].set_data(_voltages, ln_current)
    artists_ln[2].set_data(_voltages[Te_index[0]:Te_index[1]], ln_current[Te_index[0]:Te_index[1]])
    artists_ln[4].set_data(_voltages[Vp_index[0]:Vp_index[1]], ln_current[Vp_index[0]:Vp_index[1]])
    artists_ln[6].set_text(f"Te: {Te:.02f} (r\u00b2={Te_r2:.4f})   Vp: {Vp:.02f} (RMSE={Vp_rmse:.4f})")

    ax_ln.set_ylim(-15, max(ln_current)+1)
    canvas_ln.draw_idle()

    return Te, Vp, n0, Te_r2, Vp_rmse




def update_raw_graph(current, voltages, i_sat, vf_idx):
    _current = current*1E3
    i_sat_arr = np.full(len(_current), i_sat)*1E3
    artists_raw[0].set_data(voltages, _current)
    artists_raw[1].set_data(voltages, i_sat_arr)
    artists_raw[2].set_data([voltages[vf_idx], voltages[vf_idx]], [_current[vf_idx], _current[vf_idx]])
    artists_raw[3].set_text(f"Vf: {voltages[vf_idx]:.02f}      i_sat (mA): {i_sat*1E3:.04f}")
    ax_raw.set_xlim(min(voltages)-1, max(voltages)+1)
    ax_raw.set_ylim(min(_current)-0.03, max(_current)+0.03)
    canvas_raw.draw_idle()

    return voltages[vf_idx]


def update_curve_graphs(i=0):
    global x_axis, currents, voltages, indices, curr_frame, max_idx, attributes_arr, starting_frame

    #i = round_to_multiple(int(i), frame_rate)

    curr_frame = i
    slider_update.set(i)
    

    vf_idx = Langmuir_Analysis.find_vf_idx(currents[i])
    i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV([voltages[i], currents[i]])

    if (starting_frame or indices[i][0] == 0) and (button_toggle_auto.cget("text") == "Auto"):
        indices[i] = indices[i-frame_rate]
        Vp_indices = Langmuir_Analysis.find_max_Vp_fit(_voltages, ln_current)
        indices[i][2], indices[i][3] = Vp_indices[0], Vp_indices[1]
        if starting_frame:
            indices[i][1] = 5
            indices[i][0] = 3
            toggle_auto()

    for box, num in zip(idx_boxes, indices[i]):
            box.delete(0, "end")
            box.insert(0, num)

    max_idx = len(ln_current)

    Te, Vp, n0, Te_r2, Vp_rmse = update_IV_graph(i_sat, _voltages, ln_current, i)
    Vf = update_raw_graph(currents[i], voltages[i], i_sat, vf_idx)

    descriptor = x_axis[i]

    attributes_arr[i] = [descriptor, Te, Te_r2, Vp, Vp_rmse, n0, Vf, i_sat]
    np.save(fol / f"indices_{button_toggle_method.cget("text")}.npy", indices)



def auto_process_data(event=None):
    global curr_frame, indices, voltages, currents, x_axis, attributes_arr

    i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV([voltages[curr_frame], currents[curr_frame]])

    max_voltage = _voltages[indices[curr_frame][1]]
    min_voltage = _voltages[indices[curr_frame][0]]
    len_Vp = indices[curr_frame][3] - indices[curr_frame][2]
    max_vp = _voltages[indices[curr_frame][3]]
    min_vp = _voltages[indices[curr_frame][2]]

    print(f"min: {min_voltage}    max: {max_voltage}")

    auto_attr = attributes_arr.copy()
    auto_idx = indices.copy()

    for i in range(0, len(x_axis-1)):
        print(i)
        global probe_w

        vf_idx = Langmuir_Analysis.find_vf_idx(currents[i])
        i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV([voltages[i], currents[i]])

        Te_index = [np.where(_voltages >= min_voltage)[0][0], np.where(_voltages >= max_voltage)[0][0]]
        #Vp_index = [len(_voltages) - len_Vp, len(_voltages)]
        Vp_index = [np.where(_voltages >= min_vp)[0][0], np.where(_voltages >= max_vp)[0][0]]

        auto_idx[i] = [Te_index[0], Te_index[1], Vp_index[0], Vp_index[1]]

        Te, Vp, n0, Te_fit, Vp_fit, Te_r2, Vp_rmse = Langmuir_Analysis.process_IV_curve(i_sat, _voltages, ln_current, Te_index, Vp_index, probe_w, get_probe_length())
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

    Parse_Data.save_params_as_csv(fol, desc, auto_attr, button_toggle_method.cget("text"),2)
    np.save(fol / f"indices_{button_toggle_method.cget("text")}2.npy", auto_idx)

    print("done :)")




def update_all_params(event=None):
    global  indices, voltages, currents, x_axis, attributes_arr
    
    for i in range(0, len(x_axis)):
        global probe_w

        vf_idx = Langmuir_Analysis.find_vf_idx(currents[i])
        i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV([voltages[i], currents[i]])

        Te_index = [indices[i][0], indices[i][1]]
        Vp_index = [indices[i][2], indices[i][3]]

        Te, Vp, n0, Te_fit, Vp_fit, Te_r2, Vp_rmse = Langmuir_Analysis.process_IV_curve(i_sat, _voltages, ln_current, Te_index, Vp_index, probe_w, get_probe_length())
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

    Parse_Data.save_params_as_csv(fol, desc, attributes_arr, button_toggle_method.cget("text"), "")
    np.save(fol / f"indices_{button_toggle_method.cget("text")}.npy", indices)

    print("done :)")


def push_indices(event=None):

    global curr_frame, indices, voltages, currents, x_axis, attributes_arr
    
    i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV([voltages[curr_frame], currents[curr_frame]])

    max_voltage = _voltages[indices[curr_frame][1]]
    min_voltage = _voltages[indices[curr_frame][0]]
    max_vp = _voltages[indices[curr_frame][3]]
    min_vp = _voltages[indices[curr_frame][2]]
    len_Vp = indices[curr_frame][3] - indices[curr_frame][2]

    print(f"min: {min_voltage}    max: {max_voltage}")

    for i in range(curr_frame, len(indices)-1):
        print(i)

        i_sat, _voltages, ln_current = Langmuir_Analysis.semilog_IV([voltages[i], currents[i]])

        Te_index = [np.where(_voltages >= min_voltage)[0][0], np.where(_voltages >= max_voltage)[0][0]]
        Vp_index = [len(_voltages) - len_Vp, len(_voltages)]
        #Vp_index = [np.where(_voltages >= min_vp)[0][0], np.where(_voltages >= max_vp)[0][0]]

        indices[i] = [Te_index[0], Te_index[1], Vp_index[0], Vp_index[1]]

    print("Done ;)")
        


def prev_frame():
    global curr_frame
    curr_frame = curr_frame-frame_rate if curr_frame>frame_rate else 0
    update_curve_graphs(curr_frame)

def next_frame():
    global curr_frame
    curr_frame = curr_frame+frame_rate if curr_frame<len(x_axis)-frame_rate else len(x_axis)-1
    update_curve_graphs(curr_frame)


def change_fit_index(variation):
    global curr_frame
    idx = [0, 1] if button_toggle_vp.cget("text") == "Te" else [2, 3]
    match(variation):
        case "inc_low":
            if (indices[curr_frame][idx[0]]) > 1: #ending index not included
                indices[curr_frame][idx[0]] -=1
                idx_boxes[idx[0]].config(text=f"{indices[curr_frame][idx[0]]}")
            else:
                print("cannot decrease more.")
        case "dec_low":
            if (indices[curr_frame][idx[0]])+2 < indices[curr_frame][idx[1]]:
                indices[curr_frame][idx[0]] +=1
                idx_boxes[idx[0]].config(text=f"{indices[curr_frame][idx[0]]}")
            else:
                print("Bottom index must be < top index")
        case "inc_high":
            if (indices[curr_frame][idx[1]]) < max_idx:
                indices[curr_frame][idx[1]] +=1
                idx_boxes[idx[1]].config(text=f"{indices[curr_frame][idx[1]]}")
            else:
                print("cannot increase more.")
        case "dec_high":
            if (indices[curr_frame][idx[1]])-2 > indices[curr_frame][idx[0]]:
                indices[curr_frame][idx[1]] -=1
                idx_boxes[idx[1]].config(text=f"{indices[curr_frame][idx[1]]}")
            else:
                print("top index must be > bottom index")
    update_curve_graphs(curr_frame)


def toggle_fitsetting(event=None):
    if button_toggle_vp.cget("text") == "Te":
        button_toggle_vp.config(text="Vp", bg="green3")
    else:
        button_toggle_vp.config(text="Te", bg="red")

def toggle_method(event=None):
    if button_toggle_method.cget("text") == "Differential":
        button_toggle_method.config(text="Opa", bg="aquamarine")
    else:
        button_toggle_method.config(text="Differential", bg="yellow")

def toggle_auto(event=None):
    if button_toggle_auto.cget("text") == "Auto":
        button_toggle_auto.config(text="Manual", bg="MediumOrchid1")
    else:
        button_toggle_auto.config(text="Auto", bg="tomato")


def on_key_press(event):

    match(event.key):
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

        

def on_mouse_click(event):
    if event.key in ("right", "a"):
        prev_frame()
    elif event.key in ("left", "d"):
        next_frame()



def save_data(event=None):
    global attributes_arr
    match fol.parent.name:
        case "RF":
            desc = "phase"
        case "Afterglow":
            desc = "time"
        case _:
            desc = "descriptor"

    Parse_Data.save_params_as_csv(fol, desc, attributes_arr, button_toggle_method.cget("text"),"")

    root.destroy()



def update_slider(i=0):
    update_curve_graphs(int(i))



def object_with_scale_buttons(frame, object, prev_function, next_function):
    button_prev = tk.Button(master=frame, text="Prev", command=prev_function)
    button_next = tk.Button(master=frame, text="Next", command=next_function)

    button_prev.grid(row=0, column=0, sticky="nsew")
    object.grid(row=0, column=1)
    button_next.grid(row=0, column=2, sticky="nsew")


def create_spinbox_grid(parent):
    frame = tk.Frame(parent)
    
    spinidx_boxes = []
    for i in range(2):
        for j in range(2):
            sb = tk.Spinbox(frame, from_=0, to=100)
            sb.grid(row=i, column=j, padx=5, pady=5)
            spinidx_boxes.append(sb)
    
    return frame, spinidx_boxes


def create_graph_canvas(fig):
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

button_auto_full_send = tk.Button(master=button_frame, text="AUTO FULL SEND", command=auto_process_data)
button_auto_full_send.pack(fill="both")

button_push_indices = tk.Button(master=button_frame, text="Push Indices", command=push_indices)
button_push_indices.pack(fill="both")

button_auto_params = tk.Button(master=button_frame, text="Auto_params", command=update_all_params)
button_auto_params.pack(fill="both")

button_finish_curve = tk.Button(master=button_frame, text="Done :)", command=save_data)
button_finish_curve.pack(fill="both")



slider_frame = tk.Frame(root)
slider_frame.grid(row=4, column=4)
slider_update = tk.Scale(slider_frame, from_=1, to=5, orient=tk.HORIZONTAL,
                            command=update_slider, label="Frame")
object_with_scale_buttons(slider_frame, slider_update, prev_frame, next_frame)




root.mainloop()