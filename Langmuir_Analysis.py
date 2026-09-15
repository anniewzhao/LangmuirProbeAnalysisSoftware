import numpy as np
import math


N_AVOGADRO = 6.02214076 * 10 ** (23)
E_CHARGE = -1.60217663 * 10 ** (-19)
E_MASS = 9.1093837 * 10 ** (-31)
K_BOLTZMANN = 1.380649 * 10 ** (-23)
E0 = 8.854187 * 10 ** (-12)



def find_afterglow_peak(times, currents):
    row_means = currents[:][40:].mean(axis=1)
    idx = np.argmax(row_means)+40
    return idx


def clean_array(xvals, arr):
    """Remove all inf, -inf, and NaN values from a numpy array."""
    mask = np.isfinite(arr)
    return xvals[mask], arr[mask]


def fit_poly(curve_x, curve_y, bounds):
    l, u = bounds[0], bounds[1]
    p = np.polynomial.Polynomial.fit(curve_x[l:u], curve_y[l:u], 1)
    y_fit = p(curve_x)
    return y_fit, p


def calculate_fit_goodness(data, fit, fit_indices):
    _data = data[fit_indices[0]:fit_indices[1]]
    _fit = fit[fit_indices[0]:fit_indices[1]]

    ss_res = np.sum((_data - _fit) ** 2)
    ss_tot = np.sum((_data - np.mean(_data)) ** 2)
    return 1 - (ss_res / ss_tot), math.sqrt(ss_res/len(_data))


def semilog_IV(curve):
    isat_end_index = find_vf_idx(curve[1])-3
    i_sat = np.average(curve[1][:isat_end_index])
    current = curve[1] - i_sat  # subtract ion saturation current so all non-negative vals
    ln_current = np.log(current)  # take natural log
    x_voltage, ln_current = clean_array(curve[0], ln_current)

    return i_sat, x_voltage, ln_current



def calculate_electrontemp(x_voltage, ln_current, Te_index, Vp_index):
    Te_fit, Te_p = fit_poly(x_voltage, ln_current, Te_index)
    _intercept, slope = Te_p.convert().coef
    Te = 1.0 / slope
    Te_rsquared, Te_rmse = calculate_fit_goodness(ln_current, Te_fit, Te_index)

    Vp_fit, Vp_p = fit_poly(x_voltage, ln_current, Vp_index)
    intercept_i = np.argmin(abs(Vp_fit - Te_fit))
    Vp = x_voltage[intercept_i]
    Vp_rsquared, Vp_rmse = calculate_fit_goodness(ln_current, Vp_fit, Vp_index)

    return Te, Vp, Te_fit, Vp_fit, Te_rsquared, Vp_rmse



def find_max_Vp_fit(x_voltage, ln_current):
    max_indices = [len(ln_current)-15, len(ln_current)]
    max_rmse = -1
    new_rmse = 0

    while max_rmse < new_rmse:
        max_rmse = new_rmse
        Vp_fit, Vp_p = fit_poly(x_voltage, ln_current, max_indices)
        Vp_r2, new_rmse = calculate_fit_goodness(ln_current, Vp_fit, max_indices)
        max_indices[0] -=1

    max_indices[0]+=2

    return max_indices



def getIonMass(atomic_weight):
    # atomic weight [g/mol] / [atoms/mol] / g/kg
    return atomic_weight / N_AVOGADRO / 1000


def getProbeArea(d, l):
    a_tip = np.pi * (d / 2) ** 2
    a_shaft = np.pi * d * l
    return a_tip + a_shaft


def getPlasmaDensity(i_sat, e_temp, A, m_i):
    try:
        c_s = math.sqrt(
            (e_temp * -E_CHARGE) / m_i
        )  # ions are around room temp, use electrode temp readings
        return i_sat / (c_s * E_CHARGE * A)
    except ValueError:
        return -1

    
def find_vf_idx(current):
    return np.where(current[:50] <= 0)[0][-1]


def process_IV_curve(i_sat, x_voltage, ln_current, i_Te, i_Vp, probe_width, probe_len):

    try:
        Te, Vp, Te_fit, Vp_fit, Te_r2, Vp_r2 = calculate_electrontemp(x_voltage, ln_current, i_Te, i_Vp)

        area_probe = getProbeArea(probe_width, probe_len)

        mi = getIonMass(39.95)
        n0 = getPlasmaDensity(i_sat, Te, area_probe, mi) / 10**6

        return Te, Vp, n0, Te_fit, Vp_fit, Te_r2, Vp_r2
    except ValueError:
        return 0, 0, 0, None, None, 0, 0

