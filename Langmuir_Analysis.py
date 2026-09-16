"""
Langmuir_Analysis.py

Analysis routines for Langmuir probe plasma diagnostics: extracting
electron temperature (Te), plasma potential (Vp), and plasma density (n0)
from raw current-voltage sweep data, plus supporting curve-fitting,
data-cleaning, and physical-constant utilities.

Provides:
    - find_afterglow_peak, find_vf_idx: locate specific features
      (afterglow peak, floating potential index) within raw current
      traces.
    - clean_array, fit_poly, calculate_fit_goodness: general-purpose
      array cleaning and linear-fit utilities used throughout the
      analysis pipeline.
    - semilog_IV: converts a raw IV curve into semi-log form by
      subtracting ion saturation current and taking the natural log.
    - calculate_electrontemp, find_max_Vp_fit: fit Te and Vp from a
      semi-log IV curve, including automatic search for a Vp fit region.
    - getIonMass, getProbeArea, getPlasmaDensity: physical calculations
      converting fit results into plasma density, assuming a cylindrical
      probe geometry.
    - process_IV_curve: top-level pipeline combining the above into a
      single per-curve Te/Vp/n0 calculation, currently hardcoded to
      Argon's atomic weight regardless of input gas.

Module-level constants (SI units unless noted):
    N_AVOGADRO (float): Avogadro's number, atoms/mol.
    E_CHARGE (float): The charge of an electron. Defined here as *negative*
        (-1.602e-19).
    E_MASS (float): Electron mass, kg.
    K_BOLTZMANN (float): Boltzmann constant, J/K.
    E0 (float): Vacuum permittivity, F/m.
"""



import numpy as np
import math

N_AVOGADRO = 6.02214076 * 10 ** (23)
E_CHARGE = -1.60217663 * 10 ** (-19)
E_MASS = 9.1093837 * 10 ** (-31)
K_BOLTZMANN = 1.380649 * 10 ** (-23)
E0 = 8.854187 * 10 ** (-12)


def find_afterglow_peak(currents: np.ndarray) -> int:
    """
    Locate the index of peak afterglow signal in a 2D current array,
    ignoring the first 40 rows.

    Computes the mean across columns for each row from index 40 onward,
    finds the row with the maximum mean, and returns its index in the
    original (unsliced) array.

    Args:
        currents (1D np.ndarray): array of current traces (rows assumed
            to be time steps; first 40 rows excluded from the
            peak search).

    Returns:
        int: Index (into the original `currents` array) of the row with
            the maximum mean current after the first 40 rows.
    """
    row_means = currents[:][40:].mean(axis=1)
    idx = np.argmax(row_means) + 40
    return idx


def clean_array(xvals: np.ndarray, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove all inf, -inf, and NaN values from a numpy array.

    Args:
        xvals (1D np.ndarray):array of x values.
        arr (1D np.ndarray) : array of values which may include inf, -inf, and NaN.

    Returns:
        xvals (1D np.ndarray): array of x values corresponding to `arr`.
        arr (1D np.ndarray) : array of values with inf, -inf, and NaN values removed.
    """
    mask = np.isfinite(arr)
    return xvals[mask], arr[mask]


def fit_poly(
    curve_x: np.ndarray, curve_y: np.ndarray, bounds: list[int]
) -> tuple[np.ndarray, np.polynomial.Polynomial]:
    """
    Fit a first-degree (linear) polynomial to a subrange of x/y data and
    evaluate it over the full x range.

    Args:
        curve_x (1D np.ndarray): Full x-axis data.
        curve_y (1D np.ndarray): Full y-axis data (same length as curve_x).
        bounds (list[int, int]): (lower, upper) indices selecting the
            subrange of curve_x/curve_y used for the fit.

    Returns:
        y_fit (1D np.ndarray): the fitted line evaluated at every point in curve_x
        p (np.polynomial.Polynomial): the fitted Polynomial object
    """
    l, u = bounds[0], bounds[1]
    p = np.polynomial.Polynomial.fit(curve_x[l:u], curve_y[l:u], 1)
    y_fit = p(curve_x)
    return y_fit, p


def calculate_fit_goodness(
    data: np.ndarray, fit: np.ndarray, fit_indices: list[int]
) -> tuple[float, float]:
    """
    Compute R-squared and RMSE for a fit, restricted to a specified index
    range of the data.

    Args:
        data (np.ndarray): Full observed data array.
        fit (np.ndarray): Full fitted values array (same length as data).
        fit_indices (Sequence[int, int]): (lower, upper) indices
            restricting both `data` and `fit` to the range the fit
            quality is evaluated over.

    Returns:
        tuple[float, float]: (r_squared, rmse) computed over the sliced
            range.
    """
    _data = data[fit_indices[0] : fit_indices[1]]
    _fit = fit[fit_indices[0] : fit_indices[1]]

    ss_res = np.sum((_data - _fit) ** 2)
    ss_tot = np.sum((_data - np.mean(_data)) ** 2)
    return 1 - (ss_res / ss_tot), math.sqrt(ss_res / len(_data))


def semilog_IV(
    curve: list[np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Convert a raw IV curve into semi-log form by subtracting the ion
    saturation current and taking the natural log of the result.

    Estimates the ion saturation current as the average current over the
    range up to (find_vf_idx(curve[1]) - 3), subtracts it from the full
    current trace, takes the natural log, and removes any resulting
    inf/-inf/NaN entries (which occur where current - i_sat <= 0).

    Args:
        curve (list[1D np.ndarray, 1D np.ndarray]): [x_voltage, current] raw
            IV curve data, indexed positionally as curve[0]/curve[1].

    Returns:
        i_sat (float): calculated ion saturation current
        x_voltage (1D np.ndarray): array of x voltages corresponding to `ln_current`
        ln_current (1D np.ndarray): natural log of the current in the I-V curve, with bad (-inf, inf, NaN) values removed.
    """
    isat_end_index = find_vf_idx(curve[1]) - 3
    i_sat = np.average(curve[1][:isat_end_index])
    current = curve[1] - i_sat  # subtract ion saturation current so all non-negative vals
    ln_current = np.log(current)  # take natural log
    x_voltage, ln_current = clean_array(curve[0], ln_current)

    return i_sat, x_voltage, ln_current


def calculate_electrontemp(
    x_voltage: np.ndarray,
    ln_current: np.ndarray,
    Te_index: list[int],
    Vp_index: list[int],
) -> tuple[float, float, np.ndarray, np.ndarray, float, float]:
    """
    Fit electron temperature (Te) and plasma potential (Vp) from a
    semi-log IV curve.

    Fits a line to the Te-region data; the reciprocal of its slope gives
    Te. Fits a second line to the Vp-region data; Vp is taken as the
    x_voltage value where the Te fit and Vp fit lines are closest
    (their intersection point, found by nearest index rather than
    analytic intersection).

    Args:
        x_voltage (1D np.ndarray): Bias voltage values.
        ln_current (1D np.ndarray): Natural log of (current - i_sat).
        Te_index (list[int, int]): [lower, upper] indices selecting
            the data range used for the Te fit.
        Vp_index (list[int, int]): [lower, upper] indices selecting
            the data range used for the Vp fit.

    Returns:
        Te (float) : calculated electron temperature in eV.
        Vp (float) : calculated plasma potential in V.
        Te_fit (1D np.ndarray) : array of values denoting the linear Te fit.
        Vp_fit (1D np.ndarray) : array of values denoting the linear Vp fit.)
        Te_rsquared : r^2 coefficient for the Te fit.
        Vp_rmse : Roor Mean Square Error of the Vp fit.
    """
    Te_fit, Te_p = fit_poly(x_voltage, ln_current, Te_index)
    _intercept, slope = Te_p.convert().coef
    Te = 1.0 / slope
    Te_rsquared, Te_rmse = calculate_fit_goodness(ln_current, Te_fit, Te_index)

    Vp_fit, Vp_p = fit_poly(x_voltage, ln_current, Vp_index)
    intercept_i = np.argmin(abs(Vp_fit - Te_fit))
    Vp = x_voltage[intercept_i]
    Vp_rsquared, Vp_rmse = calculate_fit_goodness(ln_current, Vp_fit, Vp_index)

    return Te, Vp, Te_fit, Vp_fit, Te_rsquared, Vp_rmse


def find_max_Vp_fit(x_voltage: np.ndarray, ln_current: np.ndarray) -> list[int]:
    """
    Search for the largest upper-tail index range (near the end of the
    data) over which a linear Vp fit's RMSE is locally maximized, by
    decreasing the lower bound one step at a time until RMSE stops
    decreasing.

    Starts with a fixed 20-point window at the end of the data, fits a
    line, and repeatedly decrements the lower bound while the fit's RMSE
    keeps increasing; stops (and steps back by 2) once RMSE no longer
    increases.

    Args:
        x_voltage (1D np.ndarray): Bias voltage values.
        ln_current (1D np.ndarray): Natural log of (current - i_sat).

    Returns:
        list[int, int]: [lower, upper] index bounds into x_voltage /
            ln_current selecting the resulting Vp fit range.
    """
    max_indices = [len(ln_current) - 20, len(ln_current)]
    max_rmse = -1
    new_rmse = 0

    while max_rmse >= new_rmse:
        max_rmse = new_rmse
        Vp_fit, Vp_p = fit_poly(x_voltage, ln_current, max_indices)
        Vp_r2, new_rmse = calculate_fit_goodness(ln_current, Vp_fit, max_indices)
        max_indices[0] -= 1

    max_indices[0] += 2

    return max_indices


def getIonMass(atomic_weight: float) -> float:
    """
    Convert an atomic/molecular weight to ion mass in kg.

    Args:
        atomic_weight (float): Atomic or molecular weight in g/mol.

    Returns:
        float: Mass per ion in kg, computed as
            atomic_weight / N_AVOGADRO / 1000.
    """
    # atomic weight [g/mol] / [atoms/mol] / g/kg
    return atomic_weight / N_AVOGADRO / 1000


def getProbeArea(d: float, l: float) -> float:
    """
    Compute the effective surface area of a cylindrical Langmuir probe
    tip, including both the flat tip and the cylindrical shaft.

    Args:
        d (float): Probe diameter in meters.
        l (float): Length of probe tip, in meters.

    Returns:
        float: Total probe area (tip + shaft), in square meters.
    """
    a_tip = np.pi * (d / 2) ** 2
    a_shaft = np.pi * d * l
    return a_tip + a_shaft


def getPlasmaDensity(i_sat: float, e_temp: float, A: float, m_i: float) -> float:
    """
    Compute plasma density from ion saturation current using the Bohm
    sheath criterion.

    Args:
        i_sat (float): Ion saturation current (A).
        e_temp (float): Electron temperature (in the same energy-per-
            charge convention as E_CHARGE; see note below on sign
            requirements).
        A (float): Probe surface area.
        m_i (float): Ion mass in kg.

    Returns:
        float: Plasma density, or -1 if the intermediate sound-speed
            calculation is invalid (i.e. computing math.sqrt of a
            negative number raises ValueError).
    """
    try:
        c_s = math.sqrt(
            (e_temp * -E_CHARGE) / m_i
        )  # ions are around room temp, use electrode temp readings
        return i_sat / (c_s * E_CHARGE * A)
    except ValueError:
        return -1


def find_vf_idx(current: np.ndarray) -> int:
    """
    Find the last index within the first 50 samples where current is
    non-positive, used as an estimate of the floating potential index.

    Args:
        current (np.ndarray): Current trace.

    Returns:
        int: Index of the last non-positive value within current[:50].

    Raises:
        IndexError: If no non-positive values exist within the first 50
            samples (empty result from np.where).
    """
    return np.where(current[:50] <= 0)[0][-1]


def process_IV_curve(
    i_sat: float,
    x_voltage: np.ndarray,
    ln_current: np.ndarray,
    i_Te: list[int],
    i_Vp: list[int],
    probe_width: float,
    probe_len: float,
) -> tuple[float, float, float, np.ndarray | None, np.ndarray | None, float, float]:
    """
    Run the full electron-temperature/plasma-potential/density pipeline
    on a single semi-log IV curve, assuming a fixed ion species (argon).

    Args:
        i_sat (float): Ion saturation current for this curve (A).
        x_voltage (1D np.ndarray): Bias voltage values (V).
        ln_current (1D np.ndarray): Natural log of (current - i_sat).
        i_Te (list[int, int]): (lower, upper) indices for the Te fit
            region.
        i_Vp (list[int, int]): (lower, upper) indices for the Vp fit
            region.
        probe_width (float): Probe diameter, passed to getProbeArea.
        probe_len (float): Probe shaft length, passed to getProbeArea.

    Returns:
        tuple: (Te, Vp, n0, Te_fit, Vp_fit, Te_r2, Vp_rmse) on success,
            where n0 (plasma density) is scaled by 1e-6. Returns
            (0, 0, 0, None, None, 0, 0) if a ValueError propagates up
            from the calculation (e.g. from an invalid sqrt in
            getPlasmaDensity).
    """

    try:
        Te, Vp, Te_fit, Vp_fit, Te_r2, Vp_rmse = calculate_electrontemp(
            x_voltage, ln_current, i_Te, i_Vp
        )

        area_probe = getProbeArea(probe_width, probe_len)

        mi = getIonMass(39.95)
        n0 = getPlasmaDensity(i_sat, Te, area_probe, mi) / 10**6

        return Te, Vp, n0, Te_fit, Vp_fit, Te_r2, Vp_rmse
    except ValueError:
        return 0, 0, 0, None, None, 0, 0
