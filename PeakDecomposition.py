import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simps
import  os

from matplotlib.ticker import AutoMinorLocator, NullLocator, MultipleLocator

from lmfit.models import ExpressionModel
import time

# Define a 1-dimensional Gaussian function for peak shape modeling.
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-((x - cen) / wid) ** 2)

# Calculate the Root Mean Square Error (RMSE) between target and predicted values.
def RMSE(target, prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))

    Rmse = np.sqrt(sum(squaredError) / len(squaredError))

    return Rmse

# Evaluate the goodness of fit for a tentative peak region, retaining regions even if R-squared is exceptionally low.
def check_peak_fit_quality(temperature, signal, peak_indices, r_squared_threshold=0.5):
    if len(peak_indices) < 10:
        return False, 0

    try:
        x_peak = temperature[peak_indices]
        y_peak = signal[peak_indices]

        if len(x_peak) != len(y_peak) or len(x_peak) == 0:
            return False, 0

        signal_range = np.max(y_peak) - np.min(y_peak)
        if signal_range < 0.001:
            return False, 0

        peak_idx = np.argmax(y_peak)
        a_guess = y_peak[peak_idx]
        m_guess = x_peak[peak_idx]
        s_guess = (max(x_peak) - min(x_peak)) / 10

        AmpMin, AmpMax = 0.01, np.max(y_peak) * 10
        CenMin, CenMax = min(x_peak), max(x_peak)
        WidMin, WidMax = 0.1, (max(x_peak) - min(x_peak)) / 2

        popt, pcov = curve_fit(
            gaussian, x_peak, y_peak,
            p0=[a_guess, m_guess, s_guess],
            bounds=([AmpMin, CenMin, WidMin], [AmpMax, CenMax, WidMax]),
            maxfev=5000
        )

        y_pred = gaussian(x_peak, *popt)
        residuals = y_peak - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_peak - np.mean(y_peak)) ** 2)

        if ss_tot == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        if r_squared < -1:
            is_valid = True
            print(f"Note: R² = {r_squared:.4f} < -1, but it is identified as a valid peak region.")
        else:
            is_valid = r_squared >= r_squared_threshold

        return is_valid, r_squared

    except Exception as e:
        print(f"Goodness of fit check error: {e}")
        return False, 0

# Perform single Gaussian model fitting on the provided peak signal and visualize the results.
def PeakDecom(x, y):
    print("***Peak Fitting......")
    x = np.array(x)
    x = x.tolist()
    y = np.array(y)
    cen = min(x)
    Gmod = ExpressionModel("amp * exp(-(x-cen)**2 /(2*wid**2))/(sqrt(2*pi)*wid)")
    result = Gmod.fit(y, x=x, amp=1, cen=cen, wid=0.5)
    print(result.fit_report())

    plt.figure(dpi=600, figsize=(8, 6))
    plt.plot(x, y, 'C2.', label='Net Signal')
    plt.plot(x, result.best_fit, 'C3-', label='Gaussian fitting')

    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Temperature (℃)", size=22, labelpad=10)
    plt.ylabel("Heat capacity (KJ/mol/K)", size=22, labelpad=10)
    plt.tick_params(labelsize=16)
    plt.show()

    GauBestFit = result.best_fit
    Rmse = RMSE(y, GauBestFit)
    print("RMSE = ", Rmse)

    print(" Enthalpy change of Gaussian fitting 1:", simps(GauBestFit, x))

# Define a 2-dimensional Gaussian function for modeling multiple overlapping peaks.
def func2(x, a1, a2, m1, m2, s1, s2):
    return a1 * np.exp(-((x - m1) / s1) ** 2) + a2 * np.exp(-((x - m2) / s2) ** 2)

# Fit a multiple Gaussian model to the peak signal, visualize the output, and compute transition parameters.
def MulPeakDecom(x, y, savepath_b, normaTimes):
    print("***Peak Fitting......")
    x = np.array(x)
    y = np.array(y)

    AmpMin = 0.01
    AmpMax = 20000
    CenMin = min(x)
    CenMax = max(x)
    WidMin = 0.1
    WidMax = 100

    popt, pcov = curve_fit(func2, x, y,
                           bounds=([AmpMin, AmpMin, CenMin, CenMin, WidMin, WidMin],
                                   [AmpMax, AmpMax, CenMax, CenMax, WidMax, WidMax]))

    times = normaTimes
    plt.figure(dpi=300, figsize=(8, 6))
    plt.plot(x, y * times, "C2.", label='Net Signal')
    plt.plot(x, func2(x, *popt) * times, 'C3-', label='Gaussian Fitting')
    plt.plot(x, gaussian(x, popt[0], popt[2], popt[4]) * times, 'C4--', label='Gausssian 1')
    plt.plot(x, gaussian(x, popt[1], popt[3], popt[5]) * times, 'C5--', label='Gausssian 2')

    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Temperature (℃)", size=20, labelpad=10)
    plt.ylabel("Heat capacity (KJ/mol/K)", size=20, labelpad=10)
    plt.tick_params(labelsize=16)
    img_path = savepath_b + '.png'
    plt.savefig(img_path)
    plt.show()

    GauBestFit = gaussian(x, popt[0], popt[2], popt[4]) + gaussian(x, popt[1], popt[3], popt[5])
    Rmse = RMSE(y, GauBestFit)

    deltaH_1 = simps(gaussian(x, popt[0], popt[2], popt[4]), x)
    deltaH_2 = simps(gaussian(x, popt[1], popt[3], popt[5]), x)
    deltaH = []
    deltaH.append(deltaH_1)
    deltaH.append(deltaH_2)
    Tm = []
    Tm.append(popt[2])
    Tm.append(popt[3])

    return Tm, deltaH

# Plot raw data after Gaussian fitting and calculate the transition temperature (Tm) and enthalpy change (deltaH).
def plot_raw_data_and_calculate(x, y, savepath, normaTimes, peak_number=1):
    print(f"Start processing peak #{peak_number}...")
    x = np.array(x)
    y = np.array(y)

    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        print(f"Peak #{peak_number}: Invalid data, skipping processing.")
        return 0, 0, 0

    base_savepath = f"{savepath}_peak{peak_number}"

    Tm = 0
    deltaH = 0
    r_squared = 0
    fit_successful = False
    a_fit, m_fit, s_fit = 0, 0, 0
    y_pred = None

    try:
        deltaH = simps(y * normaTimes, x)
        print(f"Peak #{peak_number}: ΔH calculation completed - {deltaH:.4f}J/g")
    except Exception as e:
        print(f"Peak #{peak_number}: ΔH calculation failed - {e}")
        deltaH = 0

    try:
        AmpMin, AmpMax = 0.01, 20000
        CenMin, CenMax = min(x), max(x)
        WidMin, WidMax = 0.1, 100

        peak_idx = np.argmax(y)
        a_guess = y[peak_idx]
        m_guess = x[peak_idx]
        s_guess = (max(x) - min(x)) / 10

        print(f"Peak #{peak_number}: Initial parameters - Amplitude={a_guess:.4f}, Center={m_guess:.2f}℃, Width={s_guess:.2f}")

        popt, pcov = curve_fit(
            gaussian, x, y,
            p0=[a_guess, m_guess, s_guess],
            bounds=([AmpMin, CenMin, WidMin], [AmpMax, CenMax, WidMax]),
            maxfev=10000
        )

        a_fit, m_fit, s_fit = popt

        y_pred = gaussian(x, a_fit, m_fit, s_fit)

        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        Tm = m_fit
        fit_successful = True
        print(f"Peak #{peak_number}: Gaussian fitting successful - Tm={Tm:.2f}℃, R²={r_squared:.4f}")

    except Exception as e:
        print(f"Peak #{peak_number}: Gaussian fitting failed - {e}")
        if len(y) > 0:
            Tm = x[np.argmax(y)]
            print(f"Peak #{peak_number}: Using peak temperature - Tm={Tm:.2f}℃")
        else:
            Tm = 0
        r_squared = 0

    try:
        plt.rcParams['font.family'] = 'Times New Roman'

        fig, ax = plt.subplots(dpi=400, figsize=(8, 6), facecolor='none')
        ax.set_facecolor('none')

        ax.plot(x, y * normaTimes, color='#2B6299', linewidth=3, label='Net signal')

        if fit_successful and y_pred is not None:
            ax.plot(x, y_pred * normaTimes, color='#FE7A15', linestyle='--',
                    linewidth=2, label='Gaussian fitting')

        def setup_axes_ticks(ax, hide_top=True, hide_right=True):
            ax.minorticks_on()
            ax.tick_params(axis='x', which='major', direction='in', bottom=True, top=hide_top,
                           labelsize=28, length=8, width=1.5, pad=10)
            ax.tick_params(axis='y', which='major', direction='in', left=True, right=hide_right,
                           labelsize=28, length=8, width=1.5, pad=10)
            ax.tick_params(axis='x', which='minor', direction='in', bottom=True, top=hide_top,
                           length=5, width=1.2)
            ax.tick_params(axis='y', which='minor', direction='in', left=True, right=hide_right,
                           length=5, width=1.2)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            if hide_top:
                ax.xaxis.set_ticks_position('bottom')
            if hide_right:
                ax.yaxis.set_ticks_position('left')

        setup_axes_ticks(ax)

        ax.yaxis.set_major_locator(MultipleLocator(2))

        ax.set_xlabel("Temperature (℃)", size=28, labelpad=6, weight='normal')
        ax.set_ylabel("Heat capacity (J/g/K)", size=28, labelpad=6, weight='normal')

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily('Times New Roman')
            tick.set_fontsize(28)

        legend = ax.legend(
            loc='upper left',
            bbox_to_anchor=(-0.03, 1.05),
            fontsize=22,
            frameon=False,
            ncol=1
        )
        for text in legend.get_texts():
            text.set_fontfamily('Times New Roman')

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.grid(False)

        if fit_successful:
            title_info = f'Peak #{peak_number}: Tm={Tm:.2f}℃, ΔH={deltaH:.4f}J/g, R²={r_squared:.4f}'
        else:
            title_info = f'Peak #{peak_number}: Tm={Tm:.2f}℃, ΔH={deltaH:.4f}J/g'

        ax.set_title(title_info, fontsize=16, pad=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        os.makedirs(os.path.dirname(base_savepath), exist_ok=True)

        plt.savefig(f"{base_savepath}.svg", dpi=300, bbox_inches='tight', transparent=True)

        print(f"Peak #{peak_number}: Image saved to {base_savepath}.svg")
        plt.close(fig)

    except Exception as e:
        print(f"Peak #{peak_number}: Visualization failed - {e}")

    print(f"Peak #{peak_number}: Processing completed - Tm={Tm:.2f}℃, ΔH={deltaH:.4f}J/g")
    return Tm, deltaH, r_squared