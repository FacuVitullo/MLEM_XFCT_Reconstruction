import os
import glob
import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, SpanSelector

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
TOTAL_CHANNELS = 2048
MAX_ENERGY_KEV = 218.0

# Initial ROI factor (k * sigma)
INIT_ROI_SIGMA = 2.0


# =============================================================================
# ANALYTICAL HELPER FUNCTIONS
# =============================================================================

def gaussian(x, amp, mean, sigma):
    """Standard 1D Gaussian curve."""
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def fit_manual_roi(channels, spectrum, ch_min, ch_max, roi_sigma_factor=INIT_ROI_SIGMA):
    """
    Fits a Gaussian to user-selected channel bounds [ch_min, ch_max]:
    1. Removes a linear baseline connecting end points.
    2. Fits a 1D Gaussian model to the net spectrum in the region.
    3. Calculates net counts and ROI bounds based on the Gaussian sigma (in channels).
    """
    idx_mask = (channels >= ch_min) & (channels <= ch_max)
    sub_ch = channels[idx_mask]
    sub_spec = spectrum[idx_mask]

    if len(sub_ch) < 4:
        return None

    # --- 1. Linear Baseline Removal ---
    ch_start, ch_end = sub_ch[0], sub_ch[-1]
    y_start, y_end = np.mean(sub_spec[:2]), np.mean(sub_spec[-2:])
    
    slope = (y_end - y_start) / (ch_end - ch_start) if (ch_end - ch_start) != 0 else 0
    intercept = y_start - slope * ch_start
    baseline = slope * sub_ch + intercept

    net_spec = sub_spec - baseline
    net_spec = np.maximum(net_spec, 0)  # Clip negative noise

    # --- 2. Gaussian Fitting Setup ---
    amp_guess = np.max(net_spec)
    mean_guess = sub_ch[np.argmax(net_spec)]
    sigma_guess = (ch_max - ch_min) / 6.0  # Initial guess (~6 sigma span)

    if amp_guess <= 0:
        return None

    try:
        popt, _ = curve_fit(
            gaussian, 
            sub_ch, 
            net_spec, 
            p0=[amp_guess, mean_guess, sigma_guess],
            bounds=([0, ch_min, 0.1], [np.inf, ch_max, (ch_max - ch_min)])
        )
        fit_amp, fit_mean, fit_sigma = popt

        # --- 3. ROI Calculation ---
        roi_min = fit_mean - (roi_sigma_factor * fit_sigma)
        roi_max = fit_mean + (roi_sigma_factor * fit_sigma)
        
        roi_mask = (sub_ch >= roi_min) & (sub_ch <= roi_max)
        net_counts_roi = np.sum(net_spec[roi_mask])

        # Convert mean/sigma/ROI to Energy (keV) for reference display
        keV_per_channel = MAX_ENERGY_KEV / TOTAL_CHANNELS
        mean_kev = fit_mean * keV_per_channel
        sigma_kev = fit_sigma * keV_per_channel
        fwhm_kev = 2.355 * sigma_kev

        return {
            'sub_ch': sub_ch,
            'net_spec': net_spec,
            'baseline': baseline,
            'fit_curve': gaussian(sub_ch, *popt),
            'mean_ch': fit_mean,
            'sigma_ch': fit_sigma,
            'fwhm_ch': 2.355 * fit_sigma,
            'mean_kev': mean_kev,
            'sigma_kev': sigma_kev,
            'fwhm_kev': fwhm_kev,
            'roi_bounds_ch': (roi_min, roi_max),
            'net_counts': net_counts_roi,
            'manual_bounds': (ch_min, ch_max)
        }
    except Exception:
        return None


# =============================================================================
# MAIN EXECUTION & INTERACTIVE VISUALIZER
# =============================================================================

if __name__ == "__main__":
    ANGLE_DIR = r"C:\Users\facun\Desktop\Penelope2\3_metales\CorrectedData\angle_72"

    if os.path.exists(ANGLE_DIR):
        file_paths = sorted(glob.glob(os.path.join(ANGLE_DIR, "*.dat")))

        if not file_paths:
            print(f"No .dat files found in directory: {ANGLE_DIR}")
        else:
            raw_datasets = []
            print(f"Loading {len(file_paths)} spectra...")
            for file_path in file_paths:
                spectrum = np.loadtxt(file_path)
                spectrum[:10] = 0.0
                channels = np.arange(len(spectrum))
                
                raw_datasets.append({
                    'file_name': os.path.basename(file_path),
                    'channels': channels,
                    'spectrum': spectrum
                })

            fig, ax = plt.subplots(figsize=(13, 8))
            plt.subplots_adjust(bottom=0.25, top=0.88)

            curr_data = raw_datasets[0]
            line_spec, = ax.plot(curr_data['channels'], curr_data['spectrum'], color='#1f77b4', lw=1.2, label='Raw Spectrum')
            
            # --- Secondary X-Axis for Energy (keV) ---
            ax_energy = ax.secondary_xaxis(
                'top', 
                functions=(lambda ch: ch * (MAX_ENERGY_KEV / TOTAL_CHANNELS), 
                           lambda kev: kev * (TOTAL_CHANNELS / MAX_ENERGY_KEV))
            )
            ax_energy.set_xlabel("Energy Reference (keV)", labelpad=8)

            # Persistent selection state
            selected_range = [None, None]  # [ch_min, ch_max]
            
            fit_lines = []
            baseline_lines = []
            roi_spans = []

            # Live ROI Data Panel Box (Top Left)
            info_text = ax.text(
                0.02, 0.95, "Click and drag on the plot to select a peak region (in channels).", transform=ax.transAxes, 
                verticalalignment='top', fontsize=8, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='gray', alpha=0.9)
            )

            # -----------------------------------------------------------------
            # SLIDERS CONFIGURATION
            # -----------------------------------------------------------------
            ax_file  = plt.axes([0.20, 0.12, 0.65, 0.022])
            ax_sigma = plt.axes([0.20, 0.06, 0.65, 0.022])

            s_file  = Slider(ax_file, 'File Index', 0, len(raw_datasets) - 1, valinit=0, valfmt='%d')
            s_sigma = Slider(ax_sigma, 'ROI Factor (xσ)', 1.0, 4.0, valinit=INIT_ROI_SIGMA, valstep=0.1)

            def update_plot(val=None):
                idx = int(s_file.val)
                sig = s_sigma.val

                data = raw_datasets[idx]
                channels = data['channels']
                spectrum = data['spectrum']

                line_spec.set_ydata(spectrum)

                # Clear previous fit elements
                while fit_lines: fit_lines.pop().remove()
                while baseline_lines: baseline_lines.pop().remove()
                while roi_spans: roi_spans.pop().remove()

                summary_lines = [f"FILE: {data['file_name']}"]
                summary_lines.append("-" * 48)

                # Perform manual fit if a valid channel range is selected
                if selected_range[0] is not None and selected_range[1] is not None:
                    ch_min, ch_max = selected_range
                    fit_info = fit_manual_roi(channels, spectrum, ch_min, ch_max, roi_sigma_factor=sig)

                    if fit_info:
                        sub_ch = fit_info['sub_ch']

                        # Plot Baseline
                        bl, = ax.plot(sub_ch, fit_info['baseline'], '--', color='red', lw=1.2, alpha=0.8)
                        baseline_lines.append(bl)

                        # Plot Gaussian Fit
                        full_fit = fit_info['fit_curve'] + fit_info['baseline']
                        fl, = ax.plot(sub_ch, full_fit, '-', color='orange', lw=2.0)
                        fit_lines.append(fl)

                        # Plot calculated ROI span
                        r_min, r_max = fit_info['roi_bounds_ch']
                        span = ax.axvspan(r_min, r_max, color='green', alpha=0.25)
                        roi_spans.append(span)

                        # Construct output info summary (Channels + Energy Ref)
                        summary_lines.append(f"Selection: [{ch_min:.0f} - {ch_max:.0f}] ch")
                        summary_lines.append(
                            f"μ = {fit_info['mean_ch']:.1f} ch ({fit_info['mean_kev']:.2f} keV) | "
                            f"σ = {fit_info['sigma_ch']:.2f} ch | FWHM = {fit_info['fwhm_ch']:.2f} ch\n"
                            f"ROI ({sig:.1f}σ): [{r_min:.1f} - {r_max:.1f}] ch | Net Cts: {int(fit_info['net_counts'])}"
                        )
                    else:
                        summary_lines.append("Could not fit Gaussian to selected region. Try a wider selection.")
                else:
                    summary_lines.append("Drag mouse on plot to select peak region (start -> end).")

                info_text.set_text("\n".join(summary_lines))
                ax.set_title(f"File ({idx + 1}/{len(raw_datasets)}): {data['file_name']}", fontweight='bold', pad=25)
                
                fig.canvas.draw_idle()

            def on_select(xmin, xmax):
                """Callback function when a user drags a span over the spectrum."""
                if abs(xmax - xmin) > 2:  # Minimum 2 channels width
                    selected_range[0] = min(xmin, xmax)
                    selected_range[1] = max(xmin, xmax)
                    update_plot()

            # Enable SpanSelector (Drag mouse over graph)
            span_selector = SpanSelector(
                ax, on_select, 'horizontal', useblit=True,
                props=dict(alpha=0.3, facecolor='red'),
                interactive=True, drag_from_anywhere=True
            )

            s_file.on_changed(update_plot)
            s_sigma.on_changed(update_plot)

            ax.set_xlabel("Channel Index")
            ax.set_ylabel("Counts")
            # Default initial view range (channels equivalent of 5–55 keV)
            init_ch_min = int(5.0 * TOTAL_CHANNELS / MAX_ENERGY_KEV)
            init_ch_max = int(55.0 * TOTAL_CHANNELS / MAX_ENERGY_KEV)
            ax.set_xlim(init_ch_min, init_ch_max)
            ax.grid(True, alpha=0.3)

            # Legend markers
            ax.plot([], [], '--', color='red', label='Linear Baseline')
            ax.plot([], [], '-', color='orange', label='Gaussian Fit')
            ax.axvspan(0, 0, color='green', alpha=0.25, label=r'Calculated ROI ($\mu \pm k\sigma$)')
            ax.legend(loc='upper right')

            update_plot()
            plt.show()
    else:
        print(f"Directory not found: {ANGLE_DIR}")