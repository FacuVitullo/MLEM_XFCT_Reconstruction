import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# =============================================================================
# CONFIGURATION
# =============================================================================
# Set path to a specific angle directory to inspect
ANGLE_FOLDER = r"C:\Users\facun\Desktop\Penelope2\3_metales\CorrectedData\angle_0"

# Initial MORALS parameters
INIT_LAMBDA_EXP = 6.0  # Represents 10^6
INIT_P = 0.05
INIT_ITER = 10


def natural_key(string_):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', string_)]


def compute_morals_baseline(spectrum, lam, p, max_iter):
    """Computes MORALS baseline for a 1D spectrum."""
    num_channels = len(spectrum)
    D = diags([1, -2, 1], [0, 1, 2], shape=(num_channels - 2, num_channels), dtype=float)
    DTD_scaled = lam * (D.T.dot(D))

    w = np.ones(num_channels)
    baseline = np.zeros(num_channels)

    for _ in range(max_iter):
        W = diags(w, 0, dtype=float)
        A_mat = W + DTD_scaled
        baseline = spsolve(A_mat, w * spectrum)
        w = np.where(spectrum > baseline, p, 1.0 - p)

    clean = np.maximum(0.0, spectrum - baseline)
    return baseline, clean


def launch_morals_inspector(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: Folder path not found -> {folder_path}")
        return

    # Find and sort all .dat files in the angle folder
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.dat')],
        key=natural_key
    )

    if not files:
        print(f"No .dat files found in {folder_path}")
        return

    print(f"Loaded {len(files)} position spectra from: {os.path.basename(folder_path)}")

    # Pre-load raw spectral data across positions
    raw_spectra = [np.loadtxt(os.path.join(folder_path, f)) for f in files]
    channels = np.arange(len(raw_spectra[0]))

    # Setup Main Figure
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.72, top=0.9)

    # Initial Computation for Position 0
    init_pos = 0
    init_lam = 10**INIT_LAMBDA_EXP
    raw_0 = raw_spectra[init_pos]
    base_0, clean_0 = compute_morals_baseline(raw_0, init_lam, INIT_P, INIT_ITER)

    # Lines
    line_raw, = ax1.plot(channels, raw_0, color='#1f77b4', lw=1.2, label='Raw Spectrum')
    line_base, = ax1.plot(channels, base_0, color='#ff7f0e', lw=1.2, ls='--', label='Estimated Baseline')
    line_clean, = ax2.plot(channels, clean_0, color='#2ca02c', lw=1.2, label='Clean (Subtracted)')

    # Labels and Axes Formatting
    ax1.set_xlabel("Channel Index")
    ax1.set_ylabel("Raw / Baseline Counts", color='#1f77b4')
    ax2.set_ylabel("Clean Counts (Subtracted)", color='#2ca02c')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax1.grid(True, alpha=0.3)

    lines = [line_raw, line_base, line_clean]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    title_text = fig.suptitle(
        f"File [{init_pos + 1}/{len(files)}]: {files[init_pos]}\n"
        f"MORALS (λ = 10^{INIT_LAMBDA_EXP:.1f}, p = {INIT_P:.3f}, iter = {INIT_ITER})",
        fontsize=11
    )

    # --- SLIDERS LAYOUT ---
    # Bottom Slider: Position Index
    ax_pos = plt.axes([0.15, 0.06, 0.55, 0.03])
    s_pos = Slider(ax_pos, 'Position Index', 0, len(files) - 1, valinit=0, valfmt='%d')

    # Right Sliders: MORALS Parameters
    ax_lam = plt.axes([0.80, 0.65, 0.12, 0.03])
    s_lam = Slider(ax_lam, 'log10(λ)', 2.0, 10.0, valinit=INIT_LAMBDA_EXP, valstep=0.1)

    ax_p = plt.axes([0.80, 0.50, 0.12, 0.03])
    s_p = Slider(ax_p, 'p (Asym)', 0.001, 0.2, valinit=INIT_P, valstep=0.005)

    ax_iter = plt.axes([0.80, 0.35, 0.12, 0.03])
    s_iter = Slider(ax_iter, 'Iterations', 1, 30, valinit=INIT_ITER, valfmt='%d')

    # --- UPDATE CALLBACK ---
    def update(val):
        pos_idx = int(s_pos.val)
        lam_val = 10**s_lam.val
        p_val = s_p.val
        iter_val = int(s_iter.val)

        current_raw = raw_spectra[pos_idx]
        baseline, clean = compute_morals_baseline(current_raw, lam_val, p_val, iter_val)

        # Update Plot Lines
        line_raw.set_ydata(current_raw)
        line_base.set_ydata(baseline)
        line_clean.set_ydata(clean)

        # Rescale Axes
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        # Update Title
        title_text.set_text(
            f"File [{pos_idx + 1}/{len(files)}]: {files[pos_idx]}\n"
            f"MORALS (λ = 10^{s_lam.val:.1f}, p = {p_val:.3f}, iter = {iter_val})"
        )
        fig.canvas.draw_idle()

    # Attach listeners to all sliders
    s_pos.on_changed(update)
    s_lam.on_changed(update)
    s_p.on_changed(update)
    s_iter.on_changed(update)

    plt.show()


if __name__ == "__main__":
    launch_morals_inspector(ANGLE_FOLDER)