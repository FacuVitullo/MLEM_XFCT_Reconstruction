import io
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import iradon, radon

# =============================================================================
# DATA & GEOMETRY CONFIGURATION
# =============================================================================
ANGLES = np.arange(0, 348 + 12, 12)  # 30 Angles: [0, 12, 24, ..., 348]
N_POSITIONS = 41  # 41 Translational positions
FOV_SIZE_MM = 40.0  # 40 mm translation width


def natural_key(string_):
    return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", string_)]


# =============================================================================
# INTEGRATED MLEM ITERATIVE DIAGNOSTIC LOOP WITH GIF EXPORT
# =============================================================================
def mlem_iterative_debug_loop(
    spectrum_matrix,
    sensitivity_angles=None,
    max_iterations=20,
    tolerance_epsilon=1e-3,
    channel_range=(119, 137),
    use_attenuation=False,
    plot_live_iterations=True,
    plot_interval=1,
    save_gif=True,
    gif_filename="mlem_reconstruction.gif",
    frame_duration_ms=200,  # 200 ms per frame = 5 FPS
):
    """Runs MLEM and exports the iterative reconstruction as an animated GIF."""
    num_angles, num_translations, total_channels = spectrum_matrix.shape
    angles = ANGLES
    image_size = num_translations

    start_ch, end_ch = channel_range
    start_ch = max(0, start_ch)
    end_ch = min(total_channels, end_ch)

    print(
        f"[MLEM] Integrating energy channels {start_ch} to {end_ch - 1} into a single sinogram..."
    )

    # 1. Sum measured sinograms across the selected channel range
    Y = np.sum(spectrum_matrix[:, :, start_ch:end_ch], axis=2).T

    # 2. Setup Geometry & Slicing Indices
    sample_ones = np.ones((image_size, image_size))
    sample_radon = radon(sample_ones, theta=angles, circle=False)
    skimage_sinogram_rows = sample_radon.shape[0]

    row_diff = skimage_sinogram_rows - num_translations
    start_row = row_diff // 2
    end_row = start_row + num_translations

    extent_fov = [
        -FOV_SIZE_MM / 2,
        FOV_SIZE_MM / 2,
        -FOV_SIZE_MM / 2,
        FOV_SIZE_MM / 2,
    ]
    extent_sino = [
        ANGLES[0],
        ANGLES[-1],
        -FOV_SIZE_MM / 2,
        FOV_SIZE_MM / 2,
    ]

    # 3. Handle Attenuation Map and Sensitivity Calculation
    if use_attenuation:
        if sensitivity_angles is None:
            raise ValueError(
                "sensitivity_angles matrix must be provided when use_attenuation is True."
            )
        sensitivity_image = np.sum(sensitivity_angles, axis=2)
    else:
        ones_sinogram = np.ones((skimage_sinogram_rows, num_angles))
        sensitivity_image = iradon(
            ones_sinogram, theta=angles, filter_name=None, circle=False
        )

    sensitivity_image[sensitivity_image == 0] = 1e-12

    # 4. Initial uniform estimate X(0)
    initial_val = np.mean(Y) / image_size if np.mean(Y) > 0 else 1e-3
    X_initial = np.ones((image_size, image_size)) * initial_val
    X_current = X_initial.copy()

    iteration_numbers = []
    rel_change_history = []
    gif_frames = []

    forward_projection = None
    final_it = 0

    # Setup Plot Window for capturing frames
    if plot_live_iterations or save_gif:
        plt.ion()
        fig_live, ax_live = plt.subplots(figsize=(6, 5), dpi=120)
        im_live = ax_live.imshow(
            X_current, cmap="hot", origin="lower", extent=extent_fov
        )
        cbar_live = fig_live.colorbar(im_live, ax=ax_live, label="Counts")
        ax_live.set_xlabel("X (mm)")
        ax_live.set_ylabel("Y (mm)")

    # =========================================================================
    # MLEM ITERATION LOOP
    # =========================================================================
    for it in range(1, max_iterations + 1):
        fp_full = np.zeros((skimage_sinogram_rows, num_angles))

        # Forward projection
        if use_attenuation:
            for a_idx, deg in enumerate(angles):
                X_atten = X_current * sensitivity_angles[:, :, a_idx]
                fp_single = radon(X_atten, theta=[deg], circle=False)
                fp_full[:, a_idx] = fp_single[:, 0]
        else:
            fp_full = radon(X_current, theta=angles, circle=False)

        forward_projection = fp_full[start_row:end_row, :]
        forward_projection[forward_projection == 0] = 1e-12

        # Ratio Sinogram
        ratio_small = Y / forward_projection
        ratio_full = np.ones((skimage_sinogram_rows, num_angles))
        ratio_full[start_row:end_row, :] = ratio_small

        # Backprojection
        if use_attenuation:
            back_projection_image = np.zeros((image_size, image_size))
            for a_idx, deg in enumerate(angles):
                single_ratio_sino = np.zeros((skimage_sinogram_rows, 1))
                single_ratio_sino[:, 0] = ratio_full[:, a_idx]

                bp_single = iradon(
                    single_ratio_sino,
                    theta=[deg],
                    filter_name=None,
                    circle=False,
                )

                if bp_single.shape[0] != image_size:
                    r_diff = bp_single.shape[0] - image_size
                    c_start = r_diff // 2
                    c_end = c_start + image_size
                    bp_single = bp_single[c_start:c_end, c_start:c_end]

                back_projection_image += (
                    bp_single * sensitivity_angles[:, :, a_idx]
                )
        else:
            back_projection_image = iradon(
                ratio_full, theta=angles, filter_name=None, circle=False
            )
            if back_projection_image.shape[0] != image_size:
                r_diff = back_projection_image.shape[0] - image_size
                c_start = r_diff // 2
                c_end = c_start + image_size
                back_projection_image = back_projection_image[
                    c_start:c_end, c_start:c_end
                ]
            back_projection_image /= num_angles
        # Update Step
        X_next = (X_current / sensitivity_image) * back_projection_image
        X_next = np.clip(X_next, 0, None)

        norm_current = np.linalg.norm(X_current)
        rel_change = (
            np.linalg.norm(X_next - X_current) / norm_current
            if norm_current > 0
            else 0.0
        )

        iteration_numbers.append(it)
        rel_change_history.append(rel_change)
        final_it = it

        print(
            f"Iteration {it:02d}/{max_iterations:02d} | Relative Change: {rel_change:.6e}"
        )

        X_prev_estimate = X_current.copy()
        X_current = X_next.copy()

        # Capture Frame for GIF / Live View
        if (plot_live_iterations or save_gif) and (
            it % plot_interval == 0 or it == max_iterations
        ):
            im_live.set_data(X_current)
            im_live.set_clim(vmin=np.min(X_current), vmax=np.max(X_current))
            ax_live.set_title(
                f"MLEM Iteration {it:02d} / {max_iterations:02d}\nRel Change: {rel_change:.2e}",
                fontsize=11,
                fontweight="bold",
            )
            fig_live.canvas.draw()
            fig_live.canvas.flush_events()

            if save_gif:
                # Save plot to in-memory buffer
                buf = io.BytesIO()
                fig_live.savefig(
                    buf, format="png", bbox_inches="tight", dpi=100
                )
                buf.seek(0)
                gif_frames.append(Image.open(buf).copy())
                buf.close()

            plt.pause(0.01)

        # Convergence Check
        if rel_change < tolerance_epsilon:
            print(
                f"\n[MLEM CONVERGED] Stopped at iteration {it} (ΔRel: {rel_change:.6e} < {tolerance_epsilon:.2e})"
            )
            break

    if plot_live_iterations or save_gif:
        plt.ioff()
        plt.close(fig_live)

    # Save animation to GIF file
    if save_gif and gif_frames:
        gif_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=gif_frames[1:],
            duration=frame_duration_ms,  # milliseconds per frame
            loop=0,  # 0 means infinite loop
        )
        print(f"\n[SUCCESS] GIF animation saved to: {os.path.abspath(gif_filename)}")

    # 6-Panel Diagnostic Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), dpi=130)
    im0 = axes[0, 0].imshow(
        X_prev_estimate, cmap="hot", origin="lower", extent=extent_fov
    )
    axes[0, 0].set_title(f"1. Image Estimate X({final_it - 1})")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(
        Y, cmap="plasma", aspect="auto", extent=extent_sino
    )
    axes[0, 1].set_title(f"2. Sinogram Y (Ch {start_ch}–{end_ch - 1})")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(
        forward_projection, cmap="plasma", aspect="auto", extent=extent_sino
    )
    axes[0, 2].set_title("3. Forward Projection")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].plot(iteration_numbers, rel_change_history, "o-", color="navy")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("4. Relative Change Curve")
    axes[1, 0].grid(True, which="both", linestyle=":", alpha=0.6)

    im4 = axes[1, 1].imshow(
        sensitivity_image, cmap="viridis", origin="lower", extent=extent_fov
    )
    axes[1, 1].set_title("5. Sensitivity Image")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(
        X_current, cmap="hot", origin="lower", extent=extent_fov
    )
    axes[1, 2].set_title(f"6. Final Reconstructed Image X({final_it})")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"MLEM Diagnostic Result — Final Iteration {final_it}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return X_current, rel_change_history


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    CACHE_FILE = (
        r"C:\Users\facun\Desktop\Penelope2\3_metales\spectrum_matrix_clean.npy"
    )
    SENSITIVITY_FILE = r"sensitivity_per_angle.npy"

    CHANNEL_RANGE = (399, 416)
    MAX_ITERATIONS = 100
    TOLERANCE_EPSILON = 5e-2
    USE_ATTENUATION = True

    # GIF CONFIGURATION:
    SAVE_GIF = True  # Set True to generate and export the GIF
    GIF_FILENAME = "mlem_reconstruction_process.gif"
    FRAME_DURATION_MS = 150  # Lower number = faster playback speed

    if os.path.exists(CACHE_FILE):
        spectrum_matrix_clean = np.load(CACHE_FILE)

        sensitivity_angles = None
        if USE_ATTENUATION:
            if os.path.exists(SENSITIVITY_FILE):
                sensitivity_angles = np.load(SENSITIVITY_FILE)

        final_reconstruction, change_history = mlem_iterative_debug_loop(
            spectrum_matrix_clean,
            sensitivity_angles=sensitivity_angles,
            max_iterations=MAX_ITERATIONS,
            tolerance_epsilon=TOLERANCE_EPSILON,
            channel_range=CHANNEL_RANGE,
            use_attenuation=USE_ATTENUATION,
            plot_live_iterations=True,
            plot_interval=1,
            save_gif=SAVE_GIF,
            gif_filename=GIF_FILENAME,
            frame_duration_ms=FRAME_DURATION_MS,
        )