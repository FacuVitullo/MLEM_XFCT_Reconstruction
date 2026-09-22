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
# CNR / LOD ESTIMATION UTILITIES
# =============================================================================
def create_circular_roi_mask(image_size, extent_fov, center_mm, diameter_mm):
    """
    Genera una mascara booleana circular sobre la grilla de la imagen reconstruida.

    Parameters
    ----------
    image_size : int
        Numero de pixeles por lado de la imagen (se asume cuadrada, como X_current).
    extent_fov : list o tuple
        [xmin, xmax, ymin, ymax] en mm, el mismo 'extent' usado en imshow
        (en este script: extent_fov = [-FOV_SIZE_MM/2, FOV_SIZE_MM/2, ...]).
    center_mm : tuple (x_c, y_c)
        Coordenadas del centro del ROI en mm, en el mismo sistema que extent_fov.
    diameter_mm : float
        Diametro del ROI circular en mm (para un ROI de 3.9 mm, diameter_mm=3.9).

    Returns
    -------
    mask : ndarray de bool, shape (image_size, image_size)
        True dentro del circulo, False afuera.
    """
    xmin, xmax, ymin, ymax = extent_fov
    radius_mm = diameter_mm / 2.0

    # Coordenadas fisicas (mm) del centro de cada pixel, consistentes con
    # origin="lower" y el extent usado en las llamadas a imshow del script.
    px_w = (xmax - xmin) / image_size
    px_h = (ymax - ymin) / image_size
    x_coords = xmin + px_w * (np.arange(image_size) + 0.5)
    y_coords = ymin + px_h * (np.arange(image_size) + 0.5)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

    xc, yc = center_mm
    mask = (X_grid - xc) ** 2 + (Y_grid - yc) ** 2 <= radius_mm ** 2

    if not mask.any():
        raise ValueError(
            f"El ROI centrado en {center_mm} mm con diametro {diameter_mm} mm "
            "no contiene ningun pixel dentro de la imagen. Revisar coordenadas/extent."
        )
    return mask


def compute_cnr(image, roi_signal_center_mm, roi_bg_center_mm, roi_diameter_mm, extent_fov):
    """
    Calcula el Contrast-to-Noise Ratio (CNR) entre un ROI de senal y un ROI de fondo,
    ambos circulares de igual diametro, definido como:

        CNR = |mean_signal - mean_bg| / std_bg

    Parameters
    ----------
    image : ndarray, shape (N, N)
        Imagen reconstruida (p.ej. X_current devuelto por mlem_iterative_debug_loop).
    roi_signal_center_mm : tuple (x, y)
        Centro del ROI colocado sobre la region de interes (senal), en mm.
    roi_bg_center_mm : tuple (x, y)
        Centro del ROI colocado sobre una region de solo fondo, en mm.
    roi_diameter_mm : float
        Diametro comun de ambos ROIs circulares, en mm (3.9 mm en este caso).
    extent_fov : list
        [xmin, xmax, ymin, ymax] en mm, igual al extent usado para graficar `image`.

    Returns
    -------
    cnr : float
    stats : dict
        Contiene mean/std/n_pixels de cada ROI y el propio cnr.
    mask_signal, mask_bg : ndarray de bool
        Mascaras usadas, utiles para graficar los ROIs superpuestos.
    """
    image_size = image.shape[0]
    mask_signal = create_circular_roi_mask(image_size, extent_fov, roi_signal_center_mm, roi_diameter_mm)
    mask_bg = create_circular_roi_mask(image_size, extent_fov, roi_bg_center_mm, roi_diameter_mm)

    signal_vals = image[mask_signal]
    bg_vals = image[mask_bg]

    mean_signal = float(signal_vals.mean())
    mean_bg = float(bg_vals.mean())
    std_signal = float(signal_vals.std(ddof=1)) if signal_vals.size > 1 else 0.0
    std_bg = float(bg_vals.std(ddof=1)) if bg_vals.size > 1 else 0.0

    n_signal = int(signal_vals.size)
    n_bg = int(bg_vals.size)
    diff = mean_signal - mean_bg

    cnr = abs(diff) / std_bg if std_bg > 0 else np.inf

    # --- Propagacion de errores (mismo metodo que en p2.py) ---
    # Error estandar de la diferencia de medias (senal - fondo):
    if n_signal > 1 and n_bg > 1:
        sigma_diff = np.sqrt(std_signal**2 / n_signal + std_bg**2 / n_bg)
    else:
        sigma_diff = np.nan

    # Error relativo de un desvio estandar muestral estimado con n_bg muestras
    # (asume ruido de fondo aproximadamente gaussiano):
    if n_bg > 1:
        rel_sigma_bgstd = 1.0 / np.sqrt(2.0 * (n_bg - 1))
    else:
        rel_sigma_bgstd = np.nan

    # Combinacion en cuadratura -> error relativo y absoluto del CNR:
    if std_bg > 0 and diff != 0 and np.isfinite(sigma_diff) and np.isfinite(rel_sigma_bgstd):
        rel_sigma_cnr = np.sqrt((sigma_diff / abs(diff)) ** 2 + rel_sigma_bgstd**2)
        sigma_cnr = cnr * rel_sigma_cnr
    else:
        sigma_cnr = np.nan

    stats = {
        "mean_signal": mean_signal,
        "mean_bg": mean_bg,
        "std_signal": std_signal,
        "std_bg": std_bg,
        "n_pixels_signal": n_signal,
        "n_pixels_bg": n_bg,
        "cnr": cnr,
        "sigma_cnr": sigma_cnr,
    }
    return cnr, stats, mask_signal, mask_bg


def estimate_lod_from_cnr(mean_signal, mean_bg, std_bg, k=3.0):
    """
    Estima el Limit of Detection (LOD) a partir de las estadisticas del ROI de fondo,
    usando el criterio clasico de deteccion a k desviaciones estandar (k=3 por defecto,
    equivalente a ~99.7% de confianza bajo hipotesis de ruido gaussiano).

    Asume una relacion lineal senal-concentracion: senal = m * concentracion + mean_bg,
    por lo que se requiere la pendiente `m` [unidades de senal / unidad de concentracion]
    (p.ej. obtenida por calibracion con muestras de concentracion conocida) para convertir
    el umbral de senal detectable en una concentracion minima detectable.

    Parameters
    ----------
    mean_signal, mean_bg, std_bg : float
        Salidas de compute_cnr(). Aqui solo se usa std_bg y mean_bg.
    k : float
        Factor de deteccion (tipicamente 3).

    Returns
    -------
    signal_threshold : float
        Nivel de senal minimo detectable por encima del fondo: mean_bg + k * std_bg.
    """
    return mean_bg + k * std_bg


def plot_cnr_rois(
    image,
    extent_fov,
    roi_signal_center_mm,
    roi_bg_center_mm,
    roi_diameter_mm,
    stats,
    title="CNR ROIs sobre la reconstruccion final",
):
    """
    Grafica la imagen reconstruida con los dos ROIs circulares (senal y fondo)
    dibujados encima, y anota el CNR resultante.
    """
    from matplotlib.patches import Circle

    radius_mm = roi_diameter_mm / 2.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
    im = ax.imshow(image, cmap="hot", origin="lower", extent=extent_fov)
    fig.colorbar(im, ax=ax, label="Counts")

    circ_signal = Circle(
        roi_signal_center_mm, radius_mm, edgecolor="cyan", facecolor="none", lw=2, label="ROI senal"
    )
    circ_bg = Circle(
        roi_bg_center_mm, radius_mm, edgecolor="lime", facecolor="none", lw=2, label="ROI fondo"
    )
    ax.add_patch(circ_signal)
    ax.add_patch(circ_bg)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(
        f"{title}\nCNR = {stats['cnr']:.3f} ± {stats['sigma_cnr']:.3f}  "
        f"(mean_sig={stats['mean_signal']:.3g}, mean_bg={stats['mean_bg']:.3g}, std_bg={stats['std_bg']:.3g})",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


class DraggableCNRROIs:
    """
    Permite posicionar los dos ROIs circulares (senal y fondo) arrastrandolos
    con el mouse sobre la imagen reconstruida, con el CNR recalculado y
    mostrado en vivo en el titulo mientras se arrastra.

    Uso:
        selector = DraggableCNRROIs(final_reconstruction, extent_fov, ROI_DIAMETER_MM)
        plt.show()  # arrastrar los circulos; cerrar la ventana para continuar
        signal_center, bg_center = selector.get_centers()
        cnr, stats, *_ = compute_cnr(final_reconstruction, signal_center, bg_center,
                                      ROI_DIAMETER_MM, extent_fov)

    Requiere un backend interactivo de matplotlib (TkAgg, QtAgg, etc.); no
    funciona con backends no interactivos (Agg) ni en algunos notebooks sin
    "%matplotlib widget"/"%matplotlib qt".
    """

    def __init__(
        self,
        image,
        extent_fov,
        roi_diameter_mm,
        init_signal_center_mm=(0.0, 0.0),
        init_bg_center_mm=None,
    ):
        from matplotlib.patches import Circle

        self.image = image
        self.extent_fov = extent_fov
        self.roi_diameter_mm = roi_diameter_mm
        self.radius_mm = roi_diameter_mm / 2.0
        self.dragging = None  # "signal", "bg", o None

        if init_bg_center_mm is None:
            xmin, xmax, ymin, ymax = extent_fov
            # Por defecto, arranca en una esquina del FOV, lejos del centro.
            init_bg_center_mm = (xmax * 0.6, ymax * 0.6)

        self.fig, self.ax = plt.subplots(figsize=(6.5, 5.5), dpi=110)
        self.im = self.ax.imshow(image, cmap="hot", origin="lower", extent=extent_fov)
        self.fig.colorbar(self.im, ax=self.ax, label="Counts")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")

        self.circ_signal = Circle(
            init_signal_center_mm, self.radius_mm,
            edgecolor="cyan", facecolor="none", lw=2, picker=True,
        )
        self.circ_bg = Circle(
            init_bg_center_mm, self.radius_mm,
            edgecolor="lime", facecolor="none", lw=2, picker=True,
        )
        self.ax.add_patch(self.circ_signal)
        self.ax.add_patch(self.circ_bg)
        self.ax.text(*init_signal_center_mm, "S", color="cyan", ha="center", va="center", fontweight="bold")
        self._label_signal = self.ax.texts[-1]
        self.ax.text(*init_bg_center_mm, "B", color="lime", ha="center", va="center", fontweight="bold")
        self._label_bg = self.ax.texts[-1]

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self._update_title()
        self.fig.suptitle(
            "Arrastrar los circulos (cian=senal, verde=fondo). Cerrar ventana al terminar.",
            fontsize=9,
        )
        plt.tight_layout()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        # Prioriza el circulo mas cercano al click si ambos se solapan.
        d_signal = np.hypot(event.xdata - self.circ_signal.center[0], event.ydata - self.circ_signal.center[1])
        d_bg = np.hypot(event.xdata - self.circ_bg.center[0], event.ydata - self.circ_bg.center[1])
        contains_signal, _ = self.circ_signal.contains(event)
        contains_bg, _ = self.circ_bg.contains(event)
        if contains_signal and (not contains_bg or d_signal <= d_bg):
            self.dragging = "signal"
        elif contains_bg:
            self.dragging = "bg"

    def _on_release(self, event):
        self.dragging = None

    def _on_motion(self, event):
        if self.dragging is None or event.inaxes != self.ax or event.xdata is None:
            return
        new_center = (event.xdata, event.ydata)
        if self.dragging == "signal":
            self.circ_signal.center = new_center
            self._label_signal.set_position(new_center)
        else:
            self.circ_bg.center = new_center
            self._label_bg.set_position(new_center)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        try:
            cnr, stats, _, _ = compute_cnr(
                self.image, self.circ_signal.center, self.circ_bg.center,
                self.roi_diameter_mm, self.extent_fov,
            )
            self.ax.set_title(
                f"CNR = {cnr:.3f} ± {stats['sigma_cnr']:.3f}  |  "
                f"senal=({self.circ_signal.center[0]:.1f}, {self.circ_signal.center[1]:.1f}) mm  "
                f"fondo=({self.circ_bg.center[0]:.1f}, {self.circ_bg.center[1]:.1f}) mm",
                fontsize=9,
            )
        except ValueError:
            self.ax.set_title("Uno de los ROIs quedo fuera de la imagen", fontsize=9, color="red")

    def get_centers(self):
        """Devuelve (signal_center_mm, bg_center_mm) con la posicion final de cada ROI."""
        return self.circ_signal.center, self.circ_bg.center


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
    """
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
    """

    return X_current, rel_change_history


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    CACHE_FILE = (
        r"C:\Users\facun\Desktop\Penelope2\3_metales\spectrum_matrix_clean.npy"
    )
    SENSITIVITY_FILE = r"sensitivity_per_angle.npy"

    #plata ---> 200  221
    #Gadolinio ----> 393 416
    #Bismuto -----> 116 136

    CHANNEL_RANGE = (116, 136)
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

        # =====================================================================
        # CNR / LOD ESTIMATION
        # =====================================================================
        # ROI_DIAMETER_MM se asume como DIAMETRO (no radio) del circulo de 3.9 mm.
        # Si en realidad 3.9 mm es el RADIO, usar ROI_DIAMETER_MM = 2 * 3.9.
        ROI_DIAMETER_MM = 6

        extent_fov = [
            -FOV_SIZE_MM / 2,
            FOV_SIZE_MM / 2,
            -FOV_SIZE_MM / 2,
            FOV_SIZE_MM / 2,
        ]

        # Posicionamiento de los ROIs: True = arrastrar con el mouse (recomendado
        # para uso interactivo); False = usar coordenadas fijas mas abajo.
        USE_INTERACTIVE_ROI_SELECTION = True

        if USE_INTERACTIVE_ROI_SELECTION:
            roi_selector = DraggableCNRROIs(
                final_reconstruction, extent_fov, ROI_DIAMETER_MM
            )
            plt.show()  # bloquea hasta que se cierra la ventana
            ROI_SIGNAL_CENTER_MM, ROI_BG_CENTER_MM = roi_selector.get_centers()
            print(f"\n[ROI] Centro senal final: {ROI_SIGNAL_CENTER_MM} mm")
            print(f"[ROI] Centro fondo final:  {ROI_BG_CENTER_MM} mm")
        else:
            # Coordenadas fijas (mm), ajustar a la geometria real del fantoma/FOV.
            ROI_SIGNAL_CENTER_MM = (0.0, 0.0)   # centro sobre la region de interes
            ROI_BG_CENTER_MM = (10.0, 10.0)     # centro sobre una region de solo fondo

        cnr, cnr_stats, mask_signal, mask_bg = compute_cnr(
            final_reconstruction,
            roi_signal_center_mm=ROI_SIGNAL_CENTER_MM,
            roi_bg_center_mm=ROI_BG_CENTER_MM,
            roi_diameter_mm=ROI_DIAMETER_MM,
            extent_fov=extent_fov,
        )

        signal_threshold = estimate_lod_from_cnr(
            cnr_stats["mean_signal"], cnr_stats["mean_bg"], cnr_stats["std_bg"], k=3.0
        )

        print(f"\n[CNR] mean_signal={cnr_stats['mean_signal']:.4g} | "
              f"mean_bg={cnr_stats['mean_bg']:.4g} | std_bg={cnr_stats['std_bg']:.4g}")
        print(f"[CNR] N pixeles: senal={cnr_stats['n_pixels_signal']}, fondo={cnr_stats['n_pixels_bg']}")
        print(f"[CNR] CNR = {cnr:.4f} ± {cnr_stats['sigma_cnr']:.4f}")
        print(f"[LOD] Umbral de senal detectable (k=3): {signal_threshold:.4g} "
              "(unidades de X_current; convertir a concentracion con una curva de calibracion)")

        plot_cnr_rois(
            final_reconstruction,
            extent_fov,
            ROI_SIGNAL_CENTER_MM,
            ROI_BG_CENTER_MM,
            ROI_DIAMETER_MM,
            cnr_stats,
        )