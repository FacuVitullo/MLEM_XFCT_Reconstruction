import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu
from skimage.transform import resize

# =============================================================================
# FILE PATH CONFIGURATION & PARAMETERS
# =============================================================================
CT_SLICE_PATH = "mlem_sinogram_row_135.npz"
TARGET_SHAPE = (41, 41)


# =============================================================================
# DIRECT CT LOAD & 2D SQUEEZE
# =============================================================================
def load_ct_data(file_path):
    """Safely extracts CT array from .npz key or .npy file and handles 3D arrays."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CT file not found: {file_path}")

    if file_path.endswith(".npz"):
        data = np.load(file_path)
        possible_keys = [
            "fbp_initial",
            "reconstruction",
            "recon",
            "image",
            "data",
        ]
        selected_key = next(
            (pk for pk in possible_keys if pk in data.files), None
        )
        if selected_key is None:
            selected_key = next(
                (k for k in data.files if data[k].ndim >= 2), data.files[0]
            )
        ct_slice = data[selected_key]
        print(
            f"[LOADED CT] Key '{selected_key}' | Original Shape: {ct_slice.shape}"
        )
    else:
        ct_slice = np.load(file_path)
        print(f"[LOADED CT] From .npy | Original Shape: {ct_slice.shape}")

    ct_slice = np.squeeze(ct_slice)

    # Convert 3D array (30 x 41 x 2047) to 2D by integrating over energy channels
    if ct_slice.ndim == 3:
        print(
            f"[3D DETECTED] Collapsing spectral axis (axis 2, size {ct_slice.shape[2]}) by summation..."
        )
        ct_slice = np.sum(ct_slice, axis=2)
        print(f"[COLLAPSED] New 2D Shape: {ct_slice.shape}")

    return ct_slice


def prepare_raw_ct_slice(
    img, target_shape=(41, 41), median_radius=1, p_min=5, p_max=98
):
    """Pre-filters high-frequency noise, resizes to target grid, and normalizes values."""
    cleaned = img.astype(np.float64)

    # 1. Resize to target 2D spatial grid (e.g. 41x41)
    if cleaned.shape != target_shape:
        cleaned = resize(
            cleaned, target_shape, mode="reflect", anti_aliasing=True
        )

    # 2. Median filter
    if median_radius > 0:
        cleaned = median_filter(cleaned, size=median_radius)

    # 3. Robust percentile normalization
    vmin, vmax = np.percentile(cleaned, (p_min, p_max))
    if vmax == vmin:
        return np.zeros_like(cleaned, dtype=np.float32)

    clipped = np.clip(cleaned, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin + 1e-8)
    return normalized.astype(np.float32)


# =============================================================================
# INTERACTIVE ROI & MASK GENERATOR
# =============================================================================
class InteractiveROIMaskBuilder:

    def __init__(self, ct_img):
        self.ct_img = ct_img
        self.shape = ct_img.shape
        self.roi_mask = np.zeros(self.shape, dtype=bool)

    def onselect(self, eclick, erelease):
        """Callback to convert bounding box selection into binary mask."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Convert continuous coordinates to discrete array index bounds
        col_start = int(np.clip(np.floor(min(x1, x2)), 0, self.shape[1] - 1))
        col_end = int(np.clip(np.ceil(max(x1, x2)), 1, self.shape[1]))

        row_start = int(np.clip(np.floor(min(y1, y2)), 0, self.shape[0] - 1))
        row_end = int(np.clip(np.ceil(max(y1, y2)), 1, self.shape[0]))

        # Ensure at least 1 pixel is selected in both dimensions
        if col_start == col_end:
            col_end = min(self.shape[1], col_start + 1)
        if row_start == row_end:
            row_end = min(self.shape[0], row_start + 1)

        # Reset mask and mark selected region as True (White)
        self.roi_mask = np.zeros(self.shape, dtype=bool)
        self.roi_mask[row_start:row_end, col_start:col_end] = True

        # Redraw mask display
        self.ax_mask.clear()
        self.ax_mask.imshow(
            self.roi_mask, cmap="gray", origin="lower", vmin=0, vmax=1
        )
        self.ax_mask.set_title(
            f"Active ROI Mask\n({np.sum(self.roi_mask)} pixels active)"
        )
        self.ax_mask.set_xlabel("Pixel X")
        self.fig.canvas.draw_idle()

    def get_mask(self):
        self.fig, (self.ax_img, self.ax_mask) = plt.subplots(
            1, 2, figsize=(10, 4.5), dpi=120
        )

        self.ax_img.imshow(self.ct_img, cmap="viridis", origin="lower")
        self.ax_img.set_title(
            f"2D Collapsed CT Slice {self.shape}\n(Click & Drag Box to Select ROI)"
        )
        self.ax_img.set_xlabel("Pixel X")
        self.ax_img.set_ylabel("Pixel Y")

        self.ax_mask.imshow(
            self.roi_mask, cmap="gray", origin="lower", vmin=0, vmax=1
        )
        self.ax_mask.set_title("Active Mask")
        self.ax_mask.set_xlabel("Pixel X")

        self.rect_selector = RectangleSelector(
            self.ax_img,
            self.onselect,
            useblit=True,
            button=[1],
            interactive=True,
        )

        plt.tight_layout()
        plt.show()

        # Fallback to Otsu threshold if no ROI box was drawn
        if np.sum(self.roi_mask) == 0:
            print(
                "[INFO] No manual ROI drawn. Auto-generating Otsu threshold mask."
            )
            thresh = threshold_otsu(self.ct_img[self.ct_img > 0])
            self.roi_mask = self.ct_img > thresh

        return self.roi_mask


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    if os.path.exists(CT_SLICE_PATH):
        # 1. Load data and collapse 3D spectral dimension to 2D
        raw_ct = load_ct_data(CT_SLICE_PATH)

        # 2. Prepare 2D spatial CT (resized to 41x41, normalized)
        ct_ready = prepare_raw_ct_slice(
            raw_ct, target_shape=TARGET_SHAPE, median_radius=1
        )

        # 3. View and select ROI interactively
        builder = InteractiveROIMaskBuilder(ct_ready)
        ct_mask = builder.get_mask()

        # 4. Save mask
        np.save("ct_mask_for_mlem.npy", ct_mask)
        print(
            f"[SUCCESS] Mask saved! Final Shape: {ct_mask.shape} | Active pixels: {np.sum(ct_mask)}"
        )
    else:
        print(f"File not found: {CT_SLICE_PATH}")