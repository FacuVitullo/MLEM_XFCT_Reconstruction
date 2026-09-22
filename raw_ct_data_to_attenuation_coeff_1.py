import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector


def load_and_inspect_ct(file_path="aligned_ct_slice_for_xfct.npy"):
    # 1. Load the CT slice
    try:
        ct_slice = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Check the file path.")
        return

    print("=== CT Slice Information ===")
    print(f"Shape: {ct_slice.shape}")
    print(f"Data type: {ct_slice.dtype}")
    print(f"Global Min Value: {ct_slice.min():.4f}")
    print(f"Global Max Value: {ct_slice.max():.4f}")
    print(f"Global Mean Value: {ct_slice.mean():.4f}")
    print("============================\n")

    # 2. Setup figure
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(ct_slice, cmap="gray")
    fig.colorbar(im, ax=ax, label="CT Value / Density")
    ax.set_title(
        "CT Slice - Click & Drag to Select ROI\n(Values shown in terminal)"
    )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    # 3. Callback function for ROI selection
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Coordinate bounding
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        roi = ct_slice[ymin : ymax + 1, xmin : xmax + 1]

        print(f"--- ROI Selected [Y: {ymin}:{ymax}, X: {xmin}:{xmax}] ---")
        print(f"  Pixel Count : {roi.size}")
        print(f"  Mean Value  : {roi.mean():.6f}")
        print(f"  Std Dev     : {roi.std():.6f}")
        print(f"  Min Value   : {roi.min():.6f}")
        print(f"  Max Value   : {roi.max():.6f}")
        print("-" * 50)

    # 4. Attach interactive rectangle selector (compatible across all Matplotlib versions)
    toggle_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],  # Left mouse button
        interactive=True,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_and_inspect_ct()