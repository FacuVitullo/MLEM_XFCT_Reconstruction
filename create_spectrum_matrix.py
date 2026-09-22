import os
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from skimage.transform import iradon, radon

# =============================================================================
# Helper function for natural sorting of folder/file names
# =============================================================================
def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


# =============================================================================
# Data Configuration
# =============================================================================
ANGLES = np.arange(0, 348 + 12, 12)  # 30 Angles: [0, 12, 24, ..., 348]
N_POSITIONS = 41  # 41 Translational positions
FOV_SIZE_MM = 40.0  # 40 mm translation width


def load_spectral_data(root_folder, lam=1e6, p=0.05, max_iter=10, apply_baseline_removal=True):
    """Loads raw spectral data and returns the 3D spectral matrix.

    Parameters
    ----------
    apply_baseline_removal : bool
        Si True (default), calcula la baseline con MORALS (AsLS iterativo) y la
        resta de cada espectro, devolviendo max(0, raw - baseline) como antes.
        Si False, NO se hace ningun ajuste ni resta de baseline: se devuelve el
        espectro crudo (raw_spectrum) tal cual fue leido de cada archivo .dat.
        Util para comparar el CNR/LOD calculado sobre datos crudos vs. corregidos.
    """
    angle_folders = sorted(
        [
            f
            for f in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, f))
        ],
        key=natural_key,
    )

    if not angle_folders:
        raise FileNotFoundError(f"No subdirectories found in {root_folder}")

    first_file = None
    for folder in angle_folders:
        folder_path = os.path.join(root_folder, folder)
        dat_files = [f for f in os.listdir(folder_path) if f.endswith(".dat")]
        if dat_files:
            first_file = os.path.join(folder_path, dat_files[0])
            break

    if first_file is None:
        raise FileNotFoundError(
            f"No .dat files found in subdirectories of {root_folder}"
        )

    num_channels = len(np.loadtxt(first_file))

    D = diags(
        [1, -2, 1], [0, 1, 2], shape=(num_channels - 2, num_channels), dtype=float
    )
    DTD_scaled = lam * (D.T.dot(D))

    all_clean_data = []

    if apply_baseline_removal:
        print(
            f"--- Processing Spectral Matrix with MORALS baseline removal "
            f"(lam={lam:.1e}, p={p}, max_iter={max_iter}) ---"
        )
    else:
        print(
            "--- Processing Spectral Matrix WITHOUT baseline removal "
            "(raw spectra, no MORALS fit) ---"
        )

    for a_idx, folder in enumerate(angle_folders):
        folder_path = os.path.join(root_folder, folder)
        dat_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".dat")],
            key=natural_key,
        )

        angle_clean = []

        for file in dat_files[:N_POSITIONS]:
            file_path = os.path.join(folder_path, file)
            raw_spectrum = np.loadtxt(file_path)

            if apply_baseline_removal:
                w = np.ones(num_channels)
                baseline = np.zeros(num_channels)
                for _ in range(max_iter):
                    W = diags(w, 0, dtype=float)
                    A_mat = W + DTD_scaled
                    baseline = spsolve(A_mat, w * raw_spectrum)
                    w = np.where(raw_spectrum > baseline, p, 1.0 - p)

                clean_spectrum = np.maximum(0.0, raw_spectrum - baseline)
            else:
                # Sin remocion de baseline: se conserva el espectro crudo tal cual.
                clean_spectrum = raw_spectrum

            angle_clean.append(clean_spectrum)

        all_clean_data.append(angle_clean)

        print(f" Loaded {folder} ({a_idx + 1}/{len(angle_folders)})")

    spectrum_matrix_clean = np.array(all_clean_data, dtype=np.float32)

    return spectrum_matrix_clean


# =============================================================================
# Main Execution & Caching Strategy
# =============================================================================
BASE_DIR = r"C:\Users\facun\Desktop\Penelope2\3_metales\CorrectedData"

# Cambiar a False para generar la matriz SIN remocion de baseline (espectros
# crudos), y asi poder comparar el CNR/LOD calculado con ambos metodos.
APPLY_BASELINE_REMOVAL = False

# Cada modo escribe/lee su propio archivo de cache, para no pisar la matriz
# baseline-corrected con la version cruda (o viceversa).
CACHE_FILE = (
    r"C:\Users\facun\Desktop\Penelope2\3_metales\spectrum_matrix_clean.npy"
    if APPLY_BASELINE_REMOVAL
    else r"C:\Users\facun\Desktop\Penelope2\3_metales\spectrum_matrix_raw.npy"
)

MORALS_LAMBDA = 1e5  # Smoothness penalty
MORALS_P = 0.200  # Asymmetry weight
MORALS_ITER = 20  # Max iterations for baseline convergence

if os.path.exists(CACHE_FILE):
    print(f"--- Loading cached matrix from: {CACHE_FILE} ---")
    spectrum_matrix_clean = np.load(CACHE_FILE)
    print(f"Successfully loaded matrix with shape: {spectrum_matrix_clean.shape}")

elif os.path.exists(BASE_DIR):
    spectrum_matrix_clean = load_spectral_data(
        BASE_DIR,
        lam=MORALS_LAMBDA,
        p=MORALS_P,
        max_iter=MORALS_ITER,
        apply_baseline_removal=APPLY_BASELINE_REMOVAL,
    )
    # Save processed numpy array to disk
    np.save(CACHE_FILE, spectrum_matrix_clean)
    print(f"--- Saved matrix to: {CACHE_FILE} "
          f"({'baseline-corrected' if APPLY_BASELINE_REMOVAL else 'raw, sin baseline removal'}) ---")

else:
    raise FileNotFoundError(f"Neither {CACHE_FILE} nor {BASE_DIR} was found.")