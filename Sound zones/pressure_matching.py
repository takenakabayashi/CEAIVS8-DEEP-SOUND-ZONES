import numpy as np
import scipy.io
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def parse_position(filename):
    match = re.search(r'idxX_(\d+)_idxY_(\d+)', filename)
    return int(match.group(1)), int(match.group(2))


def compute_rir_array(folder):
    rir_list, position_list = [], []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".mat"):
            idxX, idxY = parse_position(file)
            position_list.append((idxX, idxY))
            data = scipy.io.loadmat(os.path.join(folder, file))
            rir_list.append(data['ImpulseResponse'].flatten())

    sorted_pairs = sorted(zip(position_list, rir_list), key=lambda x: (x[0][1], x[0][0]))
    position_list = [p[0] for p in sorted_pairs]
    rir_list      = [p[1] for p in sorted_pairs]
    return np.array(rir_list), position_list


def compute_transfer_function(rir_array, fs=48000, nfft=4096):
    """
    Returns:
        H     : (n_points, n_sources, n_freqs)  complex transfer functions
        freqs : (n_freqs,)  frequency axis in Hz
    """
    H = np.fft.rfft(rir_array, n=nfft, axis=2)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return H, freqs


# ---------------------------------------------------------------------------
# Zone selection (Dash app) — user draws bright zone only
# ---------------------------------------------------------------------------

def choose_bright_zone(grid_size=32):
    """
    Interactive Dash app.  User lasso-selects the bright zone; the dark zone
    is derived automatically afterwards via spatially_uniform_subsample().
    Returns bright_indices (list of flat grid indices).
    """
    xs, ys = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    df = pd.DataFrame({'x': xs.flatten(), 'y': ys.flatten(), 'zone': 'Unassigned'})

    app = Dash(__name__)
    result = {"bright": []}

    app.layout = html.Div([
        dcc.Graph(id="grid"),
        html.Button("Assign as Bright Zone", id="bright-btn", n_clicks=0,
                    style={"backgroundColor": "#4a90e2", "color": "white",
                           "padding": "8px 16px", "border": "none",
                           "borderRadius": "4px", "cursor": "pointer"}),
        html.Div(id="status", style={"marginTop": "10px", "fontWeight": "bold"}),
        html.Div("Lasso-select points, then click the button. "
                 "Close the browser tab when done.",
                 style={"color": "gray", "marginTop": "8px"}),
        dcc.Store(id="zone-store", data=df.to_dict("records")),
    ])

    @app.callback(
        Output("grid", "figure"),
        Input("zone-store", "data"),
        prevent_initial_call=False
    )
    def update_figure(zone_data):
        df_disp = pd.DataFrame(zone_data)
        result["bright"] = df_disp[df_disp['zone'] == 'Bright'].index.tolist()
        fig = px.scatter(
            df_disp, x='x', y='y', color='zone',
            title=f"Select Bright Zone  ({grid_size}×{grid_size} grid)",
            color_discrete_map={"Unassigned": "lightgray", "Bright": "#4a90e2"}
        )
        fig.update_layout(dragmode="lasso")
        fig.update_traces(marker=dict(size=7))
        return fig

    @app.callback(
        Output("zone-store", "data"),
        Output("status", "children"),
        Input("bright-btn", "n_clicks"),
        Input("grid", "selectedData"),
        State("zone-store", "data"),
        prevent_initial_call=True
    )
    def handle_selection(bright_clicks, selected_data, zone_data):
        ctx = callback_context
        if not ctx.triggered or selected_data is None:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id != "bright-btn":
            raise PreventUpdate

        df = pd.DataFrame(zone_data)
        for pt in selected_data.get("points", []):
            mask = (df['x'] == pt['x']) & (df['y'] == pt['y'])
            df.loc[mask, 'zone'] = 'Bright'

        bright_count = (df['zone'] == 'Bright').sum()
        return df.to_dict("records"), f"Bright Zone: {bright_count} points selected"

    app.run(debug=False, port=8050)
    return result["bright"]


# ---------------------------------------------------------------------------
# Dark zone: spatially uniform subsample of all non-bright points
# ---------------------------------------------------------------------------

def spatially_uniform_subsample(excluded_indices, grid_size, n_samples):
    """
    Divide the grid into a coarse n_bins × n_bins tiling and pick one
    non-bright point from each tile.  This ensures the dark zone control
    points are spread evenly across the room rather than clustered.

    Args:
        excluded_indices : flat indices that belong to the bright zone
        grid_size        : side length of the square grid
        n_samples        : desired number of dark control points

    Returns:
        selected : list of flat indices (length <= n_samples)
    """
    excluded = set(excluded_indices)
    all_indices = np.arange(grid_size ** 2)
    pool = all_indices[~np.isin(all_indices, list(excluded))]

    xs = pool % grid_size
    ys = pool // grid_size

    n_bins = int(np.ceil(np.sqrt(n_samples)))
    x_splits = np.array_split(np.arange(grid_size), n_bins)
    y_splits = np.array_split(np.arange(grid_size), n_bins)

    selected = []
    for bx in x_splits:
        for by in y_splits:
            mask = np.isin(xs, bx) & np.isin(ys, by)
            candidates = pool[mask]
            if len(candidates) > 0:
                chosen = candidates[np.random.randint(len(candidates))]
                selected.append(int(chosen))
            if len(selected) >= n_samples:
                break
        if len(selected) >= n_samples:
            break

    return selected[:n_samples]


# ---------------------------------------------------------------------------
# Plane wave target
# ---------------------------------------------------------------------------

def create_plane_wave_target(bright_indices, grid_size, grid_spacing_m,
                              angle_deg=0, freq_hz=1000, c=343.0):
    """
    Physically correct plane wave: exp(i k · r) evaluated at each bright point.

    Args:
        bright_indices  : flat grid indices of bright zone
        grid_size       : side length of grid
        grid_spacing_m  : metres between adjacent grid points
        angle_deg       : arrival direction (0 = +x axis)
        freq_hz         : frequency in Hz (must match solve frequency)
        c               : speed of sound m/s

    Returns:
        target : complex (n_bright,), unit norm
    """
    k = 2 * np.pi * freq_hz / c
    angle_rad = np.deg2rad(angle_deg)
    kx = k * np.cos(angle_rad)
    ky = k * np.sin(angle_rad)

    idx = np.array(bright_indices)
    x = (idx % grid_size) * grid_spacing_m
    y = (idx // grid_size) * grid_spacing_m

    target = np.exp(1j * (kx * x + ky * y))
    return target / (np.linalg.norm(target) + 1e-12)


# ---------------------------------------------------------------------------
# Pressure matching solver
# ---------------------------------------------------------------------------

def pressure_matching(H_bright, H_dark,
                      bright_indices, grid_size, grid_spacing_m,
                      freq_idx, freqs,
                      angle_deg=0, c=343.0,
                      lambda_bright=1.0, lambda_dark=1.0,
                      regularization=1e-2):
    """
    Weighted least-squares pressure matching with Tikhonov regularisation.

    min  λ_d ‖H_d W‖² + λ_b ‖H_b W − t‖² + μ ‖W‖²

    Args:
        H_bright / H_dark : (n_points, n_sources, n_freqs)
        bright_indices    : flat grid indices for bright zone (for target phase)
        freq_idx          : which frequency bin to solve
        lambda_dark       : weight on dark zone suppression  (increase to suppress more)
        lambda_bright     : weight on bright zone tracking   (keep at 1.0, tune lambda_dark)
        regularization    : Tikhonov μ

    Returns:
        W          : complex source weights (n_sources,)
        error_bright : relative bright zone error (lower = better plane wave match)
        error_dark   : absolute dark zone pressure norm (lower = better suppression)
        contrast_dB  : 10 log10( mean|p_b|² / mean|p_d|² )  (higher = better)
    """
    H_b = H_bright[:, :, freq_idx]   # (n_bright, n_sources)
    H_d = H_dark[:,   :, freq_idx]   # (n_dark,   n_sources)
    n_sources = H_b.shape[1]

    freq_hz = freqs[freq_idx]
    target  = create_plane_wave_target(bright_indices, grid_size, grid_spacing_m,
                                        angle_deg=angle_deg, freq_hz=freq_hz, c=c)

    # Build stacked system  A W = b
    A = np.vstack([
        np.sqrt(lambda_dark)    * H_d,
        np.sqrt(lambda_bright)  * H_b,
        np.sqrt(regularization) * np.eye(n_sources, dtype=complex)
    ])
    b = np.hstack([
        np.zeros(H_d.shape[0], dtype=complex),
        np.sqrt(lambda_bright)  * target,
        np.zeros(n_sources,      dtype=complex)
    ])

    W, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    p_bright = H_b @ W
    p_dark   = H_d @ W

    error_bright = (np.linalg.norm(p_bright - target) /
                    (np.linalg.norm(target) + 1e-12))
    error_dark   = np.linalg.norm(p_dark)
    contrast_dB  = 10 * np.log10(
        (np.mean(np.abs(p_bright) ** 2) + 1e-12) /
        (np.mean(np.abs(p_dark)   ** 2) + 1e-12)
    )
    return W, error_bright, error_dark, contrast_dB


# ---------------------------------------------------------------------------
# Lambda sweep (calibration helper)
# ---------------------------------------------------------------------------

def calibrate_lambda(H_B, H_D, bright_indices, grid_size, grid_spacing_m,
                     freqs, cal_freq_hz=1000, c=343.0, angle_deg=0):
    """
    Grid-search over lambda_dark × regularization at a single calibration
    frequency and return the combination with highest acoustic contrast.
    """
    cal_idx = int(np.argmin(np.abs(freqs - cal_freq_hz)))

    lambda_dark_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    reg_vals         = [1e-3, 1e-2, 1e-1]

    best_contrast = -np.inf
    best_params   = {"lambda_dark": 1.0, "regularization": 1e-2}

    print(f"\nCalibrating at {freqs[cal_idx]:.1f} Hz …")
    print(f"{'lambda_dark':>12} {'reg':>8} {'contrast_dB':>12} {'err_bright':>12}")
    print("-" * 50)

    for ld in lambda_dark_vals:
        for reg in reg_vals:
            W, eb, ed, contrast = pressure_matching(
                H_B, H_D, bright_indices, grid_size, grid_spacing_m,
                freq_idx=cal_idx, freqs=freqs,
                angle_deg=angle_deg, c=c,
                lambda_bright=1.0, lambda_dark=ld, regularization=reg
            )
            print(f"{ld:>12.2f} {reg:>8.0e} {contrast:>12.2f} {eb:>12.4f}")
            if contrast > best_contrast:
                best_contrast = contrast
                best_params = {"lambda_dark": ld, "regularization": reg}

    print(f"\n→ Best: lambda_dark={best_params['lambda_dark']}, "
          f"reg={best_params['regularization']}, contrast={best_contrast:.2f} dB")
    return best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Config ──────────────────────────────────────────────────────────────
    GRID_SPACING_M  = 0.05      # metres between adjacent mic positions — ADJUST THIS
    C               = 343.0     # speed of sound m/s
    ANGLE_DEG       = 0         # plane wave arrival direction
    N_DARK_SAMPLES  = 20        # dark zone control points (keep small: 4–10× n_sources)
    FS              = 48000
    NFFT            = 4096
    # ────────────────────────────────────────────────────────────────────────

    # Load RIRs
    print("Loading RIR data …")
    rir_source1, position_list = compute_rir_array("./VRLAB-data/source_1/h_100/")
    rir_source2, _             = compute_rir_array("./VRLAB-data/source_2/h_100/")

    rir_array = np.stack([rir_source1, rir_source2], axis=1)   # (n_pts, 2, n_samples)
    n_points  = rir_array.shape[0]
    grid_size = int(round(np.sqrt(n_points)))
    print(f"Grid: {grid_size}×{grid_size}  |  RIR shape: {rir_array.shape}")

    # Transfer functions
    H, freqs = compute_transfer_function(rir_array, fs=FS, nfft=NFFT)
    print(f"Transfer function shape: {H.shape}  (n_points, n_sources, n_freqs)")

    # ── Zone selection ───────────────────────────────────────────────────────
    print("\nOpening zone selection app …")
    bright_indices = choose_bright_zone(grid_size=grid_size)

    if len(bright_indices) == 0:
        print("No bright zone selected — exiting.")
        exit()

    # Auto-generate spatially uniform dark zone from all non-bright points
    dark_indices = spatially_uniform_subsample(bright_indices, grid_size, N_DARK_SAMPLES)

    print(f"\nBright zone : {len(bright_indices)} points")
    print(f"Dark zone   : {len(dark_indices)} points  (uniform subsample)")

    H_B = H[bright_indices]   # (n_bright, 2, n_freqs)
    H_D = H[dark_indices]     # (n_dark,   2, n_freqs)

    # ── Calibrate lambda at 1 kHz ────────────────────────────────────────────
    best_params = calibrate_lambda(
        H_B, H_D, bright_indices, grid_size, GRID_SPACING_M,
        freqs, cal_freq_hz=1000, c=C, angle_deg=ANGLE_DEG
    )
    lambda_dark   = best_params["lambda_dark"]
    regularization = best_params["regularization"]

    # ── Frequency sweep ──────────────────────────────────────────────────────
    freq_mask    = (freqs >= 100) & (freqs <= 8000)
    freq_indices = np.where(freq_mask)[0]
    step         = max(1, len(freq_indices) // 30)
    freq_indices = freq_indices[::step]

    errors_bright, errors_dark, contrasts, freq_vals = [], [], [], []

    print(f"\nRunning pressure matching across {len(freq_indices)} frequencies …")
    print(f"{'Freq (Hz)':>10} {'Contrast (dB)':>14} {'Bright Err':>12} {'Dark Energy':>12}")
    print("-" * 52)

    for fi in freq_indices:
        W, eb, ed, contrast = pressure_matching(
            H_B, H_D, bright_indices, grid_size, GRID_SPACING_M,
            freq_idx=fi, freqs=freqs,
            angle_deg=ANGLE_DEG, c=C,
            lambda_bright=1.0,
            lambda_dark=lambda_dark,
            regularization=regularization
        )
        errors_bright.append(eb)
        errors_dark.append(ed)
        contrasts.append(contrast)
        freq_vals.append(freqs[fi])
        print(f"{freqs[fi]:>10.1f} {contrast:>14.2f} {eb:>12.4f} {ed:>12.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Pressure Matching  |  λ_dark={lambda_dark}, μ={regularization}  "
                 f"|  Bright={len(bright_indices)} pts, Dark={len(dark_indices)} pts",
                 fontsize=11)

    axes[0].semilogy(freq_vals, errors_bright, 'b-o', markersize=5)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Relative Error')
    axes[0].set_title('Bright Zone: Plane Wave Error')
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(freq_vals, errors_dark, 'r-o', markersize=5)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Pressure (normalised)')
    axes[1].set_title('Dark Zone: Energy')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq_vals, contrasts, 'g-o', markersize=5)
    axes[2].axhline(0,  color='k',      linestyle='--', alpha=0.4, label='0 dB')
    axes[2].axhline(10, color='orange', linestyle='--', alpha=0.4, label='10 dB target')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Contrast (dB)')
    axes[2].set_title('Acoustic Contrast  (higher = better)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pressure_matching_results_v3.png', dpi=150)
    plt.show()
    print("\nSaved → pressure_matching_results_v3.png")