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
        freqs : (n_freqs,)  Hz
    """
    H = np.fft.rfft(rir_array, n=nfft, axis=2)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return H, freqs


# ---------------------------------------------------------------------------
# Zone selection — user draws bright zone only
# ---------------------------------------------------------------------------

def choose_bright_zone(grid_size=32):
    xs, ys = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    df = pd.DataFrame({'x': xs.flatten(), 'y': ys.flatten(), 'zone': 'Unassigned'})

    app   = Dash(__name__)
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
        if ctx.triggered[0]["prop_id"].split(".")[0] != "bright-btn":
            raise PreventUpdate

        df = pd.DataFrame(zone_data)
        for pt in selected_data.get("points", []):
            mask = (df['x'] == pt['x']) & (df['y'] == pt['y'])
            df.loc[mask, 'zone'] = 'Bright'

        n = (df['zone'] == 'Bright').sum()
        return df.to_dict("records"), f"Bright Zone: {n} points selected"

    app.run(debug=False, port=8050)
    return result["bright"]


# ---------------------------------------------------------------------------
# Dark zone: spatially uniform subsample of all non-bright points
# ---------------------------------------------------------------------------

def spatially_uniform_subsample(excluded_indices, grid_size, n_samples):
    """
    Tile the grid into n_bins×n_bins cells and pick one non-bright point
    per cell, giving even spatial coverage of the dark zone.
    """
    excluded = set(excluded_indices)
    pool = np.array([i for i in range(grid_size ** 2) if i not in excluded])

    xs = pool % grid_size
    ys = pool // grid_size

    n_bins   = int(np.ceil(np.sqrt(n_samples)))
    x_splits = np.array_split(np.arange(grid_size), n_bins)
    y_splits = np.array_split(np.arange(grid_size), n_bins)

    selected = []
    for bx in x_splits:
        for by in y_splits:
            mask       = np.isin(xs, bx) & np.isin(ys, by)
            candidates = pool[mask]
            if len(candidates) > 0:
                selected.append(int(candidates[np.random.randint(len(candidates))]))
            if len(selected) >= n_samples:
                break
        if len(selected) >= n_samples:
            break

    return selected[:n_samples]


# ---------------------------------------------------------------------------
# Acoustic Contrast Control (ACC)
# ---------------------------------------------------------------------------

def acoustic_contrast_control(H_bright, H_dark, freq_idx):
    """
    ACC as per eq. (14-15) in Kristoffersen et al. (2021).

    Solves:  q = principal eigenvector of  (R_d + λ_D I)^{-1} R_b
    where    λ_D = 0.01 * ||R_d||_2   (spectral norm, i.e. largest singular value)

    This adaptive regularization scales per-frequency with the energy in the
    dark zone, avoiding the need to hand-tune a fixed regularization value.

    Args:
        H_bright : (n_bright, n_sources, n_freqs)
        H_dark   : (n_dark,   n_sources, n_freqs)
        freq_idx : frequency bin index

    Returns:
        W            : (n_sources,) optimal complex weights
        contrast_dB  : 10 log10( mean|p_b|² / mean|p_d|² )
        bright_energy: mean |p_bright|²
        dark_energy  : mean |p_dark|²
        lambda_D     : the adaptive regularization value used
    """
    from scipy.linalg import eigh

    H_b = H_bright[:, :, freq_idx]
    H_d = H_dark[:,   :, freq_idx]
    n_sources = H_b.shape[1]

    R_b = H_b.conj().T @ H_b
    R_d = H_d.conj().T @ H_d

    # Adaptive regularization: eq. (15) — 1% of spectral norm of R_d
    lambda_D = 0.01 * np.linalg.norm(R_d, ord=2)
    R_d_reg  = R_d + lambda_D * np.eye(n_sources)

    eigenvalues, eigenvectors = eigh(R_b, R_d_reg)
    W = eigenvectors[:, -1]
    W = W / (np.linalg.norm(W) + 1e-12)

    p_bright = H_b @ W
    p_dark   = H_d @ W

    bright_energy = np.mean(np.abs(p_bright) ** 2)
    dark_energy   = np.mean(np.abs(p_dark)   ** 2)
    contrast_dB   = 10 * np.log10((bright_energy + 1e-12) / (dark_energy + 1e-12))

    return W, contrast_dB, bright_energy, dark_energy, lambda_D


def get_third_octave_bands(freq_min=1500, freq_max=8000):
    """
    Returns centre frequencies of 1/3-octave bands between freq_min and freq_max.
    Standard ISO 1/3-octave centres: 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000 …
    """
    centres = []
    fc = 1000.0
    while fc <= freq_max * 1.05:
        if fc >= freq_min:
            centres.append(fc)
        fc *= 2 ** (1 / 3)
    return np.array(centres)


def acoustic_contrast_control_bands(H_bright, H_dark, freqs,
                                     freq_min=100, freq_max=8000):
    """
    1/3-octave band ACC.

    For each band: average R_b and R_d across all bins in that band,
    solve one eigenvalue problem → one W per band.

    This smooths out modal oscillation and stabilises R_b by ensuring
    enough frequency diversity to fill out the rank of the 2×2 matrices.

    Returns:
        band_centres   : (n_bands,) Hz
        band_contrasts : (n_bands,) dB — mean contrast within each band
        band_weights   : (n_bands, n_sources) complex weights per band
    """
    from scipy.linalg import eigh

    centres = get_third_octave_bands(freq_min, freq_max)
    n_sources = H_bright.shape[1]

    band_centres   = []
    band_contrasts = []
    band_weights   = []

    for fc in centres:
        f_lo = fc / 2 ** (1 / 6)
        f_hi = fc * 2 ** (1 / 6)
        band_mask = (freqs >= f_lo) & (freqs < f_hi)
        band_idx  = np.where(band_mask)[0]

        if len(band_idx) < 2:
            continue

        # Average correlation matrices across the band
        R_b = np.zeros((n_sources, n_sources), dtype=complex)
        R_d = np.zeros((n_sources, n_sources), dtype=complex)
        for fi in band_idx:
            H_b = H_bright[:, :, fi]
            H_d = H_dark[:,   :, fi]
            R_b += H_b.conj().T @ H_b
            R_d += H_d.conj().T @ H_d
        R_b /= len(band_idx)
        R_d /= len(band_idx)

        # Adaptive regularization matching eq. (15): λ_D = 0.01 * ||R_d||_2
        lambda_D = 0.01 * np.linalg.norm(R_d, ord=2)
        R_d_reg  = R_d + lambda_D * np.eye(n_sources)

        eigenvalues, eigenvectors = eigh(R_b, R_d_reg)
        W = eigenvectors[:, -1]
        W = W / (np.linalg.norm(W) + 1e-12)

        # Evaluate mean contrast across the band using this W
        contrasts_in_band = []
        for fi in band_idx:
            H_b = H_bright[:, :, fi]
            H_d = H_dark[:,   :, fi]
            p_b = H_b @ W
            p_d = H_d @ W
            c = 10 * np.log10(
                (np.mean(np.abs(p_b) ** 2) + 1e-12) /
                (np.mean(np.abs(p_d) ** 2) + 1e-12)
            )
            contrasts_in_band.append(c)

        band_centres.append(fc)
        band_contrasts.append(np.mean(contrasts_in_band))
        band_weights.append(W)

    return np.array(band_centres), np.array(band_contrasts), np.array(band_weights)




if __name__ == "__main__":

    # ── Config ───────────────────────────────────────────────────────────────
    # VR Lab room: 6.98 x 8.12 m on a 32x32 grid
    ROOM_LX         = 6.98    # metres
    ROOM_LY         = 8.12    # metres
    GRID_N          = 32
    DX              = ROOM_LX / (GRID_N - 1)   # 0.225 m
    DY              = ROOM_LY / (GRID_N - 1)   # 0.262 m

    # Source positions (metres) from the figure
    SOURCE1_POS     = (6.65, 7.93)   # Pos1 — corner
    SOURCE2_POS     = (5.23, 3.49)   # Pos2 — mid-room

    N_DARK_SAMPLES  = 20
    FREQ_MIN        = 100     # Hz — paper evaluates from ~50 Hz; use 100 to be safe
    FREQ_MAX        = 8000    # Hz
    DARK_SEED       = 42
    FS              = 48000
    NFFT            = 4096
    # ─────────────────────────────────────────────────────────────────────────

    # Load RIRs
    print("Loading RIR data …")
    rir_source1, position_list = compute_rir_array("./VRLAB-data/source_1/h_100/")
    rir_source2, _             = compute_rir_array("./VRLAB-data/source_2/h_100/")

    rir_array = np.stack([rir_source1, rir_source2], axis=1)  # (n_pts, 2, n_samples)
    n_points  = rir_array.shape[0]
    grid_size = int(round(np.sqrt(n_points)))
    print(f"Grid: {grid_size}×{grid_size}  |  RIR shape: {rir_array.shape}")

    # Transfer functions
    H, freqs = compute_transfer_function(rir_array, fs=FS, nfft=NFFT)
    print(f"Transfer functions: {H.shape}  (n_points, n_sources, n_freqs)")

    # ── Zone selection ────────────────────────────────────────────────────────
    print("\nOpening zone selection app …")
    bright_indices = choose_bright_zone(grid_size=grid_size)

    if len(bright_indices) == 0:
        print("No bright zone selected — exiting.")
        exit()

    np.random.seed(DARK_SEED)
    dark_indices = spatially_uniform_subsample(bright_indices, grid_size, N_DARK_SAMPLES)

    print(f"\nBright zone : {len(bright_indices)} points")
    print(f"Dark zone   : {len(dark_indices)} points  (uniform subsample)")

    H_B = H[bright_indices]   # (n_bright, 2, n_freqs)
    H_D = H[dark_indices]     # (n_dark,   2, n_freqs)

    # ── Frequency sweep ───────────────────────────────────────────────────────
    freq_mask    = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    freq_indices = np.where(freq_mask)[0]
    step         = max(1, len(freq_indices) // 40)
    freq_indices = freq_indices[::step]

    contrasts      = []
    bright_energies = []
    dark_energies   = []
    freq_vals      = []

    # ── Per-frequency ACC ─────────────────────────────────────────────────────
    freq_mask    = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    freq_indices = np.where(freq_mask)[0]
    step         = max(1, len(freq_indices) // 40)
    freq_indices = freq_indices[::step]

    contrasts, bright_energies, dark_energies, freq_vals = [], [], [], []

    print(f"\nRunning per-frequency ACC ({len(freq_indices)} bins) …")
    for fi in freq_indices:
        W, contrast, be, de, lam = acoustic_contrast_control(H_B, H_D, freq_idx=fi)
        contrasts.append(contrast)
        bright_energies.append(be)
        dark_energies.append(de)
        freq_vals.append(freqs[fi])

    mean_contrast = np.mean(contrasts)
    print(f"Per-frequency mean contrast : {mean_contrast:.2f} dB")

    # ── 1/3-octave band ACC ───────────────────────────────────────────────────
    print("\nRunning 1/3-octave band ACC …")
    band_centres, band_contrasts, band_weights = acoustic_contrast_control_bands(
        H_B, H_D, freqs,
        freq_min=FREQ_MIN, freq_max=FREQ_MAX
    )
    band_mean = np.mean(band_contrasts)

    print(f"\n{'Band (Hz)':>10} {'Contrast (dB)':>14}  {'Weights':>30}")
    print("-" * 58)
    for fc, bc, bw in zip(band_centres, band_contrasts, band_weights):
        print(f"{fc:>10.0f} {bc:>14.2f}  [{bw[0]:+.3f}, {bw[1]:+.3f}]")
    print(f"\n1/3-octave mean contrast : {band_mean:.2f} dB")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    fig.suptitle(
        f"ACC  |  Bright={len(bright_indices)} pts, Dark={len(dark_indices)} pts  |  "
        f"Per-freq mean={mean_contrast:.1f} dB  |  Band mean={band_mean:.1f} dB",
        fontsize=10
    )

    # Zone energies
    axes[0].semilogy(freq_vals, bright_energies, 'b-o', markersize=4, label='Bright zone')
    axes[0].semilogy(freq_vals, dark_energies,   'r-o', markersize=4, label='Dark zone')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Mean Energy  |p|²')
    axes[0].set_title('Zone Energies (per-frequency ACC)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-frequency contrast
    axes[1].plot(freq_vals, contrasts, 'g-o', markersize=4)
    axes[1].axhline(0,  color='k',      linestyle='--', alpha=0.4, label='0 dB')
    axes[1].axhline(10, color='orange', linestyle='--', alpha=0.4, label='10 dB target')
    axes[1].axhline(mean_contrast, color='green', linestyle=':', alpha=0.7,
                    label=f'Mean {mean_contrast:.1f} dB')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Contrast (dB)')
    axes[1].set_title('Per-Frequency ACC')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 1/3-octave band contrast — bar chart, one bar per band
    axes[2].bar(band_centres, band_contrasts, width=band_centres * 0.18,
                color=['#2ecc71' if c > 0 else '#e74c3c' for c in band_contrasts],
                alpha=0.8, edgecolor='white')
    axes[2].axhline(0,         color='k',      linestyle='--', alpha=0.4)
    axes[2].axhline(10,        color='orange', linestyle='--', alpha=0.4, label='10 dB target')
    axes[2].axhline(band_mean, color='purple', linestyle=':',  alpha=0.7,
                    label=f'Mean {band_mean:.1f} dB')
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Contrast (dB)')
    axes[2].set_title('1/3-Octave Band ACC  (one W per band)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('acc_results.png', dpi=150)
    plt.show()
    print("\nSaved → acc_results.png")

    # ── Spatial pressure map at best contrast frequency ───────────────────────
    best_fi      = freq_indices[np.argmax(contrasts)]
    best_freq_hz = freqs[best_fi]
    W_best, _, _, _, _ = acoustic_contrast_control(H_B, H_D, freq_idx=best_fi)

    # Compute pressure at ALL grid points using the best W
    H_all  = H[:, :, best_fi]          # (n_points, n_sources)
    p_all  = H_all @ W_best             # (n_points,) complex pressure
    p_map  = np.abs(p_all).reshape(grid_size, grid_size)

    # Convert source positions (metres) to grid indices
    src1_xi = int(round(SOURCE1_POS[0] / DX))
    src1_yi = int(round(SOURCE1_POS[1] / DY))
    src2_xi = int(round(SOURCE2_POS[0] / DX))
    src2_yi = int(round(SOURCE2_POS[1] / DY))

    # Build zone overlay mask
    zone_map = np.zeros(grid_size * grid_size)
    zone_map[bright_indices] = 1   # bright = 1
    zone_map[dark_indices]   = 2   # dark   = 2
    zone_map = zone_map.reshape(grid_size, grid_size)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle(
        f"Spatial Pressure Map  |  f = {best_freq_hz:.0f} Hz  "
        f"(best contrast = {contrasts[np.argmax(contrasts)]:.1f} dB)",
        fontsize=11
    )

    # Pressure magnitude
    im = axes2[0].imshow(p_map, origin='lower', cmap='hot',
                          extent=[0, ROOM_LX, 0, ROOM_LY], aspect='equal')
    plt.colorbar(im, ax=axes2[0], label='|p| (normalised)')

    # Overlay bright zone (blue) and dark zone (red) control points
    bright_arr = np.array(bright_indices)
    dark_arr   = np.array(dark_indices)
    bx = (bright_arr % grid_size) * DX
    by = (bright_arr // grid_size) * DY
    dx_ = (dark_arr % grid_size) * DX
    dy_ = (dark_arr // grid_size) * DY

    axes2[0].scatter(bx, by, c='cyan',   s=40, marker='o', label='Bright zone', zorder=5)
    axes2[0].scatter(dx_, dy_, c='lime', s=20, marker='x', label='Dark zone',   zorder=5)
    axes2[0].scatter(src1_xi * DX, src1_yi * DY, c='white', s=200,
                     marker='*', label='Source 1', zorder=6, edgecolors='black')
    axes2[0].scatter(src2_xi * DX, src2_yi * DY, c='yellow', s=200,
                     marker='*', label='Source 2', zorder=6, edgecolors='black')
    axes2[0].set_xlabel('x (m)')
    axes2[0].set_ylabel('y (m)')
    axes2[0].set_title('Pressure Magnitude |p|')
    axes2[0].legend(fontsize=8, loc='upper left')

    # Zone layout map
    from matplotlib.colors import ListedColormap
    cmap_zones = ListedColormap(['lightgray', '#4a90e2', '#e74c3c'])
    axes2[1].imshow(zone_map, origin='lower', cmap=cmap_zones, vmin=0, vmax=2,
                    extent=[0, ROOM_LX, 0, ROOM_LY], aspect='equal')
    axes2[1].scatter(bx, by, c='white', s=40, marker='o', zorder=5)
    axes2[1].scatter(dx_, dy_, c='white', s=20, marker='x', zorder=5)
    axes2[1].scatter(src1_xi * DX, src1_yi * DY, c='white', s=200,
                     marker='*', label='Source 1', zorder=6, edgecolors='black')
    axes2[1].scatter(src2_xi * DX, src2_yi * DY, c='yellow', s=200,
                     marker='*', label='Source 2', zorder=6, edgecolors='black')
    axes2[1].set_xlabel('x (m)')
    axes2[1].set_ylabel('y (m)')
    axes2[1].set_title('Zone Layout  (blue=bright, red=dark control pts)')
    axes2[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('acc_spatial_map.png', dpi=150)
    plt.show()
    print("Saved → acc_spatial_map.png")