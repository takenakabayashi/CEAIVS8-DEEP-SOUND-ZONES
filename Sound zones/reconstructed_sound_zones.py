"""
Sparse Sound Field Reconstruction + Acoustic Contrast Control
=============================================================
Implements the plane-wave sparse reconstruction from eq. (16-17) in:
    Kristoffersen et al., "Deep Sound Field Reconstruction in Real Rooms", 2021

Pipeline:
    1. Load all measured RIRs from the 32x32 grid
    2. Simulate "sparse" observations by subsampling n_mic positions
    3. Solve L1-regularised plane wave decomposition at each frequency
       to reconstruct RTFs at any zone position in the room
    4. Run ACC using reconstructed RTFs → source weights q(ω)
    5. Evaluate and visualise contrast
"""

import numpy as np
import scipy.io
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from scipy.linalg import eigh
from scipy.optimize import minimize
from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate


# ---------------------------------------------------------------------------
# I/O
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

    sorted_pairs = sorted(zip(position_list, rir_list),
                          key=lambda x: (x[0][1], x[0][0]))
    position_list = [p[0] for p in sorted_pairs]
    rir_list      = [p[1] for p in sorted_pairs]
    return np.array(rir_list), position_list


def compute_transfer_function(rir_array, fs=48000, nfft=4096):
    """Returns H (n_points, n_sources, n_freqs) and freqs (n_freqs,)."""
    H     = np.fft.rfft(rir_array, n=nfft, axis=2)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return H, freqs


# ---------------------------------------------------------------------------
# Grid geometry helpers
# ---------------------------------------------------------------------------

def grid_positions(grid_size, dx, dy):
    """
    Returns (n_points, 2) array of (x, y) positions in metres for every
    flat grid index, matching the sort order used in compute_rir_array.
    Outer loop = Y (row), inner loop = X (col).
    """
    xs, ys = np.meshgrid(np.arange(grid_size) * dx,
                          np.arange(grid_size) * dy)
    # flatten in row-major order: index k → (k % grid_size, k // grid_size)
    pos = np.column_stack([xs.flatten(), ys.flatten()])
    return pos                                # (n_points, 2)


# ---------------------------------------------------------------------------
# Plane-wave dictionary
# ---------------------------------------------------------------------------

def build_plane_wave_dictionary(positions, freq_hz, c=343.0,
                                 n_angles=72, n_kr=8):
    """
    Build the steering matrix Φ ∈ C^{M × N} as in eq. (16).

    Candidate plane waves are sampled on a 2D wave-number grid:
        kn = (ω/c) * [cos θ, sin θ]   for θ ∈ [0, 2π)

    and additionally scaled versions kn * r for r ∈ (0,1] to provide
    sub-frequency candidates that help regularisation via L(ω).

    Args:
        positions : (M, 2)  observation positions in metres
        freq_hz   : scalar  frequency in Hz
        c         : speed of sound
        n_angles  : number of plane wave directions (azimuth)
        n_kr      : number of wavenumber magnitude steps per direction

    Returns:
        Phi : (M, N)  complex steering matrix
        kns : (N, 2)  wave vectors for each candidate plane wave
    """
    omega   = 2 * np.pi * freq_hz
    k_true  = omega / c                          # true wavenumber magnitude

    angles  = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    # Include the true wavenumber and sub-multiples
    k_mags  = k_true * np.linspace(1.0 / n_kr, 1.0, n_kr)

    kns = []
    for km in k_mags:
        for theta in angles:
            kns.append([km * np.cos(theta), km * np.sin(theta)])
    kns = np.array(kns)                          # (N, 2)

    # Φ[m, n] = exp(j kn · rm)
    Phi = np.exp(1j * positions @ kns.T)         # (M, N)
    return Phi, kns


def build_L_matrix(kns, freq_hz, c=343.0):
    """
    Diagonal weighting matrix L(ω) ∈ R^{N×N} from eq. (17).

    L_nn = | ‖kn‖² − (ω/c)² |
    Plane waves whose wavenumber matches the excitation frequency get
    weight ≈ 0 (not penalised); off-shell candidates are penalised
    strongly, promoting a physically sparse solution.
    """
    omega  = 2 * np.pi * freq_hz
    k_true = omega / c
    k_norms_sq = np.sum(kns ** 2, axis=1)        # (N,)
    diag   = np.abs(k_norms_sq - k_true ** 2)
    # Small floor to avoid complete zero penalty at exact matches
    diag   = np.maximum(diag, 1e-6)
    return np.diag(diag)                          # (N, N)


# ---------------------------------------------------------------------------
# Sparse reconstruction  (eq. 17 — L1 penalty)
# ---------------------------------------------------------------------------

def solve_sparse_reconstruction(s_obs, Phi, L, lam=1e-3):
    """
    Solve the sparse plane-wave reconstruction problem (eq. 17):

        min_b  ‖s_obs − Φ b‖²  +  λ ‖L b‖₁

    Since b is complex we split into real/imag and use scipy.optimize.minimize
    with L-BFGS-B, which supports bound constraints and handles large N.

    Args:
        s_obs : (M,)  complex observed pressures at mic positions
        Phi   : (M, N)  complex steering matrix
        L     : (N, N)  diagonal penalty matrix (real)
        lam   : regularisation weight λ

    Returns:
        b : (N,) complex plane-wave weights
    """
    M, N = Phi.shape
    L_diag = np.diag(L).real                     # (N,) diagonal weights

    # Stack real and imaginary parts: x = [b_re; b_im], length 2N
    def objective(x):
        b    = x[:N] + 1j * x[N:]
        resid = s_obs - Phi @ b
        data_fit = np.real(resid.conj() @ resid)
        # L1 on complex: ‖L b‖₁ = Σ_n L_nn |b_n|
        l1_pen   = lam * np.sum(L_diag * np.abs(b))
        return data_fit + l1_pen

    def gradient(x):
        b     = x[:N] + 1j * x[N:]
        resid = s_obs - Phi @ b
        # Gradient of data fit term
        grad_b  = -2 * Phi.conj().T @ resid      # complex (N,)
        # Sub-gradient of L1 term: sign(b) weighted by L_diag
        abs_b   = np.abs(b)
        phase_b = np.where(abs_b > 1e-12, b / np.where(abs_b > 1e-12, abs_b, 1.0), 0.0)
        grad_b += lam * L_diag * phase_b
        return np.concatenate([grad_b.real, grad_b.imag])

    x0  = np.zeros(2 * N)
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
    b = res.x[:N] + 1j * res.x[N:]
    return b


def reconstruct_at_positions(b, target_positions, kns):
    """
    Evaluate the reconstructed sound field at arbitrary target positions.

        p(r) = Σ_n b_n exp(j kn · r)   = Phi_target @ b

    Args:
        b               : (N,) complex plane-wave weights
        target_positions: (K, 2)  target positions in metres
        kns             : (N, 2)  wave vectors

    Returns:
        p : (K,) complex pressures
    """
    Phi_target = np.exp(1j * target_positions @ kns.T)   # (K, N)
    return Phi_target @ b


# ---------------------------------------------------------------------------
# Acoustic Contrast Control  (eq. 14-15, adaptive regularisation)
# ---------------------------------------------------------------------------

def acoustic_contrast_control(H_bright, H_dark, freq_idx):
    """
    ACC via generalised eigenvalue problem (eq. 14-15).
    λ_D = 0.01 * ‖R_d‖_2  (adaptive regularisation, eq. 15).
    Falls back to standard eigenproblem if R_d_reg is not positive definite.
    """
    if H_bright.ndim == 3:
        H_b = H_bright[:, :, freq_idx]
        H_d = H_dark[:,   :, freq_idx]
    else:
        H_b = H_bright
        H_d = H_dark

    n_sources = H_b.shape[1]
    R_b = H_b.conj().T @ H_b
    R_d = H_d.conj().T @ H_d

    # Adaptive regularisation: scale with spectral norm of R_d
    # Add a minimum floor in case R_d is near-zero (e.g. reconstructed
    # pressures vanish at this frequency)
    spectral_norm = np.linalg.norm(R_d, ord=2)
    lambda_D  = 0.01 * spectral_norm + 1e-8 * (np.linalg.norm(R_b, ord=2) + 1e-12)
    R_d_reg   = R_d + lambda_D * np.eye(n_sources)

    # Ensure R_d_reg is positive definite before calling eigh
    # If smallest eigenvalue is still <= 0, add more loading
    min_eig = np.linalg.eigvalsh(R_d_reg).min()
    if min_eig <= 0:
        R_d_reg += (-min_eig + 1e-8) * np.eye(n_sources)

    try:
        eigenvalues, eigenvectors = eigh(R_b, R_d_reg)
        W = eigenvectors[:, -1]
    except np.linalg.LinAlgError:
        # Last resort: solve as standard (non-generalised) eigenproblem
        R_d_inv = np.linalg.pinv(R_d_reg)
        eigenvalues, eigenvectors = np.linalg.eigh(R_d_inv @ R_b)
        W = eigenvectors[:, -1]

    W = W / (np.linalg.norm(W) + 1e-12)

    p_bright      = H_b @ W
    p_dark        = H_d @ W
    bright_energy = np.mean(np.abs(p_bright) ** 2)
    dark_energy   = np.mean(np.abs(p_dark)   ** 2)
    contrast_dB   = 10 * np.log10((bright_energy + 1e-12) /
                                   (dark_energy   + 1e-12))
    return W, contrast_dB, bright_energy, dark_energy, lambda_D


# ---------------------------------------------------------------------------
# Zone selection (Dash)
# ---------------------------------------------------------------------------

def choose_zones(grid_size=32, dx=0.225, dy=0.262):
    """
    Interactive zone selector.  User lasso-selects bright and dark zones.
    Returns bright_indices, dark_indices as lists of flat grid indices,
    and the zone positions in metres.
    """
    xs_m = (np.arange(grid_size * grid_size) % grid_size) * dx
    ys_m = (np.arange(grid_size * grid_size) // grid_size) * dy
    df   = pd.DataFrame({'x': xs_m, 'y': ys_m, 'zone': 'Unassigned'})

    app    = Dash(__name__)
    result = {"bright": [], "dark": []}

    app.layout = html.Div([
        dcc.Graph(id="grid"),
        html.Div([
            html.Button("Assign Bright Zone", id="bright-btn", n_clicks=0,
                        style={"backgroundColor": "#4a90e2", "color": "white",
                               "padding": "8px 16px", "border": "none",
                               "borderRadius": "4px", "marginRight": "10px"}),
            html.Button("Assign Dark Zone", id="dark-btn", n_clicks=0,
                        style={"backgroundColor": "#e74c3c", "color": "white",
                               "padding": "8px 16px", "border": "none",
                               "borderRadius": "4px"}),
        ]),
        html.Div(id="status", style={"marginTop": "10px", "fontWeight": "bold"}),
        html.Div("Lasso-select points, assign zone, repeat. "
                 "Close the browser tab when done.",
                 style={"color": "gray", "marginTop": "8px"}),
        dcc.Store(id="zone-store", data=df.to_dict("records")),
    ])

    @app.callback(Output("grid", "figure"), Input("zone-store", "data"))
    def update_figure(zone_data):
        df_d = pd.DataFrame(zone_data)
        result["bright"] = df_d[df_d['zone'] == 'Bright'].index.tolist()
        result["dark"]   = df_d[df_d['zone'] == 'Dark'].index.tolist()
        fig = px.scatter(df_d, x='x', y='y', color='zone',
                         title=f"Select Bright & Dark Zones ({grid_size}×{grid_size})",
                         color_discrete_map={"Unassigned": "lightgray",
                                             "Bright": "#4a90e2",
                                             "Dark":   "#e74c3c"})
        fig.update_layout(dragmode="lasso")
        fig.update_traces(marker=dict(size=7))
        return fig

    @app.callback(
        Output("zone-store", "data"), Output("status", "children"),
        Input("bright-btn", "n_clicks"), Input("dark-btn", "n_clicks"),
        Input("grid", "selectedData"), State("zone-store", "data"),
        prevent_initial_call=True
    )
    def handle_selection(bc, dc, selected_data, zone_data):
        ctx = callback_context
        if not ctx.triggered or selected_data is None:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger not in ["bright-btn", "dark-btn"]:
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        label = "Bright" if trigger == "bright-btn" else "Dark"
        df = pd.DataFrame(zone_data)
        for pt in selected_data.get("points", []):
            mask = (np.isclose(df['x'], pt['x']) &
                    np.isclose(df['y'], pt['y']))
            df.loc[mask, 'zone'] = label
        nb = (df['zone'] == 'Bright').sum()
        nd = (df['zone'] == 'Dark').sum()
        return df.to_dict("records"), f"Bright: {nb} pts | Dark: {nd} pts"

    app.run(debug=False, port=8050)

    all_pos = np.column_stack([xs_m, ys_m])
    bright_pos = all_pos[result["bright"]]
    dark_pos   = all_pos[result["dark"]]
    return result["bright"], result["dark"], bright_pos, dark_pos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Config ────────────────────────────────────────────────────────────────
    # VR Lab geometry
    ROOM_LX    = 6.98
    ROOM_LY    = 8.12
    GRID_N     = 32
    DX         = ROOM_LX / (GRID_N - 1)   # 0.225 m
    DY         = ROOM_LY / (GRID_N - 1)   # 0.262 m
    C          = 343.0
    FS         = 48000
    NFFT       = 4096

    # Source positions (metres)
    SOURCE1_POS = np.array([6.65, 7.93])   # Pos1 — corner
    SOURCE2_POS = np.array([5.23, 3.49])   # Pos2 — mid-room

    # Sparse reconstruction settings — tuned for low frequencies (<300 Hz)
    N_MIC_OBS   = 25      # more mics helps at low freq (half-wavelength at 300Hz ≈ 57cm)
    N_ANGLES    = 36      # fewer directions — low-freq fields are smoother
    N_KR        = 4       # fewer wavenumber steps — less dictionary bloat
    LAM_SPARSE  = None    # None = use frequency-adaptive λ (see below)
    RANDOM_SEED = 42

    # Frequency range — limit to where sparse reconstruction is reliable
    FREQ_MIN    = 50
    FREQ_MAX    = 300
    # ─────────────────────────────────────────────────────────────────────────

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading RIR data …")
    rir_s1, position_list = compute_rir_array("./VRLAB-data/source_1/h_100/")
    rir_s2, _             = compute_rir_array("./VRLAB-data/source_2/h_100/")
    rir_array = np.stack([rir_s1, rir_s2], axis=1)   # (1024, 2, n_samples)
    print(f"RIR shape: {rir_array.shape}")

    H, freqs = compute_transfer_function(rir_array, fs=FS, nfft=NFFT)
    # H: (1024, 2, n_freqs)

    # All grid positions in metres — only for measured points
    all_positions = grid_positions(GRID_N, DX, DY)   # (1024, 2) full grid
    # H rows correspond to the first n_measured positions in sorted order

    # ── Zone selection ────────────────────────────────────────────────────────
    print("\nOpening zone selection …")
    bright_idx, dark_idx, bright_pos, dark_pos = choose_zones(
        grid_size=GRID_N, dx=DX, dy=DY
    )

    if len(bright_idx) == 0 or len(dark_idx) == 0:
        print("Both zones must be selected — exiting.")
        exit()

    # Clamp zone indices to valid measured range (H may have fewer rows
    # than GRID_N^2 due to unmeasured positions in the room)
    n_measured   = H.shape[0]
    bright_idx   = [i for i in bright_idx if i < n_measured]
    dark_idx     = [i for i in dark_idx   if i < n_measured]
    bright_pos   = all_positions[bright_idx]
    dark_pos     = all_positions[dark_idx]

    if len(bright_idx) == 0 or len(dark_idx) == 0:
        print("No valid measured positions in one of the zones — "
              "please reselect zones within the measured area.")
        exit()

    print(f"Bright zone: {len(bright_idx)} pts | Dark zone: {len(dark_idx)} pts "
          f"(after clamping to {n_measured} measured positions)")

    # ── Random observation positions (virtual microphones) ────────────────────
    np.random.seed(RANDOM_SEED)
    # H only has n_measured rows (may be < GRID_N^2 due to obstacles)
    n_measured = H.shape[0]
    # Sample from valid measured indices, excluding zone points
    non_zone = list(set(range(n_measured)) - set(bright_idx) - set(dark_idx))
    obs_idx  = np.random.choice(non_zone, size=min(N_MIC_OBS, len(non_zone)),
                                 replace=False)
    obs_pos  = all_positions[obs_idx]               # (n_mic, 2)
    print(f"Observation positions: {len(obs_idx)} virtual microphones")

    # ── Frequency sweep with sparse reconstruction ────────────────────────────
    freq_mask    = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    freq_indices = np.where(freq_mask)[0]
    step         = max(1, len(freq_indices) // 30)
    freq_indices = freq_indices[::step]

    contrasts_sparse = []    # ACC using reconstructed RTFs
    contrasts_true   = []    # ACC using true measured RTFs (upper bound)
    freq_vals        = []

    print(f"\nRunning sparse reconstruction + ACC at {len(freq_indices)} frequencies …")
    print(f"{'Freq':>8} {'Contrast (sparse)':>18} {'Contrast (true)':>16}")
    print("-" * 46)

    for fi in freq_indices:
        # Frequency-adaptive λ: higher frequencies need stronger regularisation
        # Scale λ quadratically with frequency — at low freq the field is
        # smooth so less penalty needed; at higher freq modes are denser
        freq_hz = freqs[fi]
        lam = 1e-3 * (freq_hz / FREQ_MIN) ** 2 if LAM_SPARSE is None else LAM_SPARSE

        # Build plane wave dictionary at this frequency
        Phi_obs, kns = build_plane_wave_dictionary(
            obs_pos, freq_hz, c=C, n_angles=N_ANGLES, n_kr=N_KR
        )
        L = build_L_matrix(kns, freq_hz, c=C)

        # --- Sparse reconstruction per source ---
        # For each loudspeaker: use observed pressures at obs_idx to
        # recover plane wave weights, then predict at zone positions
        H_b_recon = np.zeros((len(bright_idx), 2), dtype=complex)
        H_d_recon = np.zeros((len(dark_idx),   2), dtype=complex)

        for src in range(2):
            # Observed pressures at virtual mic positions for this source
            s_obs = H[obs_idx, src, fi]              # (n_mic,)

            # Solve sparse reconstruction (eq. 17)
            b = solve_sparse_reconstruction(s_obs, Phi_obs, L, lam=lam)

            # Reconstruct at bright and dark zone positions
            H_b_recon[:, src] = reconstruct_at_positions(b, bright_pos, kns)
            H_d_recon[:, src] = reconstruct_at_positions(b, dark_pos,   kns)

        # ACC with reconstructed RTFs
        W_sp, c_sp, _, _, _ = acoustic_contrast_control(
            H_b_recon, H_d_recon, freq_idx=None
        )

        # ACC with true measured RTFs (upper bound / oracle)
        H_b_true = H[bright_idx, :, fi]
        H_d_true = H[dark_idx,   :, fi]
        W_tr, c_tr, _, _, _ = acoustic_contrast_control(
            H_b_true, H_d_true, freq_idx=None
        )

        contrasts_sparse.append(c_sp)
        contrasts_true.append(c_tr)
        freq_vals.append(freq_hz)
        print(f"{freq_hz:>8.1f} {c_sp:>18.2f} {c_tr:>16.2f}")

    contrasts_sparse = np.array(contrasts_sparse)
    contrasts_true   = np.array(contrasts_true)

    print(f"\nSparse reconstruction — mean contrast : {np.mean(contrasts_sparse):.2f} dB")
    print(f"True RTFs (oracle)    — mean contrast : {np.mean(contrasts_true):.2f} dB")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(
        f"Sparse Reconstruction + ACC  |  "
        f"n_mic={len(obs_idx)}, λ=adaptive ({FREQ_MIN}–{FREQ_MAX} Hz)  |  "
        f"Bright={len(bright_idx)} pts, Dark={len(dark_idx)} pts",
        fontsize=10
    )

    # Contrast comparison
    axes[0].plot(freq_vals, contrasts_true,   'b-o', markersize=5,
                 label=f'True RTFs (oracle)  mean={np.mean(contrasts_true):.1f} dB')
    axes[0].plot(freq_vals, contrasts_sparse, 'g-o', markersize=5,
                 label=f'Sparse recon  mean={np.mean(contrasts_sparse):.1f} dB')
    axes[0].axhline(0,  color='k',      linestyle='--', alpha=0.4)
    axes[0].axhline(10, color='orange', linestyle='--', alpha=0.4, label='10 dB')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Contrast (dB)')
    axes[0].set_title('Acoustic Contrast: Sparse vs True RTFs')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Gap between oracle and sparse (reconstruction quality indicator)
    gap = np.array(contrasts_true) - np.array(contrasts_sparse)
    axes[1].bar(freq_vals, gap,
                width=np.diff(freq_vals, prepend=freq_vals[0]) * 0.6,
                color=['#e74c3c' if g > 3 else '#2ecc71' for g in gap],
                alpha=0.8)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.4)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Contrast gap (dB)')
    axes[1].set_title('Oracle − Sparse Gap  (green = reconstruction good)')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('sparse_acc_results.png', dpi=150)
    plt.show()

    # ── Spatial map at best sparse contrast frequency ─────────────────────────
    best_i      = int(np.argmax(contrasts_sparse))
    best_fi     = freq_indices[best_i]
    best_freq   = freqs[best_fi]
    lam_best    = 1e-3 * (best_freq / FREQ_MIN) ** 2 if LAM_SPARSE is None else LAM_SPARSE

    # Reconstruct pressure across the whole room at this frequency
    Phi_obs, kns = build_plane_wave_dictionary(
        obs_pos, best_freq, c=C, n_angles=N_ANGLES, n_kr=N_KR
    )
    L = build_L_matrix(kns, best_freq, c=C)

    # Get optimal weights from sparse ACC at best frequency
    H_b_recon = np.zeros((len(bright_idx), 2), dtype=complex)
    H_d_recon = np.zeros((len(dark_idx),   2), dtype=complex)
    b_per_src = []
    for src in range(2):
        s_obs = H[obs_idx, src, best_fi]
        b     = solve_sparse_reconstruction(s_obs, Phi_obs, L, lam=lam_best)
        b_per_src.append(b)
        H_b_recon[:, src] = reconstruct_at_positions(b, bright_pos, kns)
        H_d_recon[:, src] = reconstruct_at_positions(b, dark_pos, kns)

    W_best, contrast_best, _, _, _ = acoustic_contrast_control(
        H_b_recon, H_d_recon, freq_idx=None
    )

    # Reconstruct pressure at ALL measured grid positions using W_best
    p_vals = np.zeros(n_measured, dtype=complex)
    for src in range(2):
        b = b_per_src[src]
        p_vals += reconstruct_at_positions(b, all_positions[:n_measured], kns) * W_best[src]

    # Scatter plot instead of imshow since grid may have missing points
    meas_x = all_positions[:n_measured, 0]
    meas_y = all_positions[:n_measured, 1]
    p_abs  = np.abs(p_vals)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle(
        f"Spatial Map at {best_freq:.0f} Hz  |  "
        f"Sparse contrast = {contrast_best:.1f} dB",
        fontsize=11
    )

    im = axes2[0].scatter(meas_x, meas_y, c=p_abs, cmap='hot',
                           s=30, marker='s')
    plt.colorbar(im, ax=axes2[0], label='|p| reconstructed')

    bx  = bright_pos[:, 0];  by  = bright_pos[:, 1]
    dx_ = dark_pos[:, 0];    dy_ = dark_pos[:, 1]
    ox  = obs_pos[:, 0];     oy  = obs_pos[:, 1]

    axes2[0].scatter(bx, by,   c='cyan',   s=50, marker='o',
                     label='Bright zone', zorder=5)
    axes2[0].scatter(dx_, dy_, c='lime',   s=30, marker='x',
                     label='Dark zone',   zorder=5)
    axes2[0].scatter(ox, oy,   c='white',  s=20, marker='^',
                     label='Obs. mics',   zorder=5, alpha=0.7)
    axes2[0].scatter(*SOURCE1_POS, c='white',  s=200, marker='*',
                     label='Source 1', zorder=6, edgecolors='black')
    axes2[0].scatter(*SOURCE2_POS, c='yellow', s=200, marker='*',
                     label='Source 2', zorder=6, edgecolors='black')
    axes2[0].set_xlabel('x (m)');  axes2[0].set_ylabel('y (m)')
    axes2[0].set_title('Reconstructed Pressure |p|')
    axes2[0].legend(fontsize=7, loc='upper left')

    # Zone + mic layout
    zone_colors = np.zeros(n_measured)
    zone_colors[bright_idx] = 1
    zone_colors[dark_idx]   = 2

    from matplotlib.colors import ListedColormap
    cmap_z = ListedColormap(['lightgray', '#4a90e2', '#e74c3c'])
    axes2[1].scatter(meas_x, meas_y, c=zone_colors, cmap=cmap_z,
                     vmin=0, vmax=2, s=25, marker='s')
    axes2[1].scatter(ox, oy, c='black', s=25, marker='^',
                     label='Obs. mics', zorder=5)
    axes2[1].scatter(*SOURCE1_POS, c='white',  s=200, marker='*',
                     label='Source 1', zorder=6, edgecolors='black')
    axes2[1].scatter(*SOURCE2_POS, c='yellow', s=200, marker='*',
                     label='Source 2', zorder=6, edgecolors='black')
    axes2[1].set_xlabel('x (m)');  axes2[1].set_ylabel('y (m)')
    axes2[1].set_title('Zone Layout + Observation Mics')
    axes2[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('sparse_acc_spatial.png', dpi=150)
    plt.show()
    print("Saved → sparse_acc_results.png, sparse_acc_spatial.png")

    # ── Seed stability analysis ───────────────────────────────────────────────
    # Run sparse reconstruction with N_SEEDS different random mic placements
    # to check how stable the contrast is across different observation sets
    print("\nRunning seed stability analysis …")
    N_SEEDS = 10
    all_seed_contrasts = []

    for seed in range(N_SEEDS):
        np.random.seed(seed)
        obs_idx_s = np.random.choice(non_zone,
                                      size=min(N_MIC_OBS, len(non_zone)),
                                      replace=False)
        obs_pos_s = all_positions[obs_idx_s]

        seed_contrasts = []
        for fi in freq_indices:
            fhz  = freqs[fi]
            lam  = 1e-3 * (fhz / FREQ_MIN) ** 2 if LAM_SPARSE is None else LAM_SPARSE
            Phi_s, kns_s = build_plane_wave_dictionary(
                obs_pos_s, fhz, c=C, n_angles=N_ANGLES, n_kr=N_KR
            )
            L_s = build_L_matrix(kns_s, fhz, c=C)

            H_b_s = np.zeros((len(bright_idx), 2), dtype=complex)
            H_d_s = np.zeros((len(dark_idx),   2), dtype=complex)
            for src in range(2):
                b_s = solve_sparse_reconstruction(
                    H[obs_idx_s, src, fi], Phi_s, L_s, lam=lam
                )
                H_b_s[:, src] = reconstruct_at_positions(b_s, bright_pos, kns_s)
                H_d_s[:, src] = reconstruct_at_positions(b_s, dark_pos,   kns_s)

            _, c_s, _, _, _ = acoustic_contrast_control(H_b_s, H_d_s, freq_idx=None)
            seed_contrasts.append(c_s)

        all_seed_contrasts.append(seed_contrasts)
        print(f"  Seed {seed:2d}: mean contrast = {np.mean(seed_contrasts):.2f} dB")

    all_seed_contrasts = np.array(all_seed_contrasts)   # (N_SEEDS, n_freqs)
    mean_across_seeds  = all_seed_contrasts.mean(axis=0)
    std_across_seeds   = all_seed_contrasts.std(axis=0)

    fig3, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq_vals, contrasts_true,   'b-o', markersize=4,
            label=f'Oracle (mean={np.mean(contrasts_true):.1f} dB)', zorder=5)
    ax.plot(freq_vals, mean_across_seeds, 'g-o', markersize=4,
            label=f'Sparse mean across {N_SEEDS} seeds '
                  f'({mean_across_seeds.mean():.1f} dB)', zorder=5)
    ax.fill_between(freq_vals,
                     mean_across_seeds - std_across_seeds,
                     mean_across_seeds + std_across_seeds,
                     alpha=0.25, color='green', label='±1 std dev')
    ax.axhline(0,  color='k',      linestyle='--', alpha=0.4)
    ax.axhline(10, color='orange', linestyle='--', alpha=0.4, label='10 dB')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Contrast (dB)')
    ax.set_title(f'Sparse Reconstruction Stability  |  '
                 f'n_mic={N_MIC_OBS}, {N_SEEDS} random mic placements')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sparse_acc_stability.png', dpi=150)
    plt.show()
    print("Saved → sparse_acc_stability.png")