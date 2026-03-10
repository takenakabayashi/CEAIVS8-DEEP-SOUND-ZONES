import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from scipy.linalg import eigh


def acc_solve(H_bright, H_dark, freq_idx=None):
    """
    Solve ACC at a single frequency.
    H_bright / H_dark can be (n_pts, n_src, n_freqs) or (n_pts, n_src) if
    freq_idx is None.
    Returns W (n_src,), contrast_dB, bright_energy, dark_energy, array_effort_dB.
    """
    H_b = H_bright[:, :, freq_idx] if H_bright.ndim == 3 else H_bright
    H_d = H_dark[:,   :, freq_idx] if H_dark.ndim   == 3 else H_dark

    n_src = H_b.shape[1]
    R_b   = H_b.conj().T @ H_b
    R_d   = H_d.conj().T @ H_d

    lam   = 0.01 * np.linalg.norm(R_d, ord=2) + 1e-8 * np.linalg.norm(R_b, ord=2)
    R_d_r = R_d + lam * np.eye(n_src)

    # Ensure positive definiteness
    min_eig = np.linalg.eigvalsh(R_d_r).min()
    if min_eig <= 0:
        R_d_r += (-min_eig + 1e-10) * np.eye(n_src)

    try:
        _, vecs = eigh(R_b, R_d_r)
        W = vecs[:, -1]
    except np.linalg.LinAlgError:
        _, vecs = np.linalg.eigh(np.linalg.pinv(R_d_r) @ R_b)
        W = vecs[:, -1]

    W = W / (np.linalg.norm(W) + 1e-12)

    p_b = H_b @ W
    p_d = H_d @ W

    bright_e    = np.mean(np.abs(p_b) ** 2)
    dark_e      = np.mean(np.abs(p_d) ** 2)
    contrast_dB = 10 * np.log10((bright_e + 1e-12) / (dark_e + 1e-12))
    effort_dB   = 10 * np.log10(np.real(W.conj() @ W) + 1e-12)

    return W, contrast_dB, bright_e, dark_e, effort_dB

def evaluate_zone_quality(H, freqs, bright_idx, dark_idx,
                           all_positions, grid_size, dx, dy,
                           source_positions=None,
                           freq_min=100, freq_max=8000,
                           n_map_freqs=4,
                           save_prefix='zone_eval'):
    H_B = H[bright_idx]
    H_D = H[dark_idx]

    # Frequency indices to evaluate
    mask    = (freqs >= freq_min) & (freqs <= freq_max)
    fi_all  = np.where(mask)[0]
    step    = max(1, len(fi_all) // 60)
    fi_eval = fi_all[::step]

    contrasts, efforts, weights = [], [], []

    for fi in fi_eval:
        W, c, be, de, ae = acc_solve(H_B, H_D, freq_idx=fi)
        contrasts.append(c)
        efforts.append(ae)
        weights.append(W)

    contrasts = np.array(contrasts)
    efforts   = np.array(efforts)
    freq_vals = freqs[fi_eval]

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Sound Zone Quality Report")
    print("=" * 55)
    print(f"  Bright zone : {len(bright_idx):>4} control points")
    print(f"  Dark zone   : {len(dark_idx):>4} control points")
    print(f"  Freq range  : {freq_min}–{freq_max} Hz  ({len(fi_eval)} bins)")
    print(f"  N sources   : {H.shape[1]}")
    print("-" * 55)
    print(f"  Acoustic Contrast")
    print(f"    Mean   : {np.mean(contrasts):>7.2f} dB")
    print(f"    Median : {np.median(contrasts):>7.2f} dB")
    print(f"    Max    : {np.max(contrasts):>7.2f} dB  @ {freq_vals[np.argmax(contrasts)]:.0f} Hz")
    print(f"    Min    : {np.min(contrasts):>7.2f} dB  @ {freq_vals[np.argmin(contrasts)]:.0f} Hz")
    print(f"    % bins > 0  dB : {100*np.mean(contrasts > 0):>5.1f}%")
    print(f"    % bins > 5  dB : {100*np.mean(contrasts > 5):>5.1f}%")
    print(f"    % bins > 10 dB : {100*np.mean(contrasts > 10):>5.1f}%")
    print("-" * 55)
    print(f"  Array Effort")
    print(f"    Mean   : {np.mean(efforts):>7.2f} dB")
    print(f"    Max    : {np.max(efforts):>7.2f} dB  @ {freq_vals[np.argmax(efforts)]:.0f} Hz")
    print("=" * 55)

    # ── Figure 1: Metrics across frequency ───────────────────────────────────
    fig1, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig1.suptitle(
        f"Zone Quality Metrics  |  Bright={len(bright_idx)} pts, "
        f"Dark={len(dark_idx)} pts  |  {H.shape[1]} sources",
        fontsize=11
    )

    # Acoustic contrast
    ax = axes[0]
    ax.plot(freq_vals, contrasts, color='#2ecc71', linewidth=1.5)
    ax.fill_between(freq_vals, 0, contrasts,
                    where=contrasts >= 0,  alpha=0.15, color='#2ecc71')
    ax.fill_between(freq_vals, 0, contrasts,
                    where=contrasts < 0,   alpha=0.15, color='#e74c3c')
    ax.axhline(0,  color='k',      linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(10, color='orange', linestyle='--', alpha=0.5, linewidth=1,
               label='10 dB target')
    ax.axhline(np.mean(contrasts), color='#27ae60', linestyle=':',
               alpha=0.8, label=f'Mean {np.mean(contrasts):.1f} dB')
    ax.set_ylabel('Acoustic Contrast (dB)')
    ax.set_title('Acoustic Contrast  —  higher is better, >10 dB is good')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(min(-5, np.min(contrasts) - 2), max(20, np.max(contrasts) + 2))

    # Array effort
    ax = axes[1]
    ax.plot(freq_vals, efforts, color='#e67e22', linewidth=1.5)
    ax.axhline(np.mean(efforts), color='#d35400', linestyle=':',
               alpha=0.8, label=f'Mean {np.mean(efforts):.1f} dB')
    ax.set_ylabel('Array Effort (dB)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Array Effort  —  lower is better, very high values are impractical')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_metrics.png', dpi=150)
    plt.show()

    # ── Figure 2: Spatial pressure maps at selected frequencies ──────────────
    # Pick n_map_freqs evenly spaced frequencies to visualise
    map_indices = np.linspace(0, len(fi_eval) - 1, n_map_freqs, dtype=int)
    map_fi      = fi_eval[map_indices]

    n_measured = H.shape[0]
    meas_x = all_positions[:n_measured, 0]
    meas_y = all_positions[:n_measured, 1]

    fig2, axes2 = plt.subplots(2, n_map_freqs,
                                figsize=(4 * n_map_freqs, 8))
    fig2.suptitle('Spatial Pressure Maps  |  Row 1: |p| magnitude, '
                  'Row 2: Zone layout', fontsize=11)

    # Zone colour overlay
    zone_colors = np.zeros(n_measured)
    zone_colors[bright_idx] = 1
    zone_colors[dark_idx]   = 2
    cmap_z = ListedColormap(['#dddddd', '#4a90e2', '#e74c3c'])

    for col, fi in enumerate(map_fi):
        W, c, _, _, _ = acc_solve(H_B, H_D, freq_idx=fi)

        # Pressure at all measured positions
        H_all = H[:n_measured, :, fi]
        p_all = np.abs(H_all @ W)

        freq_hz = freqs[fi]

        # Row 1: pressure magnitude
        ax = axes2[0, col]
        sc = ax.scatter(meas_x, meas_y, c=p_all, cmap='hot',
                        s=18, marker='s', vmin=0, vmax=np.percentile(p_all, 95))
        plt.colorbar(sc, ax=ax, label='|p|', shrink=0.8)

        # Overlay zone markers
        bx = all_positions[bright_idx, 0]
        by = all_positions[bright_idx, 1]
        dx_ = all_positions[dark_idx, 0]
        dy_ = all_positions[dark_idx, 1]
        ax.scatter(bx, by,   c='cyan',  s=25, marker='o', zorder=5,
                   linewidths=0.5, edgecolors='black')
        ax.scatter(dx_, dy_, c='lime',  s=20, marker='x', zorder=5)

        if source_positions:
            for k, sp in enumerate(source_positions):
                col_s = 'white' if k == 0 else 'yellow'
                ax.scatter(*sp, c=col_s, s=150, marker='*', zorder=6,
                           edgecolors='black', linewidths=0.5)

        ax.set_title(f'{freq_hz:.0f} Hz\nContrast={c:.1f} dB', fontsize=9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(0, grid_size * dx)
        ax.set_ylim(0, grid_size * dy)

        # Row 2: zone layout (same for all columns, but repeated for alignment)
        ax2 = axes2[1, col]
        ax2.scatter(meas_x, meas_y, c=zone_colors, cmap=cmap_z,
                    vmin=0, vmax=2, s=18, marker='s')
        if source_positions:
            for k, sp in enumerate(source_positions):
                col_s = 'white' if k == 0 else 'yellow'
                ax2.scatter(*sp, c=col_s, s=150, marker='*', zorder=6,
                            edgecolors='black', linewidths=0.5)
        ax2.set_title('Zone layout\n(blue=bright, red=dark)', fontsize=9)
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_xlim(0, grid_size * dx)
        ax2.set_ylim(0, grid_size * dy)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_spatial.png', dpi=150)
    plt.show()

    return {
        'freqs':     freq_vals,
        'contrasts': contrasts,
        'efforts':   efforts,
        'weights':   weights,
    }


# ---------------------------------------------------------------------------
# Main — plug directly into your existing data loading
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import scipy.io, re, os, pandas as pd, plotly.express as px
    from dash import Dash, dcc, html, Input, Output, State, callback_context
    from dash.exceptions import PreventUpdate

    # ── Config ────────────────────────────────────────────────────────────────
    ROOM_LX = 6.98;  ROOM_LY = 8.12;  GRID_N = 32
    DX = ROOM_LX / (GRID_N - 1)       # 0.225 m
    DY = ROOM_LY / (GRID_N - 1)       # 0.262 m
    SOURCE_POSITIONS = [(6.65, 7.93), (5.23, 3.49)]
    FS = 48000;  NFFT = 4096
    FREQ_MIN = 100;  FREQ_MAX = 8000
    # ─────────────────────────────────────────────────────────────────────────

    def parse_position(filename):
        m = re.search(r'idxX_(\d+)_idxY_(\d+)', filename)
        return int(m.group(1)), int(m.group(2))

    def load_rirs(folder):
        rir_list, pos_list = [], []
        for f in sorted(os.listdir(folder)):
            if f.endswith('.mat'):
                idxX, idxY = parse_position(f)
                pos_list.append((idxX, idxY))
                d = scipy.io.loadmat(os.path.join(folder, f))
                rir_list.append(d['ImpulseResponse'].flatten())
        pairs = sorted(zip(pos_list, rir_list), key=lambda x: (x[0][1], x[0][0]))
        return np.array([p[1] for p in pairs])

    print("Loading RIRs …")
    rir1 = load_rirs("./VRLAB-data/source_1/h_100/")
    rir2 = load_rirs("./VRLAB-data/source_2/h_100/")
    rir  = np.stack([rir1, rir2], axis=1)

    H     = np.fft.rfft(rir,  n=NFFT, axis=2)
    freqs = np.fft.rfftfreq(NFFT, d=1.0 / FS)

    xs_m = (np.arange(GRID_N * GRID_N) % GRID_N) * DX
    ys_m = (np.arange(GRID_N * GRID_N) // GRID_N) * DY
    all_positions = np.column_stack([xs_m, ys_m])

    # ── Zone selection (reuse Dash app) ───────────────────────────────────────
    df  = pd.DataFrame({'x': xs_m, 'y': ys_m, 'zone': 'Unassigned'})
    app = Dash(__name__)
    res = {"bright": [], "dark": []}

    app.layout = html.Div([
        dcc.Graph(id="grid"),
        html.Div([
            html.Button("Bright Zone", id="bright-btn", n_clicks=0,
                        style={"backgroundColor":"#4a90e2","color":"white",
                               "padding":"8px 16px","border":"none",
                               "borderRadius":"4px","marginRight":"10px"}),
            html.Button("Dark Zone", id="dark-btn", n_clicks=0,
                        style={"backgroundColor":"#e74c3c","color":"white",
                               "padding":"8px 16px","border":"none",
                               "borderRadius":"4px"}),
        ]),
        html.Div(id="status", style={"marginTop":"10px","fontWeight":"bold"}),
        html.Div("Lasso-select, assign zone, repeat. Close tab when done.",
                 style={"color":"gray","marginTop":"8px"}),
        dcc.Store(id="zone-store", data=df.to_dict("records")),
    ])

    @app.callback(Output("grid","figure"), Input("zone-store","data"))
    def upd_fig(zd):
        df_d = pd.DataFrame(zd)
        res["bright"] = df_d[df_d['zone']=='Bright'].index.tolist()
        res["dark"]   = df_d[df_d['zone']=='Dark'].index.tolist()
        fig = px.scatter(df_d, x='x', y='y', color='zone',
                         color_discrete_map={"Unassigned":"lightgray",
                                             "Bright":"#4a90e2","Dark":"#e74c3c"})
        fig.update_layout(dragmode="lasso")
        fig.update_traces(marker=dict(size=7))
        return fig

    @app.callback(
        Output("zone-store","data"), Output("status","children"),
        Input("bright-btn","n_clicks"), Input("dark-btn","n_clicks"),
        Input("grid","selectedData"), State("zone-store","data"),
        prevent_initial_call=True)
    def handle(bc, dc, sel, zd):
        ctx = callback_context
        if not ctx.triggered or sel is None: raise PreventUpdate
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        if trig not in ["bright-btn","dark-btn"]: raise PreventUpdate
        label = "Bright" if trig == "bright-btn" else "Dark"
        df = pd.DataFrame(zd)
        for pt in sel.get("points",[]):
            df.loc[(np.isclose(df['x'],pt['x']))&(np.isclose(df['y'],pt['y'])),'zone'] = label
        return df.to_dict("records"), f"Bright: {(df['zone']=='Bright').sum()} | Dark: {(df['zone']=='Dark').sum()}"

    app.run(debug=False, port=8050)

    n_measured  = H.shape[0]
    bright_idx  = [i for i in res["bright"] if i < n_measured]
    dark_idx    = [i for i in res["dark"]   if i < n_measured]

    if not bright_idx or not dark_idx:
        print("Need both zones selected."); exit()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = evaluate_zone_quality(
        H, freqs, bright_idx, dark_idx,
        all_positions, GRID_N, DX, DY,
        source_positions=SOURCE_POSITIONS,
        freq_min=FREQ_MIN, freq_max=FREQ_MAX,
        n_map_freqs=4,
        save_prefix='zone_eval'
    )