
import numpy as np

from data_extraction import create_FFT_grid, create_RTF_grid, magnitude_phase_plots

def nmse_db(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred)**2)
    denom = np.sum(np.abs(y_true)**2)
    
    eps = 1e-12 #avoids division by 0
    nmse = num / (denom + eps)
    
    nmse_db = 10 * np.log10(nmse + eps) #dB conversion
    
    return nmse_db

def normalize_grid(g):
    # normalize per frequency slice so scaling doesn't affect the comparison
    g_norm = np.zeros_like(g)
    for k in range(g.shape[-1]):
        m = np.max(np.abs(g[..., k]))
        if m > 0:
            g_norm[..., k] = g[..., k] / m
    return g_norm

if __name__ == "__main__":
    dir_real = './VRLAB-data/source_1/'
    grid_real, freq_real = create_FFT_grid(dir_real, fs=48000, target_freq=40, heights=[100])

    dir_sim = './Sound zones - MATLAB/individual_RTFs/source_1/'
    grid_sim, freq_sim = create_RTF_grid(dir_sim, target_freq=40, heights=[100])

    magnitude_phase_plots(grid_real, freq_real, [100], block=False)
    magnitude_phase_plots(grid_sim, freq_sim, [100])

    nmse_db = nmse_db(normalize_grid(grid_real), normalize_grid(grid_sim))
    print(f"Simulation Accuracy (NMSE): {nmse_db:.2f} dB")