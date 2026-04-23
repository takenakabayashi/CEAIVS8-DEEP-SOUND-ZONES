"""
magnitude_phase_plots:
    Plots the magnitude and phase from the created FFT grid
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy

def nmse_db(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred)**2)
    denom = np.sum(np.abs(y_true)**2)
    
    eps = 1e-12 #avoids division by 0
    nmse = num / (denom + eps)
    
    nmse_db = 10 * np.log10(nmse + eps) #dB conversion
    
    return nmse_db

#filter out points with pressure (approximately) equal to zero
def filter_zero_targets(X, y, magnitude_threshold=1e-8):
    target_magnitude = np.abs(y).flatten()
    valid_mask = target_magnitude > magnitude_threshold

    return X[valid_mask], y[valid_mask], valid_mask

def stack_complex_targets(y_complex):
    return np.hstack(
        (
            np.real(y_complex).astype(np.float32),
            np.imag(y_complex).astype(np.float32),
        )
    )

def validation_nmse_metric(y_true, y_pred):
    y_true_complex = y_true[:, 0] + 1j * y_true[:, 1]
    y_pred_complex = y_pred[:, 0] + 1j * y_pred[:, 1]
    return nmse_db(y_true_complex, y_pred_complex)

def compute_fft_at_target_freq(x, fs, target_freq):
    X = np.fft.rfft(x)

    #FFT is discrete: it computes the FFT at specific frequencies bins determined by the sampling frequency and the length of the input signal
    freq = np.fft.rfftfreq(x.size, d=1/fs) #array of frequencies corresponding to the bins
    idx = np.argmin(np.abs(freq - target_freq)) #finds index where frequency is closest to target_freq

    return X[idx], freq[idx] #returns the FFT value at the closest sampled frequency (freq[idx]) to the target frequency

#Creates a 32x32xlen(heights) grid of the FFT values at a target frequency for each impulse response for all specified heights
#NOTE: we are using a 32x32 grid with indexes 0-31, but in the ISOBEL measurements reports the points are at the intersection of the tiles, not on the tiles themselves, so it's actually a 31x31 grid?
#TODO: fix the grid
def create_FFT_grid(room: dict, source, fs, target_freq):
    grid = np.zeros(room["grid_size"], dtype=complex)

    directory = Path(room["directory"]) / f"source_{source}"
    heights = room["heights"]

    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist")

    for height_idx, height_val in enumerate(heights):
        height_dir = directory / f"h_{height_val}"
        if os.path.isdir(height_dir):
            for entry in os.scandir(height_dir):
                if entry.is_file() and entry.name.endswith('.mat'):
                    idxX = int(entry.name.split('_')[1])
                    idxY = int(entry.name.split('_')[-1].split('.')[0])

                    ir_data = scipy.io.loadmat(entry.path)['ImpulseResponse'].flatten()

                    #files indexes go from 1 to N, grid has indexes 0-(N-1) so we subtract 1
                    grid[idxX-1, idxY-1, height_idx], approximated_freq = compute_fft_at_target_freq(ir_data, fs, target_freq)

        else:
            raise ValueError(f"Directory {height_dir} does not exist")
        
    return grid, approximated_freq

def magnitude_phase_plots(room: dict, source: int, fs: float, target_freq: float, block=True):
    grid, approx_freq = create_FFT_grid(room, source, fs, target_freq)

    magnitude = np.abs(grid)
    phase = np.angle(grid)
    
    plot_extent = (0, room["room_dimensions"][0], 0, room["room_dimensions"][1])

    fig = plt.figure(figsize=(18, 14), layout="constrained")
    fig.suptitle(f'Room Response at {approx_freq:.1f} Hz', fontsize=18)
    
    #this is to adapt the number of subplots to the number of heights
    n = len(room["heights"])
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    
    subfigs = fig.subfigures(nrows, ncols)
    subfigs_flat = np.array(subfigs).flatten()

    for height_idx, height_val in enumerate(room["heights"]):
        sf = subfigs_flat[height_idx]
        
        sf.suptitle(f'Height {height_val}', fontsize=14)
        
        axs = sf.subplots(1, 2)
        
        mag_slice = magnitude[:, :, height_idx]
        phase_slice = phase[:, :, height_idx]

        # Magnitude
        ax_mag = axs[0]
        mag_plot = ax_mag.imshow(mag_slice, extent=plot_extent, origin='lower', cmap='viridis', aspect='equal')
        ax_mag.set_title('Magnitude', fontsize=10)
        ax_mag.set_xlabel('X (m)')
        ax_mag.set_ylabel('Y (m)')
        sf.colorbar(mag_plot, ax=ax_mag, shrink=0.8)

        # Phase
        ax_phase = axs[1]
        phase_plot = ax_phase.imshow(phase_slice, extent=plot_extent, origin='lower', cmap='twilight', aspect='equal')
        ax_phase.set_title('Phase', fontsize=10)
        ax_phase.set_xlabel('X (m)')
        ax_phase.set_ylabel('Y (m)')
        sf.colorbar(phase_plot, ax=ax_phase, shrink=0.8)

    plt.show(block=block)
