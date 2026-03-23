import os
import scipy
import numpy as np
import matplotlib.pyplot as plt

def compute_fft_at_target_freq(x, fs, target_freq):
    X = np.fft.rfft(x)

    #FFT is discrete: it computes the FFT at specific frequencies bins determined by the sampling frequency and the length of the input signal
    freq = np.fft.rfftfreq(x.size, d=1/fs) #array of frequencies corresponding to the bins
    idx = np.argmin(np.abs(freq - target_freq)) #finds index where frequency is closest to target_freq

    return X[idx], freq[idx] #returns the FFT value at the closest sampled frequency (freq[idx]) to the target frequency

#Creates a 32x32x4 grid of the FFT values at a target frequency for each impulse response for all heights
def create_FFT_grid(directory, fs, target_freq, heights):
    grid = np.zeros((32, 32, len(heights)), dtype=complex)

    for height_idx, height_val in enumerate(heights):
        dir = os.path.join(directory, f"h_{height_val}")
        if os.path.isdir(dir):
            for entry in os.scandir(dir):
                if entry.is_file() and entry.name.endswith('.mat'):
                    idxX = int(entry.name.split('_')[1])
                    idxY = int(entry.name.split('_')[-1].split('.')[0])

                    ir_data = scipy.io.loadmat(entry.path)['ImpulseResponse'].flatten()

                    #files indexes go from 1 to 32, grid has indexes 0-31 so we subtract 1
                    grid[idxX-1, idxY-1, height_idx], approximated_freq = compute_fft_at_target_freq(ir_data, fs, target_freq)

    return grid, approximated_freq

def magnitude_phase_plots(grid, approximated_freq, heights):
    magnitude = np.abs(grid)
    phase = np.angle(grid)
    
    room_dims = [0, 4.14, 0, 7.80]

    fig = plt.figure(figsize=(18, 14), layout="constrained")
    fig.suptitle(f'Room Response at {approximated_freq:.1f} Hz', fontsize=18)
    
    #this is to adapt the number of subplots to the number of heights
    a, b = divmod(len(heights), 2)
    subfigs = fig.subfigures(a + b, 2 if not b else 1)
    subfigs_flat = np.array(subfigs).flat

    for height_idx, height_val in enumerate(heights):
        sf = subfigs_flat[height_idx]
        
        sf.suptitle(f'Height {height_val} cm', fontsize=14)
        
        axs = sf.subplots(1, 2)
        
        mag_slice = magnitude[:, :, height_idx]
        phase_slice = phase[:, :, height_idx]

        # Magnitude
        ax_mag = axs[0]
        mag_plot = ax_mag.imshow(mag_slice, extent=room_dims, origin='lower', cmap='viridis', aspect='equal')
        ax_mag.set_title('Magnitude', fontsize=10)
        ax_mag.set_xlabel('X (m)')
        ax_mag.set_ylabel('Y (m)')
        sf.colorbar(mag_plot, ax=ax_mag, shrink=0.8)

        # Phase
        ax_phase = axs[1]
        phase_plot = ax_phase.imshow(phase_slice, extent=room_dims, origin='lower', cmap='twilight', aspect='equal')
        ax_phase.set_title('Phase', fontsize=10)
        ax_phase.set_xlabel('X (m)')
        ax_phase.set_ylabel('Y (m)')
        sf.colorbar(phase_plot, ax=ax_phase, shrink=0.8)

    plt.show()

if __name__ == "__main__":
    
    fs = 48000 #Isobel sampling frequency
    target_freq = 41 #Hz
    heights = [100, 130, 160, 190]

    directory = 'ISOBEL_SF_Dataset/Listening Room/ListeningRoom_SoundField_IRs/source_1/'
    
    grid, approximated_freq = create_FFT_grid(directory, fs, target_freq, heights)
    magnitude_phase_plots(grid, approximated_freq, heights)