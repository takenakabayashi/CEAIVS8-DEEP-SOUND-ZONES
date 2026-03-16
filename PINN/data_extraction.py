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

#Creates a 32x32 grid of the FFT values at a target frequency for each impulse response in a directory
def create_FFT_grid(directory, fs, target_freq):
    grid = np.zeros((32, 32), dtype=complex)

    for entry in os.scandir(directory):
        if entry.is_file():
            idxX = int(entry.name.split('_')[1])
            idxY = int(entry.name.split('_')[-1].split('.')[0])

            ir_data = scipy.io.loadmat(entry.path)['ImpulseResponse'].flatten()

            #files indexes go from 1 to 32, grid has indexes 0-31 so we subtract 1
            grid[idxX-1, idxY-1], approximated_freq = compute_fft_at_target_freq(ir_data, fs, target_freq)

    return grid, approximated_freq

def magnitude_phase_plots(grid, approximated_freq):
    magnitude = np.abs(grid) #TODO: convert in dB
    phase = np.angle(grid)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    #Magnitude plot
    magnitude_plot = ax[0].imshow(magnitude, extent=[2, 31, 2, 31], origin='lower', cmap='viridis')
    ax[0].set_title(f'Magnitude at {approximated_freq:.1f} Hz')
    fig.colorbar(magnitude_plot, ax=ax[0], label='Amplitude')

    #Phase plot
    phase_plot = ax[1].imshow(phase, extent=[2, 31, 2, 31], origin='lower', cmap='twilight')
    ax[1].set_title(f'Phase at {approximated_freq:.1f} Hz')
    fig.colorbar(phase_plot, ax=ax[1], label='Degrees')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    fs = 48000 #Isobel sampling frequency
    target_freq = 41

    directory = 'ISOBEL_SF_Dataset/Listening Room/ListeningRoom_SoundField_IRs/source_1/h_100/'
    
    grid, approximated_freq = create_FFT_grid(directory, fs, target_freq)
    magnitude_phase_plots(grid, approximated_freq)