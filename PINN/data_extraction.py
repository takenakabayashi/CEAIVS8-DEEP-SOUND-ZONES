import os
import scipy
import numpy as np
import matplotlib.pyplot as plt

def compute_fft_at_target_freq(x, fs, target_freq):
    X = np.fft.rfft(x)
    freq = np.fft.rfftfreq(x.size, d=1/fs)

    idx = np.argmin(np.abs(freq - target_freq)) #finds index where frequency is closest to target_freq

    return X[idx], freq[idx] #returns the FFT value at the closest sampled frequency (freq[idx]) to the target frequency

if __name__ == "__main__":
    
    fs = 44100 #Isobel sampling frequency
    target_freq = 1000
    approximated_freq = 0

    """ data = scipy.io.loadmat('PINN/idxX_2_idxY_2.mat')
    print(data.keys())
    x = data['ImpulseResponse']

    fft_at_target, actual_freq = compute_fft_at_target_freq(x, fs, target_freq)
    print(type(fft_at_target[0]))
    
    print(f"FFT at {actual_freq:.1f}Hz: {fft_at_target}") """

    directory = 'ISOBEL_SF_Dataset/VR Lab/VRLab_SoundField_IRs/source_1/h_100/'
    grid = np.zeros((30, 30), dtype=complex)

    for entry in os.scandir(directory):
        if entry.is_file():
            idxX = int(entry.name.split('_')[1])
            idxY = int(entry.name.split('_')[-1].split('.')[0])
            #print(f"idxX: {idxX}, idxY: {idxY}")

            ir_data = scipy.io.loadmat(entry.path)['ImpulseResponse'].flatten()

            #indexes in the file names go from 2 to 31, but as the grid uses indexes from 0 to 29 we subtract 2
            grid[idxX-2, idxY-2], approximated_freq = compute_fft_at_target_freq(ir_data, fs, target_freq)
    

    magnitude = np.abs(grid) #need to convert in dB
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