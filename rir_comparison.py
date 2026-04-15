"""
Compares simulated RIRs with measured ISOBEL RIRs at a specified frequency and heights
"""
import numpy as np

from PINN.utils import create_FFT_grid, magnitude_phase_plots

def nmse_db(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred)**2)
    denom = np.sum(np.abs(y_true)**2)
    
    eps = 1e-12 #avoids division by 0
    nmse = num / (denom + eps)
    
    nmse_db = 10 * np.log10(nmse + eps) #dB conversion
    
    return nmse_db

if __name__ == "__main__":
    dir_real = 'ISOBEL_SF_Dataset/VR Lab/VRLab_SoundField_IRs/source_1/'
    grid_real, freq_real = create_FFT_grid(dir_real, fs=48000, target_freq=40, heights=[100])

    dir_sim = 'individual_RIRs/source_1/'
    grid_sim, freq_sim = create_FFT_grid(dir_sim, fs=48000, target_freq=40, heights=[100])

    magnitude_phase_plots(grid_real, freq_real, [100], block=False)
    magnitude_phase_plots(grid_sim, freq_sim, [100])

    nmse_db = nmse_db(grid_real, grid_sim)
    print(f"Simulation Accuracy (NMSE): {nmse_db:.2f} dB")