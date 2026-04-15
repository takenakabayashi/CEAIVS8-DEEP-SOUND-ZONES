"""
create_FFT_grid function: 
    Creates a 3D grid of FFT values of each impulse response at a target frequency, for all specified heights
    The grid is used as input (by converting the grid indexes to x,y,z coordinates) and boundary conditions (FFT values) for the PINN model
magnitude_phase_plots function:
    Plots the magnitude and phase from the created FFT grid

Run this code to visualize the room response at target_freq = 41 and for all heights = [100, 130, 160, 190]
"""

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

#Creates a 32x32xlen(heights) grid of the FFT values at a target frequency for each impulse response for all specified heights
def create_FFT_grid(directory, fs, target_freq, heights):
    grid = np.zeros((32, 32, len(heights)), dtype=complex)

    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist")

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

        else:
            raise ValueError(f"Directory {dir} does not exist")
        
    return grid, approximated_freq

def magnitude_phase_plots(grid, approximated_freq, heights, block=True):
    magnitude = np.abs(grid)
    phase = np.angle(grid)
    
    room_dims = [0, 4.14, 0, 7.80]

    fig = plt.figure(figsize=(18, 14), layout="constrained")
    fig.suptitle(f'Room Response at {approximated_freq:.1f} Hz', fontsize=18)
    
    #this is to adapt the number of subplots to the number of heights
    n = len(heights)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    
    subfigs = fig.subfigures(nrows, ncols)
    subfigs_flat = np.array(subfigs).flatten()

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

    plt.show(block=block)

#X should be an array of [x,y,z,lx,ly,lz,sx,sy,sz] coordinates normalized between 0 and 1 (point coordinates, room dimensions and source coordinates)
def extract_data_ISOBEL():
    LR = {
        "directory": "ISOBEL_SF_Dataset/Listening Room/ListeningRoom_SoundField_IRs/",
        "sources_positions": [(0.17, 7.53, 1.0), (1.42, 2.08, 1.0)], #source_1 and source_2
        "room_dimensions": (4.14, 7.80, 2.78),
        "heights": [100, 130, 160, 190]
        }

    VR = {
        "directory": "ISOBEL_SF_Dataset/VR Lab/VRLab_SoundField_IRs/",
        "sources_positions": [(6.65, 7.93, 1.0), (5.23, 3.49, 1.0)], #source_1 and source_2
        "room_dimensions": (6.98, 8.12, 3.03),
        "heights": [100, 130, 160, 190]
        }
    
    PR = {
        "directory": "ISOBEL_SF_Dataset/Product Room/ProductRoom_SoundField_IRs/",
        "sources_positions": [(0.32, 0.22, 1.0), (4.48, 4.81, 1.0)], #source_1 and source_2
        "room_dimensions": (9.13, 12.03, 2.60),
        "heights": [130, 160, 190]
        }
    
    #Room B is missing sources position
    """ RB = {
        "directory": "ISOBEL_SF_Dataset/Room B/RoomB_SoundField_IRs/",
        "sources_positions": [(), ()], #source_1 and source_2
        "room_dimensions": (4.16, 6.46, 2.30),
        "heights": [100] 
    } """

    rooms = [LR, VR, PR] #check sources z coordinate

    fs = 48000 #ISOBEL dataset sampling frequency
    target_freq = 41 #Hz

    X_tot = []
    y_tot = []

    for room in rooms:
        dir = room["directory"]
        heights = room["heights"]

        if not os.path.isdir(dir):
            raise ValueError(f"Directory {dir} does not exist")
        
        for source_idx, source_pos in enumerate(room["sources_positions"]):
            source_path = os.path.join(dir, f"source_{source_idx+1}")
            if not os.path.isdir(source_path):
                raise ValueError(f"Directory {source_path} does not exist")
        
            l_x, l_y, l_z = room["room_dimensions"] #in meters

            x_vals = np.linspace(0, 1, 32)
            y_vals = np.linspace(0, 1, 32)
            z_vals = np.array(heights) / (100.0 * l_z)

            X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = Z.flatten()

            n_points = len(X_flat)

            x_dim = np.full(n_points, l_x)
            y_dim = np.full(n_points, l_y)
            z_dim = np.full(n_points, l_z)

            sx = np.full(n_points, source_pos[0])
            sy = np.full(n_points, source_pos[1])
            sz = np.full(n_points, source_pos[2])

            X_coords = np.vstack((X_flat, Y_flat, Z_flat, x_dim, y_dim, z_dim, sx, sy, sz)).T.astype(np.float32)
            
            grid, approximated_freq = create_FFT_grid(source_path, fs, target_freq, heights)

            y_target = grid.flatten().reshape(-1, 1).astype(np.complex64)

            X_tot.append(X_coords)
            y_tot.append(y_target)
        
    X_tot = np.concatenate(X_tot, axis=0)
    y_tot = np.concatenate(y_tot, axis=0)
    
    return X_tot, y_tot


if __name__ == "__main__":
    
    fs = 48000 #Isobel sampling frequency
    target_freq = 41
    heights = [100, 130, 160, 190]

    #X, y = extract_data_ISOBEL()

    directory = 'ISOBEL_SF_Dataset/Listening Room/ListeningRoom_SoundField_IRs/source_1/'
    
    grid, approximated_freq = create_FFT_grid(directory, fs, target_freq, heights)
    magnitude_phase_plots(grid, approximated_freq, heights)