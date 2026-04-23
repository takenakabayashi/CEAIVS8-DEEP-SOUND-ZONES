import os
import h5py
import numpy as np

from utils import create_FFT_grid
from config import ISOBEL_FS, ISOBEL_ROOMS, SIMULATED_DATA_FILE

def extract_grid(room, source, fs, target_freq):
    grid, approximated_freq = create_FFT_grid(room, source, fs, target_freq)

    #The input array X needs to be an array of (x,y,z) real-life coordinates in meters
    #We need to convert grid indexes into meters using the room dims, and then reshape them into (x,y,z) coords
    l_x, l_y, l_z = room["room_dimensions"] #room dimensions in meters
    n_x, n_y, n_z = grid.shape

    #TODO: fix the spacing, explained in the comment in create_FFT_grid function in utils.py
    x_vals = np.linspace(0, l_x, n_x)
    y_vals = np.linspace(0, l_y, n_y)
    z_vals = np.linspace(0, l_z, n_z)

    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    X = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T.astype(np.float32)

    y = grid.flatten().reshape(-1, 1).astype(np.complex64)

    return X, y

#X is an array [x,y,z,sx,sy,sz,lx,ly,lz] (point coordinates, source coordinates, room dimensions)
#x,y,z coordinates are normalized between 0 and 1
def extract_data_ISOBEL(target_freq, fs=ISOBEL_FS, rooms = list(ISOBEL_ROOMS.values())):
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

#Same as extract_data_ISOBEL but for the simulated data (stored in a h5 file)
#X is an array [x,y,z,sx,sy,sz,lx,ly,lz] (point coordinates, source coordinates, room dimensions)
#x,y,z coordinates are normalized between 0 and 1
def extract_data_simulated(target_freq, file_path=SIMULATED_DATA_FILE, subset=None): #add absorption coeff as input?
    X_list = []
    y_list = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    with h5py.File(file_path, 'r') as f:
        #Gets frequency index with value nearest to target_freq
        first_room = list(f.keys())[0]
        freqs = f[first_room]['freqs'][0]  #shape (K,)
        freq_idx = np.abs(freqs - target_freq).argmin()
        
        print(f"Extracting data for {freqs[freq_idx]:.2f} Hz")

        #Iterates through each room
        for room_name in list(f.keys())[:subset] if subset is not None else list(f.keys()):
            group = f[room_name]
            
            #Room constants
            room_dim = group['room_dim'][0]
            Lx, Ly, Lz = room_dim
            
            source_pos = group['source_pos'][:] #(8, 3)
            receiver_pos = group['receiver_pos'][:] #(n_receivers, 3)
            
            #Loads RTFs for the specific frequency
            #(K, 8, n_receivers) -> (8, n_receivers)
            rtf_real = group['RTF_real'][freq_idx, :, :]
            rtf_imag = group['RTF_imag'][freq_idx, :, :]
            
            n_receivers = receiver_pos.shape[0]

            #Iterate through sources
            for s_idx in range(8):
                xs, ys, zs = source_pos[s_idx]
                
                #Normalized receiver coords (0-1)
                x_norm = receiver_pos[:, 0] / Lx
                y_norm = receiver_pos[:, 1] / Ly
                z_norm = receiver_pos[:, 2] / Lz
                
                #[x, y, z, Lx, Ly, Lz, xs, ys, zs]
                X_src = np.zeros((n_receivers, 9))
                X_src[:, 0] = x_norm
                X_src[:, 1] = y_norm
                X_src[:, 2] = z_norm
                X_src[:, 3] = Lx
                X_src[:, 4] = Ly
                X_src[:, 5] = Lz
                X_src[:, 6] = xs
                X_src[:, 7] = ys
                X_src[:, 8] = zs
                
                #Targets
                y_complex = rtf_real[s_idx, :] + 1j * rtf_imag[s_idx, :]
                
                X_list.append(X_src)
                y_list.append(y_complex)

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.complex64).reshape(-1, 1)

    return X, y