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

def get_max_min_room_dims(file_path=SIMULATED_DATA_FILE):
    Lx_min, Ly_min, Lz_min = float('inf'), float('inf'), float('inf')
    Lx_max, Ly_max, Lz_max = float('-inf'), float('-inf'), float('-inf')

    with h5py.File(file_path, "r") as f:
        for room in f.keys():
            Lx, Ly, Lz = f[room]["room_dim"][0]

            Lx_min, Ly_min, Lz_min = min(Lx_min, Lx), min(Ly_min, Ly), min(Lz_min, Lz)
            Lx_max, Ly_max, Lz_max = max(Lx_max, Lx), max(Ly_max, Ly), max(Lz_max, Lz)

    return [Lx_min, Ly_min, Lz_min], [Lx_max, Ly_max, Lz_max]

#Same as extract_data_ISOBEL but for the simulated data (stored in a h5 file)
#X is an array [x,y,z,sx,sy,sz,lx,ly,lz] (point coordinates, source coordinates, room dimensions)
#x,y,z coordinates are normalized between 0 and 1
def extract_data_simulated(df, target_freq, file_path=SIMULATED_DATA_FILE, max_points_per_room=None): #add absorption coeff as input?
    X_list = []
    y_list = []

    min_room_dims, max_room_dims = get_max_min_room_dims(file_path)
    min_room_dims = np.array(min_room_dims)
    max_room_dims = np.array(max_room_dims)

    with h5py.File(file_path, "r") as f:
        for _, row in df.iterrows():
            grp = f[row["room"]]

            RTF_real = grp["RTF_real"][:]  # (F, S, R)
            RTF_imag = grp["RTF_imag"][:]
            receiver_pos = grp["receiver_pos"][:]  # (R, 3)
            source_pos = grp["source_pos"][:]      # (S, 3)
            room_dim = grp["room_dim"][0]
            freqs = grp["freqs"][0]

            # Select frequency
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            real = RTF_real[freq_idx]
            imag = RTF_imag[freq_idx]

            # Subsample receivers (IMPORTANT)
            num_receivers = receiver_pos.shape[0]
            sample_size = num_receivers if max_points_per_room is None else min(max_points_per_room, num_receivers)
            idx = np.random.choice(
                num_receivers,
                size=sample_size,
                replace=False
            )

            receiver_pos = receiver_pos[idx]

            for s_idx, src in enumerate(source_pos):
                for rec, r_idx in zip(receiver_pos, idx):

                    # Normalize spatial coords
                    xyz_norm = rec / room_dim 
                    src_norm = src / room_dim
                    room_norm = (room_dim - min_room_dims) / (max_room_dims - min_room_dims)

                    X_list.append([
                        xyz_norm[0], xyz_norm[1], xyz_norm[2],
                        room_norm[0], room_norm[1], room_norm[2],
                        src_norm[0], src_norm[1], src_norm[2],
                    ])

                    y_list.append(real[s_idx, r_idx] + 1j * imag[s_idx, r_idx])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.complex64).reshape(-1, 1)
    
    return X, y