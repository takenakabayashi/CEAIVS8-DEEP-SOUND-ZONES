import os
import h5py
import numpy as np

#from https://gist.github.com/jbohnslav/92ea022c06356b880e4d60ab978eed27
def print_hdf5(h5py_obj, level=-1, print_full_name: bool = True, print_attrs: bool = True) -> None:
    """ Prints the name and shape of datasets in a H5py HDF5 file.
    Parameters
    ----------
    h5py_obj: [h5py.File, h5py.Group]
        the h5py.File or h5py.Group object
    level: int
        What level of the file tree you are in
    print_full_name
        If True, the full tree will be printed as the name, e.g. /group0/group1/group2/dataset: ...
        If False, only the current node will be printed, e.g. dataset:
    print_attrs
        If True: print all attributes in the file
    Returns
    -------
    None
    """
    def is_group(f):
        return type(f) == h5py._hl.group.Group

    def is_dataset(f):
        return type(f) == h5py._hl.dataset.Dataset

    def print_level(level, n_spaces=5) -> str:
        if level == -1:
            return ''
        prepend = '|' + ' ' * (n_spaces - 1)
        prepend *= level
        tree = '|' + '-' * (n_spaces - 2) + ' '
        return prepend + tree

    for key in h5py_obj.keys():
        entry = h5py_obj[key]
        name = entry.name if print_full_name else os.path.basename(entry.name)
        if is_group(entry):
            print('{}{}'.format(print_level(level), name))
            print_hdf5(entry, level + 1, print_full_name=print_full_name)
        elif is_dataset(entry):
            shape = entry.shape
            dtype = entry.dtype
            print('{}{}: {} {}'.format(print_level(level), name,
                                       shape, dtype))
    if level == -1:
        if print_attrs:
            print('attrs: ')
            for key, value in h5py_obj.attrs.items():
                print(' {}: {}'.format(key, value))

def extract_data_simulated(file_path='simulatedData.h5', target_freq=40): #add absorption coeff as input?
    X_list = []
    y_list = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    with h5py.File(file_path, 'r') as f:
        #Gets frequency index with value nearest to target_freq
        first_room = list(f.keys())[0]
        freqs = f[first_room]['freqs'][0]  # Shape (K,)
        freq_idx = np.abs(freqs - target_freq).argmin()
        
        print(f"Extracting data for {freqs[freq_idx]:.2f} Hz")

        #Iterates through each room
        #for room_name in f.keys(): #use for room_name in list(f.keys())[:500] for a subset
        for room_name in list(f.keys())[:2]:
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

if __name__ == "__main__":

    f = h5py.File('simulatedData.h5', 'r')
    #print(list(f.keys()))

    print_hdf5(f)

    with h5py.File("simulatedData.h5", "r") as f:
        for room_key in f.keys():
            room = f[room_key]

            #print(room_key, room["RTF_real"].shape)

            rtf_real = room["RTF_real"][:] #shape (40, 8, 510)
            rtf_imag = room["RTF_imag"][:] #(40, 8, 510)
            receiver_pos = room["receiver_pos"][:] #(510, 3)
            source_pos = room["source_pos"][:] #(8, 3)
            freqs = room["freqs"][0] #(40,)
            room_dim = room["room_dim"][0] #(3,)
            t60 = room["T60"][0, 0] #scalar
            alpha = room["alpha"][0, 0] #scalar

            rtf_complex = rtf_real + 1j * rtf_imag

            print(room.keys())
            print(receiver_pos[0])
            print(source_pos[0])
            print(rtf_real[10, 2, 100]) #freq bin 10, source 2, receiver 100
