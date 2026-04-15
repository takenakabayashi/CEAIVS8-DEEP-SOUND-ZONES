import os
import h5py

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