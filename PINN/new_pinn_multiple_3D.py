"""
Predicts FFT of the impulse response at a specific (x,y,z) position for any room.
Uses all sources and all heights for multiple rooms as training data, modeled as a 3D problem.

Input:
    - normalized (x,y,z)
    - room dimensions (L, W, H)
    - source position (xs, ys, zs)

Output:
    - real + imaginary FFT at target frequency

Now uses ROOM-LEVEL SPLIT (no leakage)
"""

import os
import h5py
import torch
import numpy as np
import deepxde as dde

os.environ["DDE_BACKEND"] = "pytorch"

from data_split import create_pd, get_train_val_test_data

# -----------------------------
# Constants
# -----------------------------
file_path = r"PATH_TO_H5_FILE"

target_freq = 40  # Hz
c = 343.0
omega = 2 * np.pi * target_freq
k = omega / c

max_points_per_room = 2000  # attempt to fix memory issue

# -----------------------------
# Utility functions
# -----------------------------
def nmse_db(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred) ** 2)
    denom = np.sum(np.abs(y_true) ** 2)
    return 10 * np.log10(num / (denom + 1e-12) + 1e-12)


def filter_zero_targets(X, y, magnitude_threshold=1e-8):
    mask = np.abs(y).flatten() > magnitude_threshold
    return X[mask], y[mask], mask


def stack_complex_targets(y_complex):
    return np.hstack((np.real(y_complex), np.imag(y_complex))).astype(np.float32)


def validation_nmse_metric(y_true, y_pred):
    y_true_complex = y_true[:, 0] + 1j * y_true[:, 1]
    y_pred_complex = y_pred[:, 0] + 1j * y_pred[:, 1]
    return nmse_db(y_true_complex, y_pred_complex)


# -----------------------------
# Data extraction 
# -----------------------------
def extract_data_simulated(df, file_path):
    X_list = []
    y_list = []

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
            idx = np.random.choice(
                num_receivers,
                size=min(max_points_per_room, num_receivers),
                replace=False
            )

            receiver_pos = receiver_pos[idx]

            for s_idx, src in enumerate(source_pos):
                for r_idx, rec in enumerate(receiver_pos):

                    # Normalize spatial coords
                    xyz_norm = rec / room_dim

                    X_list.append([
                        xyz_norm[0], xyz_norm[1], xyz_norm[2],
                        room_dim[0], room_dim[1], room_dim[2],
                        src[0], src[1], src[2],
                    ])

                    y_list.append(real[s_idx, r_idx] + 1j * imag[s_idx, r_idx])

    return np.array(X_list, dtype=np.float32), np.array(y_list)


# -----------------------------
# Custom Validation PDE class
# -----------------------------
class ValidationPDE(dde.data.PDE):
    def __init__(self, *args, validation_x, validation_y, **kwargs):
        self.validation_x = validation_x.astype(np.float32)
        self.validation_y = validation_y.astype(np.float32)
        super().__init__(*args, **kwargs)

    def test(self):
        return self.validation_x, self.validation_y, None


# -----------------------------
# Helmholtz PDE
# -----------------------------
def pde(x, y):
    Lx, Ly, Lz = x[:, 3:4], x[:, 4:5], x[:, 5:6]
    y0, y1 = y[:, 0:1], y[:, 1:2]

    # second derivatives
    y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0) / (Lx ** 2)
    y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1) / (Ly ** 2)
    y0_zz = dde.grad.hessian(y, x, component=0, i=2, j=2) / (Lz ** 2)

    y1_xx = dde.grad.hessian(y, x, component=1, i=0, j=0) / (Lx ** 2)
    y1_yy = dde.grad.hessian(y, x, component=1, i=1, j=1) / (Ly ** 2)
    y1_zz = dde.grad.hessian(y, x, component=1, i=2, j=2) / (Lz ** 2)

    # physical coords
    abs_x = x[:, 0:1] * Lx
    abs_y = x[:, 1:2] * Ly
    abs_z = x[:, 2:3] * Lz

    xs, ys, zs = x[:, 6:7], x[:, 7:8], x[:, 8:9]

    sigma = 0.1
    dist = (abs_x - xs) ** 2 + (abs_y - ys) ** 2 + (abs_z - zs) ** 2
    f = (1 / ((sigma * np.sqrt(2 * np.pi)) ** 3)) * torch.exp(-0.5 * dist / sigma**2)

    return [
        -y0_xx - y0_yy - y0_zz - k**2 * y0 - f,
        -y1_xx - y1_yy - y1_zz - k**2 * y1
    ]


# -----------------------------
# LOAD + SPLIT 
# -----------------------------
df = create_pd(file_path)
train_df, val_df, test_df = get_train_val_test_data(df)

print(f"Rooms: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# Extract data per split
X_train, y_train = extract_data_simulated(train_df, file_path)
X_val, y_val = extract_data_simulated(val_df, file_path)
X_test, y_test = extract_data_simulated(test_df, file_path)

# Filter zeros
X_train, y_train, _ = filter_zero_targets(X_train, y_train)
X_val, y_val, _ = filter_zero_targets(X_val, y_val)
X_test, y_test, _ = filter_zero_targets(X_test, y_test)

print(f"Points: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# -----------------------------
# Prepare training
# -----------------------------
y_train_real = np.real(y_train).astype(np.float32)
y_train_imag = np.imag(y_train).astype(np.float32)

bc_real = dde.icbc.PointSetBC(X_train, y_train_real, component=0)
bc_imag = dde.icbc.PointSetBC(X_train, y_train_imag, component=1)

geom = dde.geometry.geometry_nd.Hypercube(xmin=[0]*9, xmax=[1]*9)

y_val_targets = stack_complex_targets(y_val)

data = ValidationPDE(
    geom,
    pde,
    [bc_real, bc_imag],
    num_domain=10000,
    anchors=X_train,
    validation_x=X_val,
    validation_y=y_val_targets,
)

net = dde.nn.FNN([9] + [50]*3 + [2], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile(
    "adam",
    lr=1e-3,
    loss="MSE",
    loss_weights=[1, 1, 100, 100],
    metrics=[validation_nmse_metric],
)

# -----------------------------
# TRAIN
# -----------------------------
losshistory, train_state = model.train(
    iterations=10000,
    display_every=1000,
    batch_size=128,
)

# -----------------------------
# TEST
# -----------------------------
y_pred = model.predict(X_test)

y_pred_complex = y_pred[:, 0] + 1j * y_pred[:, 1]
nmse = nmse_db(y_test, y_pred_complex)

print(f"Test NMSE: {nmse:.2f} dB")