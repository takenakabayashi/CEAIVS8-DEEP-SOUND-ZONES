""" 
Predicts FFT of the impulse responses along a 32x32 grid for any room.
Uses all sources and all heights for multiple rooms as training data, modeled as a 3D problem.
It takes normalized x,y,z coordinates, room dimensions and source position as input, output is the predicted real and imaginary part of the FFT at the specified target frequency.
TODO: add Robin boundary conditions, look into activation function, number of iterations and loss weights, normalize by converting to dB
"""
import os
from sklearn.model_selection import train_test_split
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np

from data_extraction import extract_data

target_freq = 40 #Hz
c = 343.0 #m/s
omega = 2 * np.pi * target_freq
k = omega / c

def nmse_db(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred)**2)
    denom = np.sum(np.abs(y_true)**2)
    
    eps = 1e-12 #avoids division by 0
    nmse = num / (denom + eps)
    
    nmse_db = 10 * np.log10(nmse + eps) #dB conversion
    
    return nmse_db

#filter out points with pressure (approximately) equal to zero
def filter_zero_targets(X, y, magnitude_threshold=1e-8):
    target_magnitude = np.abs(y).flatten()
    valid_mask = target_magnitude > magnitude_threshold

    return X[valid_mask], y[valid_mask], valid_mask

def stack_complex_targets(y_complex):
    return np.hstack(
        (
            np.real(y_complex).astype(np.float32),
            np.imag(y_complex).astype(np.float32),
        )
    )

def validation_nmse_metric(y_true, y_pred):
    y_true_complex = y_true[:, 0] + 1j * y_true[:, 1]
    y_pred_complex = y_pred[:, 0] + 1j * y_pred[:, 1]
    return nmse_db(y_true_complex, y_pred_complex)

#dde.data.PDE wrapper, for printing statements
class ValidationPDE(dde.data.PDE):
    def __init__(self, *args, validation_x, validation_y, **kwargs):
        self.validation_x = validation_x.astype(np.float32)
        self.validation_y = validation_y.astype(np.float32)
        super().__init__(*args, **kwargs)

    def test(self):
        self.test_x = self.validation_x
        self.test_y = self.validation_y
        self.test_aux_vars = None
        return self.test_x, self.test_y, self.test_aux_vars

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if not isinstance(loss_fn, (list, tuple)):
            loss_fn = [loss_fn] * 4
        else:
            loss_fn = list(loss_fn)

        pde_real_loss_fn, pde_imag_loss_fn, real_loss_fn, imag_loss_fn = loss_fn

        pde_residual_real, pde_residual_imag = self.pde(inputs, outputs)
        pde_loss_real = pde_real_loss_fn(torch.zeros_like(pde_residual_real), pde_residual_real)
        pde_loss_imag = pde_imag_loss_fn(torch.zeros_like(pde_residual_imag), pde_residual_imag)
        loss_real = real_loss_fn(targets[:, 0:1], outputs[:, 0:1])
        loss_imag = imag_loss_fn(targets[:, 1:2], outputs[:, 1:2])
        displayed_loss_real = loss_real
        displayed_loss_imag = loss_imag

        return [pde_loss_real, pde_loss_imag, displayed_loss_real, displayed_loss_imag]
    
#Helmholtz PDE implementation from https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.sound.hard.abc.html#helmholtz-sound-hard-scattering-problem-with-absorbing-boundary-conditions
#inhomogeneous helmholtz equation because the source is inside the domain
def pde(x, y):  #here x is the input (x and y coordinates) of the model and y the output (pressure)
    Lx = x[:, 3:4]
    Ly = x[:, 4:5]
    Lz = x[:, 5:6]

    y0, y1 = y[:, 0:1], y[:, 1:2] #y0 is the real part of the pressure, y1 the imaginary part

    # divide by dimension^2 because of previous normalization (chain rule)
    y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0) / (Lx ** 2)
    y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1) / (Ly ** 2)
    y0_zz = dde.grad.hessian(y, x, component=0, i=2, j=2) / (Lz ** 2)

    y1_xx = dde.grad.hessian(y, x, component=1, i=0, j=0) / (Lx ** 2)
    y1_yy = dde.grad.hessian(y, x, component=1, i=1, j=1) / (Ly ** 2)
    y1_zz = dde.grad.hessian(y, x, component=1, i=2, j=2) / (Lz ** 2)

    # physical source distance in meters
    abs_x = x[:, 0:1] * Lx
    abs_y = x[:, 1:2] * Ly
    abs_z = x[:, 2:3] * Lz

    #f = delta(x-xs) models a point source at location xs, source: https://arxiv.org/pdf/1712.06091
    xs, ys, zs = x[:, 6:7], x[:, 7:8], x[:, 8:9] #source coordinates in meters
    sigma = 0.1
    dist = (abs_x - xs)**2 + (abs_y - ys)**2 + (abs_z - zs)**2
    f = (1 / ((sigma * np.sqrt(2 * np.pi)) ** 3)) * torch.exp(-0.5 * dist / sigma**2) 

    return [-y0_xx - y0_yy - y0_zz - k ** 2 * y0 - f,
            -y1_xx - y1_yy - y1_zz - k ** 2 * y1]

X, y = extract_data()
X, y, valid_mask = filter_zero_targets(X, y)

removed_points = int((~valid_mask).sum())
print(f"Filtered out {removed_points} points")

# Split train/test
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42
)

validation_fraction = 0.5

#Split again to get test and validation data
X_val, X_test, y_val, y_test = train_test_split(
    X_holdout,
    y_holdout,
    test_size=1 - validation_fraction,
    random_state=42,
)

y_train_real = np.real(y_train).astype(np.float32)
y_train_imag = np.imag(y_train).astype(np.float32)

print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

#Add observed (Isobel) data as boundary conditions
#https://github.com/lululxvi/deepxde/issues/1952#issuecomment-2724018030
bc_data_real = dde.icbc.PointSetBC(X_train, y_train_real, component=0)
bc_data_imag = dde.icbc.PointSetBC(X_train, y_train_imag, component=1)

#https://github.com/lululxvi/deepxde/issues/1762#issuecomment-2158327633
geom = dde.geometry.geometry_nd.Hypercube(xmin=[0] * 9, xmax=[1] * 9)

y_val_targets = stack_complex_targets(y_val)

data = ValidationPDE(
    geom,
    pde,
    [bc_data_real, bc_data_imag],
    num_domain=10000,
    num_boundary=0,
    anchors=X_train,
    validation_x=X_val,
    validation_y=y_val_targets,
)

net = dde.nn.FNN([9] + [50] * 3 + [2], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile(
    "adam", 
    lr=1e-3, 
    loss="MSE", 
    loss_weights=[1, 1, 100, 100],
    metrics=[validation_nmse_metric],
    )

losshistory, train_state = model.train(
    iterations=10000,
    display_every=1000,
)

#Model evaluation
y_pred = model.predict(X_test)
y_pred_real, y_pred_imag = y_pred[:, 0], y_pred[:, 1]

y_test_real = np.real(y_test).flatten()
y_test_imag = np.imag(y_test).flatten()

test_mse_real = np.mean((y_test_real - y_pred_real)**2)
test_mse_imag = np.mean((y_test_imag - y_pred_imag)**2)
print(f"Test MSE real: {test_mse_real:.6f}")
print(f"Test MSE imaginary: {test_mse_imag:.6f}")

#NMSE
y_pred = y_pred_real + 1j * y_pred_imag
y_test = y_test_real + 1j * y_test_imag

nmse = nmse_db(y_test, y_pred)
print(f"Test NMSE: {nmse:.2f} dB")