""" 
Predicts FFT of the impulse response at a specific (x,y,z) position for any room.
Uses all sources and all heights for multiple rooms as training data, modeled as a 3D problem.
It takes normalized x,y,z coordinates, room dimensions and source position as input, output is the predicted real and imaginary part of the FFT at the specified target frequency.
TODO: add room absorption parameter, add Robin boundary conditions, look into activation function, number of iterations and loss weights
"""
import os
from matplotlib import pyplot as plt
import torch

from config import SIMULATED_DATA_FILE
from data_split import get_train_val_test_data

os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np

from data_extraction import extract_data_ISOBEL, extract_data_simulated, get_max_min_room_dims
from utils import filter_zero_targets, nmse_db, stack_complex_targets, validation_nmse_metric

TARGET_FREQ = 41 #Hz

val_fraction = 0.5

c = 343.0 #m/s
omega = 2 * np.pi * TARGET_FREQ
k = omega / c

L_min, L_max = get_max_min_room_dims(file_path=SIMULATED_DATA_FILE)
L_min = torch.tensor(L_min)
L_max = torch.tensor(L_max)

#dde.data.PDE wrapper
#The only thing this does is change the printing statements during training to print validation loss and test metric
#Prints every display_every=1000 iterations:
#Step | Train loss | Test loss (says test but it's actually the validation loss) | Test metric (NMSE in dB, same metric used in the ISOBEL paper)
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

        #print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        return [pde_loss_real, pde_loss_imag, displayed_loss_real, displayed_loss_imag]
    
#Helmholtz PDE implementation from https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.sound.hard.abc.html#helmholtz-sound-hard-scattering-problem-with-absorbing-boundary-conditions
#inhomogeneous helmholtz equation because the source is inside the domain
def pde(x, y):  #here x is the input (x and y coordinates) of the model and y the output (pressure)
    y0, y1 = y[:, 0:1], y[:, 1:2] #y0 is the real part of the pressure, y1 the imaginary part
    
    Lx = x[:, 3:4] * (L_max[0] - L_min[0]) + L_min[0]
    Ly = x[:, 4:5] * (L_max[1] - L_min[1]) + L_min[1]
    Lz = x[:, 5:6] * (L_max[2] - L_min[2]) + L_min[2]

    #Divide by dimension^2 because of previous normalization of coordinates (chain rule)
    y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0) / (Lx ** 2)
    y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1) / (Ly ** 2)
    y0_zz = dde.grad.hessian(y, x, component=0, i=2, j=2) / (Lz ** 2)

    y1_xx = dde.grad.hessian(y, x, component=1, i=0, j=0) / (Lx ** 2)
    y1_yy = dde.grad.hessian(y, x, component=1, i=1, j=1) / (Ly ** 2)
    y1_zz = dde.grad.hessian(y, x, component=1, i=2, j=2) / (Lz ** 2)

    # Point coordinates in meters
    pos_x = x[:, 0:1] * Lx
    pos_y = x[:, 1:2] * Ly
    pos_z = x[:, 2:3] * Lz

    # Source coordinates in meters
    pos_xs = x[:, 6:7] * Lx
    pos_ys = x[:, 7:8] * Ly
    pos_zs = x[:, 8:9] * Lz

    #f = delta(x-xs) models a point source at location xs, source: https://arxiv.org/pdf/1712.06091
    sigma = 0.1
    dist = (pos_x - pos_xs)**2 + (pos_y - pos_ys)**2 + (pos_z - pos_zs)**2
    f = (1 / ((sigma * np.sqrt(2 * np.pi)) ** 3)) * torch.exp(-0.5 * dist / sigma**2) 

    return [-y0_xx - y0_yy - y0_zz - k ** 2 * y0 - f,
            -y1_xx - y1_yy - y1_zz - k ** 2 * y1]

train_df, val_df, test_df = get_train_val_test_data(file_path=SIMULATED_DATA_FILE)

# Extract data per split
X_train, y_train = extract_data_simulated(df=train_df, target_freq=TARGET_FREQ, file_path=SIMULATED_DATA_FILE, max_points_per_room=1000)
X_val, y_val = extract_data_simulated(df=val_df, target_freq=TARGET_FREQ, file_path=SIMULATED_DATA_FILE, max_points_per_room=1000)
X_test, y_test = extract_data_simulated(df=test_df, target_freq=TARGET_FREQ, file_path=SIMULATED_DATA_FILE, max_points_per_room=1000)

# Filter zeros
X_train, y_train, mask_train = filter_zero_targets(X_train, y_train)
X_val, y_val, mask_val = filter_zero_targets(X_val, y_val)
X_test, y_test, mask_test = filter_zero_targets(X_test, y_test)

print(f"Filtered out {int((~mask_train).sum())} train points | {int((~mask_val).sum())} val points | {int((~mask_test).sum())} test points")

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
    num_domain=1000,
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
    loss_weights=[1, 1, 100, 100], #pde real, pde imag, data real, data imag
    metrics=[validation_nmse_metric],
    )

losshistory, train_state = model.train(
    iterations=10000,
    display_every=1000,
    batch_size=128,
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

dde.utils.external.save_loss_history(losshistory, "loss.dat")
dde.utils.plot_loss_history(losshistory)
plt.show()