""" 
Simple test implementation, can be deleted later
It was used at the beginning for testing DeepXDE on a simple 2D problem before moving to the actual 3D problem
Uses only source 1 and height 100 for the listening room, modeled as a 2D problem
It takes x,y coordinates as input and outputs the predicted real and imaginary part of the FFT at the specified target frequency
"""
import os
from sklearn.model_selection import train_test_split
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np

from data_extraction import create_FFT_grid

target_freq = 40 #Hz
c = 343.0 #m/s
omega = 2 * np.pi * target_freq
k = omega / c

#Helmholtz PDE implementation from https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.sound.hard.abc.html#helmholtz-sound-hard-scattering-problem-with-absorbing-boundary-conditions
#inhomogeneous helmholtz equation because the source is inside the domain
def pde(x, y): #here x is the input (x and y coordinates) of the model and y the output (pressure)
    y0, y1 = y[:, 0:1], y[:, 1:2] #y0 is the real part of the pressure, y1 the imaginary part

    y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

    y1_xx = dde.grad.hessian(y, x,component=1, i=0, j=0)
    y1_yy = dde.grad.hessian(y, x,component=1, i=1, j=1)

    #f = delta(x-xs) models a point source at location xs, source: https://arxiv.org/pdf/1712.06091
    xs, ys = 0.17, 7.53
    sigma = 0.1
    dist = (x[:, 0:1] - xs)**2 + (x[:, 1:2] - ys)**2
    f = (1 / (sigma * np.sqrt(2 * np.pi))) * torch.exp(-0.5 * dist / sigma**2) #dirac delta approximation with gaussian with mean 0 and small std sigma

    return [-y0_xx - y0_yy - k ** 2 * y0 - f,
            -y1_xx - y1_yy - k ** 2 * y1] #dirac delta has no imaginary part

fs = 48000 #ISOBEL dataset sampling frequency
directory = 'ISOBEL_SF_Dataset/Listening Room/ListeningRoom_SoundField_IRs/source_1/'

grid, approximated_freq = create_FFT_grid(directory, fs, target_freq, heights=[100])

#The input array X_train needs to be an array of (x,y) real-life coordinates in meters
#We need to convert grid indexes into meters using the room dims, and then reshape them into (x,y) coords
l_x, l_y = 4.14, 7.80 #room dimensions in meters
n_x, n_y, _ = grid.shape #32x32 grid

x_vals = np.linspace(0, l_x, n_x)
y_vals = np.linspace(0, l_y, n_y)

X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
X = np.vstack((X.flatten(), Y.flatten())).T.astype(np.float32)

y = grid.flatten().reshape(-1, 1).astype(np.complex64)

#Split train/test into 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

y_train_real = np.real(y_train).astype(np.float32)
y_train_imag = np.imag(y_train).astype(np.float32)

#Add observed (Isobel) data as boundary conditions
#https://github.com/lululxvi/deepxde/issues/1952#issuecomment-2724018030
bc_data_real = dde.icbc.PointSetBC(X_train, y_train_real, component=0)
bc_data_imag = dde.icbc.PointSetBC(X_train, y_train_imag, component=1)

geom = dde.geometry.Rectangle([0, 0], [l_x, l_y])

data = dde.data.PDE(
    geom,
    pde,
    [bc_data_real, bc_data_imag],
    num_domain=5000,
    num_boundary=0,
    anchors=X_train
)

net = dde.nn.FNN([2] + [50] * 3 + [2], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss="MSE", loss_weights=[1, 1, 100, 100]) #loss_weights = PDE real, PDE imaginary, data real, data imaginary
losshistory, train_state = model.train(iterations=5000)

#Model evaluation
y_pred = model.predict(X_test)
y_pred_real, y_pred_imag = y_pred[:, 0], y_pred[:, 1]

y_test_real = np.real(y_test).flatten()
y_test_imag = np.imag(y_test).flatten()

test_mse_real = np.mean((y_test_real - y_pred_real)**2)
test_mse_imag = np.mean((y_test_imag - y_pred_imag)**2)
print(f"Test MSE real: {test_mse_real:.6f}")
print(f"Test MSE imaginary: {test_mse_imag:.6f}")