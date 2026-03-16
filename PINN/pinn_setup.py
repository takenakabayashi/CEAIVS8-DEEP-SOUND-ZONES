""" 
Current implementation uses only source 1 and height 100, modeled as a 2D problem
It takes x,y coordinates as input and outputs the predicted real and imaginary part of the FFT at the specified target frequency
TODO: split data into test/train, add Robin boundary conditions, fix loss functions and weights, look into activation function and number of iterations
"""
import os
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np

from data_extraction import create_FFT_grid

target_freq = 40 #Hz
c = 343.0 #m/s
omega = 2 * np.pi * target_freq
k = omega / c

def pde(x, y):
    y0, y1 = y[:, 0:1], y[:, 1:2]

    y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

    y1_xx = dde.grad.hessian(y, x,component=1, i=0, j=0)
    y1_yy = dde.grad.hessian(y, x,component=1, i=1, j=1)

    return [-y0_xx - y0_yy - k ** 2 * y0,
            -y1_xx - y1_yy - k ** 2 * y1]

fs = 48000 #ISOBEL dataset sampling frequency
directory = 'ISOBEL_SF_Dataset/Listening Room/ListeningRoom_SoundField_IRs/source_1/h_100/'

grid, approximated_freq = create_FFT_grid(directory, fs, target_freq)

#The input array X_train needs to be an array of (x,y) real-life coordinates in meters
#We need to convert grid indexes into meters using the room dims, and then reshape them into (x,y) coords
l_x, l_y = 7.11, 5.43 #room dimensions in meters
n_x, n_y = grid.shape #32x32 grid

x_vals = np.linspace(0, l_x, n_x)
y_vals = np.linspace(0, l_y, n_y)

X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
X_train = np.vstack((X.flatten(), Y.flatten())).T.astype(np.float32)

#Reshapes grid values to get real and imaginary parts into two different arrays
y_train_real = np.real(grid).flatten().reshape(-1, 1).astype(np.float32)
y_train_imag = np.imag(grid).flatten().reshape(-1, 1).astype(np.float32)

#Adds observed (Isobel) data as boundary conditions
#https://github.com/lululxvi/deepxde/issues/1952#issuecomment-2724018030
bc_data_real = dde.icbc.PointSetBC(X_train, y_train_real, component=0)
bc_data_imag = dde.icbc.PointSetBC(X_train, y_train_imag, component=1)

geom = dde.geometry.Rectangle([0, 0], [l_x, l_y])

data = dde.data.PDE(
    geom,
    pde,
    [bc_data_real, bc_data_imag],
    num_domain=5000,
    num_boundary=0, #robin boundary conditions to be added later
    anchors=X_train
)

net = dde.nn.FNN([2] + [50] * 3 + [2], "tanh", "Glorot uniform")
model = dde.Model(data, net)

#TODO: specify different loss functions for PDE and data, look into weights
model.compile("adam", lr=1e-3, loss_weights=[1, 1, 100, 100]) #loss_weights = PDE real, PDE imaginary, data real, data imaginary
losshistory, train_state = model.train(iterations=5000)