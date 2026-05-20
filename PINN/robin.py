""" 
Predicts FFT of the impulse response at a specific (x,y,z) position for any room.
Uses all sources and all heights for multiple rooms as training data, modeled as a 3D problem.
It takes normalized x,y,z coordinates, room dimensions and source position as input, output is the predicted real and imaginary part of the FFT at the specified target frequency.
TODO: add room absorption parameter, add Robin boundary conditions, look into activation function, number of iterations and loss weights
"""
print('Importing libraries...')
import time # for testing
t_0 = time.perf_counter()
import numpy as np
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_rooms", type=int, default=100)
parser.add_argument("--max_points_per_room", type=int, default=100)
parser.add_argument("--num_domain", type=int, default=500)
args = parser.parse_args()
num_rooms = args.num_rooms
max_points_per_room = args.max_points_per_room
num_domain = args.num_domain

import torch
torch.cuda.reset_peak_memory_stats() # for finding peak memory usage later
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde

from config import SIMULATED_DATA_FILE
from data_split import get_train_val_test_data
from data_extraction_with_alpha import extract_data_ISOBEL, extract_data_simulated, get_max_min_room_dims
from utils import filter_zero_targets, nmse_db, stack_complex_targets, validation_nmse_metric

t_1 = time.perf_counter() - t_0
print(f'Imported libraries! ({t_1:.5f} seconds)\n')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
dde.config.set_random_seed(SEED)

TARGET_FREQ = 31.5 #Hz

val_fraction = 0.5

c = 343.0 #m/s
omega = 2 * np.pi * TARGET_FREQ
k = omega / c

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L_min, L_max = get_max_min_room_dims(file_path=SIMULATED_DATA_FILE)
L_min = torch.tensor(L_min, device=device)
L_max = torch.tensor(L_max, device=device)

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
            loss_fn = [loss_fn] * 6
        else:
            loss_fn = list(loss_fn)

        (   
            pde_real_loss_fn, 
            pde_imag_loss_fn, 
            data_real_loss_fn, 
            data_imag_loss_fn ,
            robin_real_loss_fn,
            robin_imag_loss_fn,
        ) = loss_fn

        # pde losses
        pde_residual_real, pde_residual_imag = self.pde(inputs, outputs)
        
        pde_loss_real = pde_real_loss_fn(
            torch.zeros_like(pde_residual_real), 
            pde_residual_real
        )
        pde_loss_imag = pde_imag_loss_fn(
            torch.zeros_like(pde_residual_imag), 
            pde_residual_imag
        )
        
        # data losses
        data_loss_real = data_real_loss_fn(
            targets[:, 0:1], 
            outputs[:, 0:1]
        )
        data_loss_imag = data_imag_loss_fn(
            targets[:, 1:2], 
            outputs[:, 1:2]
        )

        # robin bc losses
        robin_r = robin_real(inputs, outputs, None)
        robin_i = robin_imag(inputs, outputs, None)

        robin_loss_real = robin_real_loss_fn(
            torch.zeros_like(robin_r),
            robin_r
        )

        robin_loss_imag = robin_imag_loss_fn(
            torch.zeros_like(robin_i),
            robin_i
        )

        return [pde_loss_real, pde_loss_imag, data_loss_real, data_loss_imag, robin_loss_real, robin_loss_imag]
    
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
    # we could first compute real and imag gradients, then compute gradients on each dimension from that
    # it would use jacobians instead, and we would only have 2 computation graphs (hessians makes 6)

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
    # we can pre-compute the constant 'coeff = 1 / ((sigma * np.sqrt(2 * np.pi)) ** 3)'
    # jimmy would be a happy man if we used 'x*x' rather than 'x**2' :)

    # r=rooms, mp=max_points, co=num_domain
    # mp = max_points_per_room
    # with open('memory_usage/r{num_rooms_{mp}-{mp}-{mp}_co{num_domain}', 'a') as memory_file:
        # print(f"{time.strftime('%H:%M:%S')} | GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB", file=memory_file)

    return [-y0_xx - y0_yy - y0_zz - k ** 2 * y0 - f,
            -y1_xx - y1_yy - y1_zz - k ** 2 * y1]

print('Loading simulated data...')
t_0 = time.perf_counter()

train_df, val_df, test_df = get_train_val_test_data(file_path=SIMULATED_DATA_FILE, subset_size=num_rooms)
# Extract data per split
X_train, y_train = extract_data_simulated(df=train_df, target_freq=TARGET_FREQ, file_path=SIMULATED_DATA_FILE, max_points_per_room=max_points_per_room)
X_val, y_val = extract_data_simulated(df=val_df, target_freq=TARGET_FREQ, file_path=SIMULATED_DATA_FILE, max_points_per_room=max_points_per_room)
X_test, y_test = extract_data_simulated(df=test_df, target_freq=TARGET_FREQ, file_path=SIMULATED_DATA_FILE, max_points_per_room=max_points_per_room)

t_1 = time.perf_counter() - t_0
print(f'Data has been loaded! ({t_1:.5f} seconds)\n')

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
geom = dde.geometry.geometry_nd.Hypercube(xmin=[0] * 10, xmax=[1] * 10) # 10 dimension when absorption is included

# ROBIN BOUNDARY CONDITION
def boundary_fn(x, on_boundary): # this assumes that all floors, wall, and ceiling has the same absorption properties
    eps = 1e-6
    return on_boundary and (
        abs(x[0]) < eps or abs(x[0] - 1) < eps or
        abs(x[1]) < eps or abs(x[1] - 1) < eps or
        abs(x[2]) < eps or abs(x[2] - 1) < eps
    )

def compute_impedance_numpy(abs_coeff):
    abs_coeff = np.clip(abs_coeff, 0.1, 0.9) # since full 0 or 1 might give issues (we only have 0.1, 0.3, 0.6, 0.9)
    impedance = (1 + np.sqrt(1 - abs_coeff)) / (1 - np.sqrt(1 - abs_coeff))
    return impedance

def compute_impedance_torch(abs_coeff):
    abs_coeff = torch.clamp(abs_coeff, 0.1, 0.9) # since full 0 or 1 might give issues (we only have 0.1, 0.3, 0.6, 0.9)
    impedance = (1 + torch.sqrt(1 - abs_coeff)) / (1 - torch.sqrt(1 - abs_coeff))
    return impedance

'''
abs_coeff = 0.6 # placeholder, until we also gather absorption from the data
Z = compute_impedance_numpy(abs_coeff)
# a = 1.0 / Z
# b = 1.0
k_over_Z = k / Z
'''

def get_physical_normal(x):
    # X in normalized [0,1]
    eps = 1e-6

    norm_x = torch.where(torch.abs(x[:, 0:1]) < eps, -1.0,
                torch.where(torch.abs(x[:, 0:1] - 1) < eps, 1.0, 0.0))
    
    norm_y = torch.where(torch.abs(x[:, 1:2]) < eps, -1.0,
                torch.where(torch.abs(x[:, 1:2] - 1) < eps, 1.0, 0.0))
    
    norm_z = torch.where(torch.abs(x[:, 2:3]) < eps, -1.0,
                torch.where(torch.abs(x[:, 2:3] - 1) < eps, 1.0, 0.0))

    return torch.cat([norm_x, norm_y, norm_z], dim=1)

def robin_real(x, y, X):
    Lx = x[:, 3:4] * (L_max[0] - L_min[0]) + L_min[0]
    Ly = x[:, 4:5] * (L_max[1] - L_min[1]) + L_min[1]
    Lz = x[:, 5:6] * (L_max[2] - L_min[2]) + L_min[2]

    alpha = x[:, 9:10]
    Z = compute_impedance_torch(alpha)
    k_over_Z = k / Z

    grad_u_r = dde.grad.jacobian(y, x, i=0)  # du_r
    grad_u_r[:, 0:1] /= Lx
    grad_u_r[:, 1:2] /= Ly
    grad_u_r[:, 2:3] /= Lz
    grad_u_r = grad_u_r[:, 0:3] # we do not want the rest of the 9d hypercube to stay in memory

    # we only want to use the coordinates, not room dimensions and source position
    normal = get_physical_normal(x)

    # the boundary point might be a corner and ruin the normal
    normal_norm = torch.norm(normal, dim=1, keepdim=True) + 1e-12
    normal = normal / normal_norm

    dur_dn = torch.sum(grad_u_r * normal, dim=1, keepdim=True) # du_r / dn

    u_i = y[:, 1:2]

    return dur_dn - k_over_Z * u_i

def robin_imag(x, y, X):
    Lx = x[:, 3:4] * (L_max[0] - L_min[0]) + L_min[0]
    Ly = x[:, 4:5] * (L_max[1] - L_min[1]) + L_min[1]
    Lz = x[:, 5:6] * (L_max[2] - L_min[2]) + L_min[2]

    alpha = x[:, 9:10]
    Z = compute_impedance_torch(alpha)
    k_over_Z = k / Z
    
    grad_u_i = dde.grad.jacobian(y, x, i=1)  # du_i
    grad_u_i[:, 0:1] /= Lx
    grad_u_i[:, 1:2] /= Ly
    grad_u_i[:, 2:3] /= Lz
    grad_u_i = grad_u_i[:, 0:3] # we do not want the rest of the 9d hypercube to stay in memory

    # we only want to use the coordinates, not room dimensions and source position
    normal = get_physical_normal(x)

    # the boundary point might be a corner and ruin the normal
    normal_norm = torch.norm(normal, dim=1, keepdim=True) + 1e-12
    normal = normal / normal_norm

    dui_dn = torch.sum(grad_u_i * normal, dim=1, keepdim=True) # du_i / dn

    u_r = y[:, 0:1]

    return dui_dn + k_over_Z * u_r

# dde.icbc.RobinBC() connot represent complex coupling, so we have to do this
bc_robin_real = dde.icbc.OperatorBC(geom, robin_real, boundary_fn)
bc_robin_imag = dde.icbc.OperatorBC(geom, robin_imag, boundary_fn)

y_val_targets = stack_complex_targets(y_val)

data = ValidationPDE(
    geom,
    pde,
    [bc_data_real, bc_data_imag, bc_robin_real, bc_robin_imag],
    num_domain=num_domain, # change back to numeric value after testing
    num_boundary=2048, # we dont have any training points on the boundary
    # anchors=X_train,
    validation_x=X_val,
    validation_y=y_val_targets,
)

net = dde.nn.FNN([10] + [32] * 3 + [2], "tanh", "Glorot uniform") # 10 dimension when absorption is included
model = dde.Model(data, net)

pde_w = 1 # pde real, pde imag
data_w = 100 # data real, data imag
robin_w = 10 # robin real, robin imag
loss_weights = [pde_w, pde_w, data_w, data_w, robin_w, robin_w] #1/100/10
model.compile(
    "adam", 
    lr=1e-3, 
    loss="MSE", 
    loss_weights=loss_weights,
    metrics=[validation_nmse_metric],
)

# callback
checker = dde.callbacks.ModelCheckpoint(
    "ckpt/robin_dyn", 
    save_better_only=True, 
    period=1000
)

resampler = dde.callbacks.PDEPointResampler(
    period=500,
    pde_points=True,
    bc_points=False  # keeps PointSetBC stable (anchor points)
)

losshistory, train_state = model.train(
    iterations=25000,
    display_every=100,
    batch_size=128,
    model_save_path="ckpt/robin_dyn",
    callbacks=[checker],
    # callbacks=[resampler],
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

peak = torch.cuda.max_memory_allocated()
print(f'\nRooms: {num_rooms}\nMax Points: {max_points_per_room}\nCollocation Points: {num_domain}')
print(f"Peak memory: {peak/1e9:.2f} GB")



'''
# ============================================================
# VERIFY ROBIN BOUNDARY CONDITION AFTER TRAINING
# ============================================================

import torch
import numpy as np
import deepxde as dde

# ------------------------------------------------------------
# Sample fresh boundary points
# ------------------------------------------------------------
N_bc = 2000

X_bc = geom.random_boundary_points(N_bc).astype(np.float32)

# ------------------------------------------------------------
# Keep ONLY physical room walls
# (ignore boundaries in room dims/source position dimensions)
# Also exclude corners/edges
# ------------------------------------------------------------
eps = 1e-6

x_hits = (np.abs(X_bc[:, 0]) < eps) | (np.abs(X_bc[:, 0] - 1) < eps)
y_hits = (np.abs(X_bc[:, 1]) < eps) | (np.abs(X_bc[:, 1] - 1) < eps)
z_hits = (np.abs(X_bc[:, 2]) < eps) | (np.abs(X_bc[:, 2] - 1) < eps)

# Only keep points lying on EXACTLY one wall
num_hits = x_hits.astype(int) + y_hits.astype(int) + z_hits.astype(int)

mask = num_hits == 1

X_bc = X_bc[mask]

print(f"\nUsing {len(X_bc)} boundary points for Robin BC verification")

# ------------------------------------------------------------
# Convert to tensor with gradients enabled
# ------------------------------------------------------------
x_tensor = torch.tensor(
    X_bc,
    dtype=torch.float32,
    requires_grad=True,
)

# ------------------------------------------------------------
# Predict solution
# ------------------------------------------------------------
y_tensor = model.net(x_tensor)

u_r = y_tensor[:, 0:1]
u_i = y_tensor[:, 1:2]

# ------------------------------------------------------------
# Compute gradients
# ------------------------------------------------------------
grad_u_r = dde.grad.jacobian(y_tensor, x_tensor, i=0)
grad_u_i = dde.grad.jacobian(y_tensor, x_tensor, i=1)

# ------------------------------------------------------------
# Convert normalized derivatives to physical derivatives
# ------------------------------------------------------------
Lx = x_tensor[:, 3:4] * (L_max[0] - L_min[0]) + L_min[0]
Ly = x_tensor[:, 4:5] * (L_max[1] - L_min[1]) + L_min[1]
Lz = x_tensor[:, 5:6] * (L_max[2] - L_min[2]) + L_min[2]

# Real gradients
grad_u_r[:, 0:1] /= Lx
grad_u_r[:, 1:2] /= Ly
grad_u_r[:, 2:3] /= Lz

# Imag gradients
grad_u_i[:, 0:1] /= Lx
grad_u_i[:, 1:2] /= Ly
grad_u_i[:, 2:3] /= Lz

# Keep ONLY spatial derivatives
grad_u_r = grad_u_r[:, 0:3]
grad_u_i = grad_u_i[:, 0:3]

# ------------------------------------------------------------
# Compute physical outward normals
# ------------------------------------------------------------
def get_physical_normal(X):
    eps = 1e-6

    norm_x = torch.where(
        torch.abs(X[:, 0:1]) < eps,
        -1.0,
        torch.where(
            torch.abs(X[:, 0:1] - 1) < eps,
            1.0,
            0.0,
        ),
    )

    norm_y = torch.where(
        torch.abs(X[:, 1:2]) < eps,
        -1.0,
        torch.where(
            torch.abs(X[:, 1:2] - 1) < eps,
            1.0,
            0.0,
        ),
    )

    norm_z = torch.where(
        torch.abs(X[:, 2:3]) < eps,
        -1.0,
        torch.where(
            torch.abs(X[:, 2:3] - 1) < eps,
            1.0,
            0.0,
        ),
    )

    normal = torch.cat([norm_x, norm_y, norm_z], dim=1)

    # Normalize normals
    normal_norm = torch.norm(normal, dim=1, keepdim=True) + 1e-12
    normal = normal / normal_norm

    return normal

normal = get_physical_normal(x_tensor)

# ------------------------------------------------------------
# Compute normal derivatives
# ------------------------------------------------------------
dur_dn = torch.sum(grad_u_r * normal, dim=1, keepdim=True)
dui_dn = torch.sum(grad_u_i * normal, dim=1, keepdim=True)

# ------------------------------------------------------------
# Compute Robin residuals
#
# Robin BC:
#   du_r/dn - (k/Z) u_i = 0
#   du_i/dn + (k/Z) u_r = 0
# ------------------------------------------------------------
robin_res_real = dur_dn - k_over_Z * u_i
robin_res_imag = dui_dn + k_over_Z * u_r

# ------------------------------------------------------------
# Compute statistics
# ------------------------------------------------------------
mean_real = torch.mean(torch.abs(robin_res_real)).item()
mean_imag = torch.mean(torch.abs(robin_res_imag)).item()

max_real = torch.max(torch.abs(robin_res_real)).item()
max_imag = torch.max(torch.abs(robin_res_imag)).item()

rms_real = torch.sqrt(torch.mean(robin_res_real**2)).item()
rms_imag = torch.sqrt(torch.mean(robin_res_imag**2)).item()

# print("\n================================================")
# print("ROBIN BC VERIFICATION")
# print("================================================")

# print(f"REAL residual:")
# print(f"  Mean abs : {mean_real:.3e}")
# print(f"  RMS      : {rms_real:.3e}")
# print(f"  Max abs  : {max_real:.3e}")

# print(f"\nIMAG residual:")
# print(f"  Mean abs : {mean_imag:.3e}")
# print(f"  RMS      : {rms_imag:.3e}")
# print(f"  Max abs  : {max_imag:.3e}")

# ------------------------------------------------------------
# Relative residuals (VERY IMPORTANT)
# ------------------------------------------------------------
u_mag = torch.sqrt(u_r**2 + u_i**2)

mean_u_mag = torch.mean(u_mag).item()

rel_real = mean_real / (mean_u_mag + 1e-12)
rel_imag = mean_imag / (mean_u_mag + 1e-12)

print("\nRelative residuals:")
print(f"  REAL: {rel_real:.3e}")
print(f"  IMAG: {rel_imag:.3e}")

# print("================================================")
'''