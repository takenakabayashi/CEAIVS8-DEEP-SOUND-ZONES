""" 
Single room approach: PINN model to predict FFT of the impulse responses along a 32x32 grid for one specific room, given n_mic training points. Currently using listening room.
Currently uses one source (hard-coded) and all heights, modeled as a 3D problem.
It takes x,y,z coordinates as input, output is the predicted real and imaginary part of the FFT at the specified target frequency.
TODO: add Robin boundary conditions, look into activation function, number of iterations and loss weights
"""
import os
from sklearn.model_selection import train_test_split
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np

from data_extraction import extract_grid
from utils import filter_zero_targets, nmse_db, stack_complex_targets, validation_nmse_metric
from config import ISOBEL_FS, ISOBEL_ROOMS

TARGET_FREQ = 41 #Hz
ROOM = ISOBEL_ROOMS["VR"]
SOURCE = 1

abs_coeff_rb = 0.083
abs_coeff_vr = 0.122
abs_coeff_lr = 0.046
abs_coeff_pr = 0.060

n_mic = 15 #number of points used for training
val_fraction = 0.5

c = 343.0 #m/s
omega = 2 * np.pi * TARGET_FREQ
k = omega / c

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

def pde(x, y):
    y0, y1 = y[:, 0:1], y[:, 1:2] 

    y0_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    y0_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    y0_zz = dde.grad.hessian(y, x, component=0, i=2, j=2)

    y1_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    y1_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    y1_zz = dde.grad.hessian(y, x, component=1, i=2, j=2)

    xs, ys, zs = ROOM["sources_positions"][SOURCE] #source position
    sigma = 0.1
    dist = (x[:, 0:1] - xs)**2 + (x[:, 1:2] - ys)**2 + (x[:, 2:3] - zs)**2
    f = (1 / ((sigma * np.sqrt(2 * np.pi)) ** 3)) * torch.exp(-0.5 * dist / sigma**2) 

    return [-y0_xx - y0_yy - y0_zz - k ** 2 * y0 - f,
            -y1_xx - y1_yy - y1_zz - k ** 2 * y1] 

X, y = extract_grid(ROOM, SOURCE, ISOBEL_FS, TARGET_FREQ)
X, y, valid_mask = filter_zero_targets(X, y)

removed_points = int((~valid_mask).sum())
print(f"Filtered out {removed_points} points")

if len(X) <= n_mic:
    raise ValueError(f"Not enough non-zero targets after filtering: {len(X)} points available, but train_size={n_mic}")

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, train_size=n_mic, random_state=42)

#Split again to get test and validation data
X_val, X_test, y_val, y_test = train_test_split(
    X_holdout,
    y_holdout,
    test_size=1 - val_fraction,
    random_state=42,
)

# alpha = abs_coeff_vr

# def append_alpha(X, alpha):
#     alpha_col = np.full((X.shape[0], 1), alpha, dtype=np.float32)
#     return np.hstack([X, alpha_col])

# X_train_alpha = append_alpha(X_train, alpha)
# X_val_alpha = append_alpha(X_val, alpha)
# X_test_alpha = append_alpha(X_test, alpha)

y_train_real = np.real(y_train).astype(np.float32)
y_train_imag = np.imag(y_train).astype(np.float32)

print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

bc_data_real = dde.icbc.PointSetBC(X_train, y_train_real, component=0)
bc_data_imag = dde.icbc.PointSetBC(X_train, y_train_imag, component=1)

l_x, l_y, l_z = ROOM["room_dimensions"]
geom = dde.geometry.geometry_nd.Hypercube([0, 0, 0], [l_x, l_y, l_z])
# geom = dde.geometry.Cuboid([0,0,0], [l_x,l_y,l_z])
# alpha_domain = dde.geometry.TimeDomain(0.1, 0.9)
# geom = dde.geometry.GeometryXTime(geom, alpha_domain)

# ROBIN BOUNDARY CONDITION
def boundary_fn(x, on_boundary): # this assumes that all floors, wall, and ceiling has the same absorption properties
    eps = 1e-6
    return on_boundary and (
        abs(x[0]) < eps or abs(x[0] - 1) < eps or # on x wall
        abs(x[1]) < eps or abs(x[1] - 1) < eps or # on y wall
        abs(x[2]) < eps or abs(x[2] - 1) < eps    # on z wall
    )

def compute_impedance_numpy(abs_coeff):
    eps = 5e-3
    abs_coeff = np.clip(abs_coeff, abs_coeff-eps, abs_coeff_vr+eps) # just to avoid weird sampling
    impedance = (1 + np.sqrt(1 - abs_coeff)) / (1 - np.sqrt(1 - abs_coeff))
    return impedance

def compute_impedance_torch(abs_coeff):
    eps = 5e-3
    abs_coeff = np.clip(abs_coeff, abs_coeff-eps, abs_coeff_vr+eps) # just to avoid weird sampling
    impedance = (1 + torch.sqrt(1 - abs_coeff)) / (1 - torch.sqrt(1 - abs_coeff))
    return impedance

def get_physical_normal(x): # x in normalized [0,1]
    eps = 1e-6

    # find the reflected surface(s)
    norm_x = torch.where(torch.abs(x[:, 0:1]) < eps, -1.0,
                torch.where(torch.abs(x[:, 0:1] - 1) < eps, 1.0, 0.0))

    norm_y = torch.where(torch.abs(x[:, 1:2]) < eps, -1.0,
                torch.where(torch.abs(x[:, 1:2] - 1) < eps, 1.0, 0.0))
    
    norm_z = torch.where(torch.abs(x[:, 2:3]) < eps, -1.0,
                torch.where(torch.abs(x[:, 2:3] - 1) < eps, 1.0, 0.0))

    # the boundary point might be a corner and ruin the normal
    norm = torch.cat([norm_x, norm_y, norm_z], dim=1)
    normal_norm = torch.norm(norm, dim=1, keepdim=True) + 1e-12
    normal = norm / normal_norm

    return normal

def robin_real(x, y, X): # X is unused but required by the library (it is just x in a NumPy array)
    grad_u_r = dde.grad.jacobian(y, x, i=0)  # du_r
    grad_u_r[:, 0:1]
    grad_u_r[:, 1:2]
    grad_u_r[:, 2:3]

    # find the reflected surface
    normal = get_physical_normal(x)

    dur_dn = torch.sum(grad_u_r * normal, dim=1, keepdim=True) # du_r / dn

    alpha = abs_coeff_vr
    Z = compute_impedance_torch(torch.tensor(alpha))
    k_over_Z = k / Z

    u_i = y[:, 1:2]

    return dur_dn - k_over_Z * u_i

def robin_imag(x, y, X): # X is unused but required by the library (it is just x in a NumPy array)
    grad_u_i = dde.grad.jacobian(y, x, i=1)  # du_i
    grad_u_i[:, 0:1]
    grad_u_i[:, 1:2]
    grad_u_i[:, 2:3]

    # find the reflected surface
    normal = get_physical_normal(x)

    dui_dn = torch.sum(grad_u_i * normal, dim=1, keepdim=True) # du_i / dn

    alpha = abs_coeff_vr
    Z = compute_impedance_torch(torch.tensor(alpha))
    k_over_Z = k / Z

    u_r = y[:, 0:1]

    return dui_dn + k_over_Z * u_r

# dde.icbc.RobinBC() cannot represent complex coupling, so we have to do this
bc_robin_real = dde.icbc.OperatorBC(geom, robin_real, boundary_fn)
bc_robin_imag = dde.icbc.OperatorBC(geom, robin_imag, boundary_fn)

y_val_targets = stack_complex_targets(y_val)

data = ValidationPDE(
    geom,
    pde,
    [bc_data_real, bc_data_imag, bc_robin_real, bc_robin_imag],
    num_domain=10000,
    num_boundary=2048,
    anchors=X_train,
    validation_x=X_val,
    validation_y=y_val_targets,
)

net = dde.nn.FNN([3] + [50] * 3 + [2], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile(
    "adam",
    lr=1e-3,
    loss="MSE",
    loss_weights=[1, 1, 100, 100, 10, 10],
    metrics=[validation_nmse_metric],
)

losshistory, train_state = model.train(
    iterations=10000,
    display_every=100,
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

y_pred = y_pred_real + 1j * y_pred_imag
y_test = y_test_real + 1j * y_test_imag

nmse_db_val = nmse_db(y_test, y_pred)
print(f"Test NMSE: {nmse_db_val:.2f} dB")