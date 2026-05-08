import numpy as np
from scipy.linalg import eigh

def acoustic_contrast_control(H_bright, H_dark, freq_idx=None):
    H_b = H_bright[:, :, freq_idx] if H_bright.ndim == 3 else H_bright
    H_d = H_dark[:, :, freq_idx]   if H_dark.ndim == 3 else H_dark
    
    src_number = H_b.shape[1]
    
    R_b = H_b @ H_b.conj().T
    R_d = H_d @ H_d.conj().T
    
    # regularization to ensure stability and avoid singularities
    lam = 1e-3 * np.linalg.norm(R_b, 'fro') / src_number
    R_d_reg = R_d + lam * np.eye(R_d.shape[0])
    
    # ensure there is positive defiteness for the eigenvalue problem
    R_b_reg = R_b + lam * np.eye(R_b.shape[0])
    
    min_eig = np.linalg.eigvalsh(R_d_reg).min()
    if min_eig <= 0:
        R_d_reg += (abs(min_eig) + 1e-6) * np.eye(R_d_reg.shape[0])
    
    # compute the weights using the generalised eigenvalue problem
    try:
        _, vecs = eigh(R_b_reg, R_d_reg)
    except np.linalg.LinAlgError:
        _, vecs = np.linalg.eigh(np.linalg.pinv(R_d_reg) @ R_b_reg)
        
    W = vecs[:, -1]
    W /= np.linalg.norm(W) + 1e-10
    
    # compute the resulting pressures in bright and dark zones
    pres_B = H_b @ W
    pres_D = H_d @ W
    
    # compute energies and contrast
    bright_e = np.mean(np.abs(pres_B)**2)
    dark_e = np.mean(np.abs(pres_D)**2)
    contrast_dB = 10 * np.log10((bright_e + 1e-10) / (dark_e + 1e-10))
    effort_dB = 10 * np.log10(np.sum(np.abs(W)**2) + 1e-10)
    
    return W, contrast_dB, bright_e, dark_e, effort_dB

def pressure_matching(H_bright, H_dark, target_bright, target_dark, freq_idx=None):
    H_b = H_bright[:, :, freq_idx] if H_bright.ndim == 3 else H_bright
    H_d = H_dark[:, :, freq_idx]   if H_dark.ndim == 3 else H_dark
    
    src_number = H_b.shape[1]
    
    # Stack the bright and dark zone targets and transfer functions
    H_stack = np.vstack((H_b, H_d))
    target_stack = np.hstack((target_bright, target_dark))
    
    # Solve the least squares problem to find the weights
    W, _, _, _ = np.linalg.lstsq(H_stack, target_stack, rcond=None)
    
    # compute the resulting pressures in bright and dark zones
    pres_B = H_b @ W
    pres_D = H_d @ W
    
    # compute energies and contrast
    bright_e = np.mean(np.abs(pres_B)**2)
    dark_e = np.mean(np.abs(pres_D)**2)
    contrast_dB = 10 * np.log10((bright_e + 1e-10) / (dark_e + 1e-10))
    effort_dB = 10 * np.log10(np.sum(np.abs(W)**2) + 1e-10)
    
    return W, contrast_dB, bright_e, dark_e, effort_dB