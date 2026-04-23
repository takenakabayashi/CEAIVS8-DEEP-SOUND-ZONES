"""
Compares simulated RIRs with measured ISOBEL RIRs at a specified target frequency
"""

from PINN.config import ISOBEL_FS, ISOBEL_ROOMS, PROJECT_ROOT
from PINN.utils import create_FFT_grid, magnitude_phase_plots, nmse_db

sim_room = {
            "LR": {
            "name": "Listening Room",
            "directory": PROJECT_ROOT / "individual_RIRs",
            "sources_positions": [(0.17, 7.53, 1.0), (1.42, 2.08, 1.0)],
            "room_dimensions": (4.14, 7.80, 2.78),
            "grid_size": (32, 32, 1),
            "heights": [100],
        },
    }

if __name__ == "__main__":
    magnitude_phase_plots(ISOBEL_ROOMS["LR"], source=1, fs=ISOBEL_FS, target_freq=41, block=False)
    magnitude_phase_plots(sim_room["LR"], source=1, fs=48000, target_freq=41)

    grid_real, _ = create_FFT_grid(ISOBEL_ROOMS["LR"], source=1, fs=ISOBEL_FS, target_freq=41)
    grid_sim, _ = create_FFT_grid(sim_room["LR"], source=1, fs=48000, target_freq=41)

    nmse_db = nmse_db(grid_real, grid_sim)
    print(f"Simulation Accuracy (NMSE): {nmse_db:.2f} dB")
