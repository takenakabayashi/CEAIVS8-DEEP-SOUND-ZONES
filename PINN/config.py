from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#Edit these to your own dataset paths
ISOBEL_DATASET_DIR = PROJECT_ROOT / "ISOBEL_SF_Dataset"
SIMULATED_DATA_FILE = PROJECT_ROOT / "simulatedData.h5"

ISOBEL_FS = 48000 #Hz, ISOBEL dataset sampling frequency

ISOBEL_ROOMS = {
    "LR": {
        "name": "Listening Room",
        "directory": ISOBEL_DATASET_DIR / "Listening Room" / "ListeningRoom_SoundField_IRs",
        "sources_positions": [(0.17, 7.53, 1.0), (1.42, 2.08, 1.0)],
        "room_dimensions": (4.14, 7.80, 2.78),
        "grid_size": (32, 32, 4),
        "heights": [100, 130, 160, 190],
    },
    "VR": {
        "name": "VR Lab",
        "directory": ISOBEL_DATASET_DIR / "VR Lab" / "VRLab_SoundField_IRs",
        "sources_positions": [(6.65, 7.93, 1.0), (5.23, 3.49, 1.0)],
        "room_dimensions": (6.98, 8.12, 3.03),
        "grid_size": (32, 32, 4),
        "heights": [100, 130, 160, 190],
    },
    "PR": {
        "name": "Product Room",
        "directory": ISOBEL_DATASET_DIR / "Product Room" / "ProductRoom_SoundField_IRs",
        "sources_positions": [(0.32, 0.22, 1.0), (4.48, 4.81, 1.0)],
        "room_dimensions": (9.13, 12.03, 2.60),
        "grid_size": (32, 32, 3),
        "heights": [130, 160, 190],
    },
    # Room B is missing sources position
    # "RB": {
    #     "directory": ISOBEL_DATASET_DIR / "Room B" / "RoomB_SoundField_IRs",
    #     "sources_positions": [(), ()],  # source_1 and source_2
    #     "room_dimensions": (4.16, 6.46, 2.30),
    #     "grid_size": (32, 32, 1),
    #     "heights": [100],
    # },
}
