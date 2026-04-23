import h5py, os, pandas as pd
from sklearn.model_selection import train_test_split

#file_path = r'C:\Users\nicol\Desktop\Code Projects\Python\AVS8\CEAIVS8-DEEP-SOUND-ZONES\simulatedData.h5'

def create_pd(file_path):

    rows = []

    with h5py.File(file_path, "r") as f:
        rooms = list(f.keys())

        for room in rooms:
            grp = f[room]

            room_dim = grp["room_dim"][0]
            receivers = grp["receiver_pos"].shape[0]

            T60 = grp["T60"][0][0]
            alpha = grp["alpha"][0][0]

            L, W, H = room_dim
            volume = L * W * H

            rows.append({
                "room": room,
                "L": L,
                "W": W,
                "H": H,
                "volume": volume,
                "n_receivers": receivers,
                "T60": T60,
                "alpha": alpha
            })
    
    # Create a panda dataframe from the file
    df = pd.DataFrame(rows)

    # Takes the full room path and extracts just the name
    df["room_id"] = df["room"].apply(lambda x: os.path.basename(x))
    
    #print(df.head()) # print head to check if it match expectations
    #print(len(df)) # Check if there indeed are 20000 entries

    # --------------------------------------------------
    # The next two are essentiel for proper distribution
    # --------------------------------------------------

    # Create volume bins (quantiles = balanced)
    df["volume_bin"] = pd.qcut(df["volume"], q=5, labels=False, duplicates="drop")

    # Combine alpha + volume_bin into one stratification label
    df["stratify_key"] = df["alpha"].astype(str) + "_" + df["volume_bin"].astype(str)

    return df


def get_train_val_test_data(file_path):
    
    # Call function that returns a panda dataframe of the data
    df = create_pd(file_path=file_path)

    # Train split: 70%, val/test: 30%
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["stratify_key"],
        random_state=42
    )

    # Val split: 15%, test: 15%, they split the remaining entries in temp_df in half
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["stratify_key"],
        random_state=42
    )

    # Resets the index to 0
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, val_df, test_df

""" 
train_df, val_df, test_df = get_train_val_test_data(file_path=file_path)

print(train_df.head())


print(len(train_df), len(val_df), len(test_df))

# The following shows the distribution in volume,
# alpha and T60. The data was split up like this,
# so both train, val and test got a fair chunk of
# all sizes 


import seaborn as sns
import matplotlib.pyplot as plt

for col in ["volume", "alpha", "T60"]:
    plt.figure()
    sns.kdeplot(train_df[col], label="train")
    sns.kdeplot(val_df[col], label="val")
    sns.kdeplot(test_df[col], label="test")
    plt.title(col)
    plt.legend()
    plt.show()  """