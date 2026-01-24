import os
import shutil
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import signal

# -----------------------------
# CONFIG
# -----------------------------
telem_path = Path('2014 Targa Sixty-Six')  # path telemetries
v_min = 10  # minimum velocity to consider data

# Select telemetries for training, validation and test
idx_training = [1,7] 
idx_valid    = [6]
idx_test     = [2]

# Define the min - max values for the dataset:
data_info = pd.DataFrame([{
    "vx_min": 12.5,
    "vx_max": 65.0,
    "ax_min": -8.0,
    "ax_max": 5.0,
    "ay_max": 10.0,
    "yaw_max": 30.0,
    "delta_max": 85.0
}])

cols = ["time","handwheelAngle","vxCG","axCG","ayCG","yawAngle","yawRate"]


# -----------------------------
# DOWNLOAD DATA
# -----------------------------
BASE_URL = "https://stacks.stanford.edu/file/hd122pw0365"
OUTPUT_DIR = telem_path

FILES = [
    "20140221_01_01_03_250lm.csv",
    "20140221_01_02_03_250lm.csv",
    "20140221_02_01_03_250lm.csv",
    "20140221_03_01_03_250lm.csv",
    "20140221_03_02_03_250lm.csv",
    "20140221_03_03_03_250lm.csv",
    "20140221_04_01_03_250lm.csv",
    "20140222_01_01_03_250lm.csv",
    "20140222_02_01_03_250lm.csv",
    "channels.txt",
    "README.txt",
    "250lm.csv"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in FILES:
    url = f"{BASE_URL}/{filename}"
    out_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(out_path):
        print(f"[SKIP] {filename} already exists.")
        continue

    print(f"[DOWNLOAD] {filename}")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

print("Download completed.")


# -----------------------------
# UTILS FUNCTIONS
# -----------------------------
def process_data(path):
    # Load data
    df = pd.read_csv(path, encoding="latin-1", skiprows=list(range(10)), header=[0], low_memory=False)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Find first valid row
    first_valid_idx = df.notna().all(axis=1).idxmax()
    df = df.loc[first_valid_idx:].reset_index(drop=True)

    # Remove NaN values before filtering
    df = df.dropna(subset=cols)

    # Filter the data (low-pass Butterworth)
    fs = 1000  # original sampling frequency
    f_cut = 5  # cutoff frequency
    order = 4
    b, a = signal.butter(order, f_cut / (0.5 * fs), btype='low')

    for col in cols:
        if col != "yawAngle":  # Do not filter yawAngle
            df[col] = signal.filtfilt(b, a, df[col].to_numpy())
        else:
            yaw_wrap = np.array(df[col], dtype=float)
            yaw_umwrap = np.rad2deg(np.unwrap(np.deg2rad(yaw_wrap)))
            df[col] = signal.filtfilt(b, a, yaw_umwrap)

    # Resample to 200Hz (50ms)
    df["time"] = pd.to_timedelta(df["time"], unit="s")
    df = df.set_index("time")
    df_resampled = df.resample("50ms").last() #apply(lambda x: trim_mean(x.dropna(), 0.3))
    df_resampled = df_resampled.reset_index()
    df_resampled["time"] = df_resampled["time"].dt.total_seconds()

    # Filter by velocity
    df_resampled = df_resampled[df_resampled['vxCG'] > v_min]
    df_resampled = df_resampled.reset_index(drop=True)

    return df_resampled

# -----------------------------
# VEHICLE PARAMETERS
# -----------------------------
vehicle_data = pd.read_csv(telem_path / "250lm.csv",header=0,skiprows=[1])

delta_fl = vehicle_data["roadWheelAngleFL"].to_numpy()
delta_fr = vehicle_data["roadWheelAngleFR"].to_numpy()
delta_driver = vehicle_data["handwheelAngle"].to_numpy()

den = 0.5*(delta_fl+delta_fr)
tau = np.round(np.nanmean(np.divide(delta_driver, den, out=np.full_like(den,np.nan), where=den!=0)),1)

parameters = pd.DataFrame({
    'mass'          : [vehicle_data["mass"][0]],
    'diff_ratio'    : [vehicle_data["diffRatio"][0]],
    'gear_ratio_1'  : [vehicle_data["gearRatio1"][0]],
    'gear_ratio_2'  : [vehicle_data["gearRatio2"][0]],
    'gear_ratio_3'  : [vehicle_data["gearRatio3"][0]],
    'gear_ratio_4'  : [vehicle_data["gearRatio4"][0]],
    'gear_ratio_5'  : [vehicle_data["gearRatio5"][0]],
    'Rr'            : [0.5*(vehicle_data["rollingCircumferenceRL"][0]+vehicle_data["rollingCircumferenceRR"][0])/(2*np.pi)],
    'steer_tau'     : [tau],
    'L'             : [vehicle_data["wheelbase"][0]]
})

param_dir = Path("parameters")
param_dir.mkdir(parents=True, exist_ok=True)
parameters.to_csv(param_dir / "params.csv", index=False, encoding="latin-1")
data_info.to_csv(param_dir / "data_info.csv", index=False, encoding="latin-1")

# -----------------------------
# DATASET PROCESSING
# -----------------------------
# Combine all indices and avoid duplicates
all_indices = sorted(set(idx_training + idx_valid + idx_test))

# Get all telemetry file paths
datapath = sorted(telem_path.glob("*_250lm.csv"))

# Process only the telemetries needed
datasets = {}
for idx in all_indices:
    path = datapath[idx]
    data_out = process_data(path)
    datasets[idx] = data_out

# -----------------------------
# SAVE CSV
# -----------------------------
def save_datasets(indices, out_dir_name, prefix):
    out_dir = Path("telemetries") / out_dir_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    for k, i in enumerate(indices):
        datasets[i][cols].to_csv(out_dir / f"{prefix}_{k}.csv", index=False)

save_datasets(idx_training, "training", "telemetry_train")
save_datasets(idx_valid, "validation", "telemetry_valid")
save_datasets(idx_test, "test", "telemetry_test")

# -----------------------------
# PLOT HISTOGRAMS
# -----------------------------
def plot_histograms(indices, labels_prefix, save_name, data_info):
    vx_min, vx_max = data_info['vx_min'].iloc[0], data_info['vx_max'].iloc[0]
    ax_min, ax_max = data_info['ax_min'].iloc[0], data_info['ax_max'].iloc[0]
    ay_max, yaw_max, delta_max = data_info['ay_max'].iloc[0], data_info['yaw_max'].iloc[0], data_info['delta_max'].iloc[0]

    fig,ax = plt.subplots(3,2,figsize=(8,6))
    for k,i in enumerate(indices):
        df = datasets[i]
        ax[0,0].hist(df['vxCG'], bins=15, alpha=0.5, label=f"{labels_prefix}_{i}")
        ax[0,1].hist(df['axCG'], bins=15, alpha=0.5, label=f"{labels_prefix}_{i}")
        ax[1,0].hist(df['yawRate'], bins=15, alpha=0.5, label=f"{labels_prefix}_{i}")
        ax[1,1].hist(df['ayCG'], bins=15, alpha=0.5, label=f"{labels_prefix}_{i}")
        ax[2,0].hist(df['handwheelAngle'], bins=15, alpha=0.5, label=f"{labels_prefix}_{i}")
        ax[2,1].plot(
            df['ayCG'],
            df['handwheelAngle']/tau - df['yawRate']/df['vxCG']*parameters["L"][0],'.',
            label=f"{labels_prefix}_{i}", alpha=0.5)

    # Titles and limits
    ax[0,0].set_title('Speed distribution'); ax[0,0].legend()
    ax[0,0].axvline(vx_min,color='red',linestyle='--'); ax[0,0].axvline(vx_max,color='red',linestyle='--'); ax[0,0].set_xlabel("(m/s)")

    ax[0,1].set_title('Longitudinal acceleration'); ax[0,1].legend()
    ax[0,1].axvline(ax_min,color='red',linestyle='--'); ax[0,1].axvline(ax_max,color='red',linestyle='--'); ax[0,1].set_xlabel("(m/s^2)")

    ax[1,0].set_title('Yaw Rate'); ax[1,0].legend()
    ax[1,0].axvline(-yaw_max,color='red',linestyle='--'); ax[1,0].axvline(yaw_max,color='red',linestyle='--'); ax[1,0].set_xlabel("(deg/s)")

    ax[1,1].set_title('Lateral acceleration'); ax[1,1].legend()
    ax[1,1].axvline(-ay_max,color='red',linestyle='--'); ax[1,1].axvline(ay_max,color='red',linestyle='--'); ax[1,1].set_xlabel("(m/s^2)")

    ax[2,0].set_title('Steer distribution'); ax[2,0].legend()
    ax[2,0].axvline(-delta_max,color='red',linestyle='--'); ax[2,0].axvline(delta_max,color='red',linestyle='--'); ax[2,0].set_xlabel("(deg)")

    ax[2,1].set_title('Handling Diagram'); ax[2,1].legend(); ax[2,1].set_xlabel("a_y (m/s^2)")

    plt.tight_layout()
    fig.savefig(save_name)
    plt.close(fig)


fig_dir = Path("Fig")
fig_dir.mkdir(parents=True, exist_ok=True)
plot_histograms(idx_training,"training","Fig/Fig_training_data.pdf",data_info)
plot_histograms(idx_valid,"valid","Fig/Fig_valid_data.pdf",data_info)
plot_histograms(idx_test,"test","Fig/Fig_test_data.pdf",data_info)