"""
Bridge: Raghu preprocessed CSVs → PosNegDM format.

Menggabungkan rl_*_scaled.csv (train/val/test) menjadi:
  - final/sepsis_final_data_withTimes.csv  ← untuk train.py (Decision Transformer)
  - final/observations.json                ← untuk nnclsfier (mortality classifier)

Wajib dijalankan dengan Python 3 (bukan preproc_venv Python 2.7).
"""

import os, json
import numpy as np
import pandas as pd

BASE = "/home/yusuf/Yusuf/Kuliah/AI/Project/ICU/JC-Reproduce"
PROCESSED = os.path.join(BASE, "data", "processed")
FINAL = os.path.join(BASE, "data", "final")

# === 1. Load & gabung 3 scaled CSV ===
print("Loading scaled CSVs...")
train = pd.read_csv(os.path.join(PROCESSED, "rl_train_set_scaled.csv"))
val   = pd.read_csv(os.path.join(PROCESSED, "rl_val_set_scaled.csv"))
test  = pd.read_csv(os.path.join(PROCESSED, "rl_test_set_scaled.csv"))
print(f"  Train: {len(train)} rows")
print(f"  Val:   {len(val)} rows")
print(f"  Test:  {len(test)} rows")

# Gabung semua (train → val → test)
all_df = pd.concat([train, val, test], ignore_index=True)
print(f"  Total: {len(all_df)} rows")

# === 2. Urutkan per icustayid & charttime ===
all_df = all_df.sort_values(["icustayid", "charttime"]).reset_index(drop=True)

# === 3. Sequential traj ID & step ===
unique_icustayids = all_df["icustayid"].unique()
icustayid_to_traj = {iid: idx + 1 for idx, iid in enumerate(unique_icustayids)}  # 1-based

all_df["traj"] = all_df["icustayid"].map(icustayid_to_traj)

def compute_step_per_traj(df):
    """Return step counter (0-indexed) for each contiguous traj block."""
    return df.groupby("traj").cumcount()

all_df["step"] = compute_step_per_traj(all_df)

print(f"  Unique trajectories: {all_df['traj'].nunique()}")
print(f"  Max steps in a trajectory: {all_df.groupby('traj').size().max()}")

# === 4. Combined action: a = vaso_input * 5 + iv_input ===
assert all_df["vaso_input"].between(0, 4).all(), "vaso_input di luar range 0-4"
assert all_df["iv_input"].between(0, 4).all(), "iv_input di luar range 0-4"

all_df["a"] = (all_df["vaso_input"] * 5 + all_df["iv_input"]).astype(int)
assert all_df["a"].between(0, 24).all(), "combined action di luar range 0-24"

# Reward: scale dari ±100 ke ±1 (original PosNegDM pake ±1)
all_df["r"] = all_df["reward"] / 100.0

# === 5. 47 observation columns (sama persis dengan read_dst di utils.py) ===
OBSERVATION_ITEMS = [
    "gender", "mechvent", "max_dose_vaso", "re_admission", "age",
    "Weight_kg", "GCS", "HR", "SysBP", "MeanBP", "DiaBP", "Temp_C",
    "RR", "FiO2_1", "Potassium", "Sodium", "Chloride", "Glucose",
    "Magnesium", "Calcium", "Hb", "WBC_count", "Platelets_count",
    "PTT", "PT", "Arterial_pH", "paO2", "paCO2", "Arterial_BE",
    "HCO3", "Arterial_lactate", "SOFA", "SIRS", "Shock_Index",
    "PaO2_FiO2", "cumulated_balance", "SpO2", "BUN", "Creatinine",
    "SGOT", "SGPT", "Total_bili", "INR",
    "input_total", "input_4hourly", "output_total", "output_4hourly",
]

missing = [c for c in OBSERVATION_ITEMS if c not in all_df.columns]
if missing:
    raise KeyError(f"Kolom observasi tidak ditemukan di data: {missing}")

# === 6. Output CSV (format sesuai PosNegDM) ===
OUTPUT_COLS = (
    ["traj", "step", "charttime", "icustayid"]
    + OBSERVATION_ITEMS
    + ["a", "r"]
)

os.makedirs(FINAL, exist_ok=True)
csv_path = os.path.join(FINAL, "sepsis_final_data_withTimes.csv")

all_df[OUTPUT_COLS].to_csv(csv_path, index=False)
print(f"\n✅ CSV: {csv_path}")
print(f"   {len(all_df)} rows × {len(OUTPUT_COLS)} cols")

# === 7. Observations.json (untuk mortality classifier) ===
# Format: [X_train, labels, X_test]
#   X_train: 2D array [n_timesteps_train, 47]
#   labels:  nested list [[label_per_traj], [label_per_traj], ...]
#            label: 1.0 = survived, -1.0 = died
#   X_test:  2D array [n_timesteps_test, 47]
#
# Kita pakai mortality_90d sebagai label.
# Train = train+val, Test = test (biar sesuai split Raghu).
print("\nBuilding observations.json...")

train_val_df = pd.concat([train, val], ignore_index=True)
test_df = test.copy()

# Sort by icustayid, charttime for both
train_val_df = train_val_df.sort_values(["icustayid", "charttime"]).reset_index(drop=True)
test_df = test_df.sort_values(["icustayid", "charttime"]).reset_index(drop=True)

# Fitur: 47 observation columns
X_train_val = train_val_df[OBSERVATION_ITEMS].values
X_test = test_df[OBSERVATION_ITEMS].values

# Labels: mortality_90d per trajectory, flatten per-timestep
#   mortality_90d: 1 = died, 0 = survived
#   PosNegDM: 1.0 = survived, -1.0 = died
def make_labels_per_timestep(df):
    """Labels per timestep: 1.0 (survived) / -1.0 (died), grouped per traj."""
    labels_per_traj = []
    for _, group_df in df.groupby("icustayid"):
        mort = group_df["mortality_90d"].iloc[0]  # sama untuk semua baris pasien
        label = 1.0 if mort == 0 else -1.0
        n = len(group_df)
        labels_per_traj.append([label] * n)
    return labels_per_traj

labels_train_val = make_labels_per_timestep(train_val_df)
labels_test = make_labels_per_timestep(test_df)

observations = [X_train_val.tolist(), labels_train_val, X_test.tolist()]
obs_path = os.path.join(FINAL, "observations.json")
with open(obs_path, "w") as f:
    json.dump(observations, f)

print(f"✅ JSON: {obs_path}")
print(f"   X_train_val: {X_train_val.shape}")
print(f"   X_test:      {X_test.shape}")
print(f"   Label groups (train_val): {len(labels_train_val)} trajectories")
print(f"   Label groups (test):      {len(labels_test)} trajectories")

# === 8. Verifikasi read_dst bisa baca CSV kita ===
print("\nVerifying with read_dst...")
import sys
sys.path.insert(0, os.path.join(BASE, "Repo Source",
    "PosNegDM-Reinforced-Sequential-Decision-Making-for-Sepsis-Treatment"))
from utils import read_dst

trajectories = read_dst(csv_path)
print(f"   Trajectories: {len(trajectories)}")
print(f"   Sample shapes:")
for k in ["observations", "actions", "rewards", "dones"]:
    print(f"     {k}: {trajectories[0][k].shape}")
print(f"   Action values: {np.unique(trajectories[0]['actions'])}")
print(f"   Reward sum (first traj): {trajectories[0]['rewards'].sum():.4f}")
print()

print("=== Done ===")
