import subprocess
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
from demeter.constants import ROOT_DIRECTORY
import torch.cuda as cda

size_list = np.linspace(100, 10000, 40, dtype=int)
csv_file = os.path.join(
    ROOT_DIRECTORY,
    "examples/results",
    "benchmark_results.csv"
)

if cda.is_available():
    device_index = cda.current_device()
    gpu_name = cda.get_device_name(device_index)
    total_memory = cda.get_device_properties(device_index).total_memory

    print(f"GPU Name: {gpu_name}")
    print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")
else:
    print("CUDA is not available.")

# Lire le CSV s’il existe
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)
else:
    existing_df = pd.DataFrame(columns=["width", "height", "save_gpu", "mem_usage", "exec_time"])

for save_gpu in [False, True]:
    for size in size_list:
        # Vérifie si cette config est déjà présente
        already_done = (
            ((existing_df["width"] == size) &
             (existing_df["height"] == size) &
             (existing_df["save_gpu"] == save_gpu))
        ).any()

        if already_done:
            print(f"Déjà fait: size=({size},{size}), save_gpu={save_gpu} → Ignoré.")
            continue

        # Sinon, on lance le benchmark
        print(f"Lancement: size=({size},{size}), save_gpu={save_gpu}")
        subprocess.run([
            "python3",
             "examples/3_utils/bench_execute_meta.py",
            str(size),
            str(size),
            str(save_gpu),
            csv_file
        ])


# Lecture des résultats
df = pd.read_csv(csv_file)

# Nettoyage / transformation
df["size"] = df["width"] * df["height"]  # on suppose que width = height
df["mem_usage"] = pd.to_numeric(df["mem_usage"], errors='coerce')
df["exec_time"] = pd.to_numeric(df["exec_time"], errors='coerce')

# Étape 4 : Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f" {gpu_name}; Total Memory: {total_memory / (1024 ** 3):.2f} GB")

for save_gpu in [False, True]:
    sub_df = df[df["save_gpu"] == save_gpu]
    label = 'with save_gpu' if save_gpu else 'no save_gpu'
    ax[0].plot(sub_df["size"], sub_df["mem_usage"], label=label)
    ax[1].plot(sub_df["size"], sub_df["exec_time"], label=label)

ax[0].set_title("Memory usage")
ax[0].set_xlabel("Image pixel count")
ax[0].set_ylabel("Bytes")
ax[0].legend()
ax[0].grid(True)

ax[1].set_title("Execution time")
ax[1].set_xlabel("Image pixel count")
ax[1].set_ylabel("Seconds")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
img_path = os.path.join(os.path.dirname(csv_file),"benchmark_plot.png")
plt.savefig(img_path)
print(f"Plot saved to {img_path}")