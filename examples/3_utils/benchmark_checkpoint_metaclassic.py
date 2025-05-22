"""
In this file we perform a benchmark tracking execution time and memory usage with
or without the option for saving gpu memory.

This file execute the program `examples/3_utils/execute_meta.py`

By defauft I provide the results for my laptop. You can change the path of
`csv_file` to try on your computer.

"""


import subprocess
import numpy as np
import matplotlib.pyplot as plt
import csv
import os, ast
import pandas as pd
import torch

from demeter.constants import ROOT_DIRECTORY
import torch.cuda as cda
from math import prod

from demeter.utils.toolbox import convert_bytes_size

size_list = np.linspace(100, 1000, 10, dtype=int)
name_csv = "benchmark_results_memory_classic.csv"


csv_file = os.path.join(
    ROOT_DIRECTORY,
    "examples/results/benchmark/",
    name_csv
)

# 1. collect gpu information
if cda.is_available():
    device_index = cda.current_device()
    gpu_name = cda.get_device_name(device_index)
    total_memory = cda.get_device_properties(device_index).total_memory

    print(f"GPU Name: {gpu_name}")
    print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")
else:
    raise ValueError("CUDA is not available.")


# Lire le CSV s’il existe
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)
    print(existing_df.keys())
    existing_df['resize factor'] = existing_df['resize factor'].apply(ast.literal_eval)

else:
    existing_df = pd.DataFrame(columns=[
                "img shape",
                'resize factor',
                "save gpu",
                "mem usage bytes",
                "exec time sec",
            ])

#%%
for save_gpu in [False, True]:
    for size in size_list:
        # Vérifie si cette config est déjà présente
        already_done = (
            ((existing_df["resize factor"] == (size, size, 1)) &
             (existing_df["save gpu"] == save_gpu))
        ).any()

        if already_done:
            print(f"Déjà fait: size=({(size, size, 1)}), save_gpu={save_gpu} → Ignoré.")
            continue

        # Sinon, on lance le benchmark
        print(f"\nLancement: size=({(size, size, 1)}), save_gpu={save_gpu}")
        print("python3",
            "examples/3_utils/execute_meta.py",
            "--width", str(size),
            "--height", str(size),
            "--save_gpu", str(save_gpu).lower(),
            "--csv_file", str(csv_file)
        )
        subprocess.run([
            "python3",
            "examples/3_utils/execute_meta.py",
            "--width", str(size),
            "--height", str(size),
            "--save_gpu", str(save_gpu).lower(),
            "--csv_file", str(csv_file),
        ], check=True)


# Lecture des résultats
df = pd.read_csv(csv_file)
df['img shape'] = df['img shape'].apply(ast.literal_eval)

# Nettoyage / transformation
# df["size"] = df["width"] * df["height"]  # on suppose que width = height
df["size"] = df["img shape"].apply(prod)

#%%
df = pd.read_csv(csv_file)
df['img shape'] = df['img shape'].apply(ast.literal_eval)

# Nettoyage / transformation
# df["size"] = df["width"] * df["height"]  # on suppose que width = height
df["size"] = df["img shape"].apply(prod)

df["mem usage bytes"] = pd.to_numeric(df["mem usage bytes"], errors='coerce')
df["exec time sec"] = pd.to_numeric(df["exec time sec"], errors='coerce')

# Étape 4 : Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f" {gpu_name}; Total Memory: {total_memory / (1024 ** 3):.2f} GB")

for save_gpu in [False, True]:
    sub_df = df[df["save gpu"] == save_gpu]

    # Ajouter calcul du coef du modèle y = ax + b
    df_min = sub_df[sub_df["mem usage bytes"] == sub_df["mem usage bytes"].min()]
    df_max = sub_df[sub_df["mem usage bytes"] == sub_df["mem usage bytes"].max()]
    a = ((df_max["mem usage bytes"].item() - df_min["mem usage bytes"].item()) /
         (df_max["size"].item() - df_min["size"].item()))
    b = df_min["mem usage bytes"].item() - a * df_min["size"].item()
    print(f"modèle checkpoint : {save_gpu} : "
          f"a: ({a}, {convert_bytes_size(a)}), "
          f"b: ({b}, {convert_bytes_size(b)})")

    label = 'with checkpoint' if save_gpu else 'no checkpoint'
    size_max = df_max["size"].max().item()
    x = torch.linspace(0,size_max * (1 + .3), 5 )
    ax[0].plot(x, a*x +b, '--', label=f"model checkpoint : {save_gpu}",
               c= 'C1' if save_gpu else 'C2')

    ax[0].plot(sub_df["size"], sub_df["mem usage bytes"],
            label=label,
            marker='o',
            c = 'C1' if save_gpu else 'C2'
    )

    ax[1].plot(sub_df["size"], sub_df["exec time sec"] /60,
               label=label,
               marker='o',
               c = 'C1' if save_gpu else 'C2'
    )

ax[0].hlines(total_memory, 0, size_max * (1 + .3),
             label="memory_limit",
             linestyle ='--',
             color ='k'
             )

ax[0].set_title("Memory usage")
ax[0].set_xlabel("Image pixel count")
ax[0].set_ylabel("Bytes")
ax[0].legend()
ax[0].grid(True)

ax[1].set_title("Execution time")
ax[1].set_xlabel("Image pixel count")
ax[1].set_ylabel("minutes")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
img_path = os.path.join(os.path.dirname(csv_file),f"benchmark_plot_memory_meta.png")
plt.show()
# plt.savefig(img_path)
print(f"Plot saved to {img_path}")