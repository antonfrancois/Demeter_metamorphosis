"""
In this file we perform a benchmark tracking execution time and memory usage with
or without the option for saving gpu memory.

This file execute the program `examples/3_benchmark/execute_meta.py`

By defauft I provide the results for my laptop. You can change the path of
`csv_file` to try on your computer.

"""

import numpy as np
import subprocess
import matplotlib.pyplot as plt
import csv
import os, ast
import pandas as pd
import torch

from demeter.constants import ROOT_DIRECTORY
import torch.cuda as cda
from math import prod, sqrt

from demeter.utils.toolbox import convert_bytes_size

env = os.environ.copy()
env["MKL_THREADING_LAYER"] = "GNU"

# size_list = np.linspace(2500, 3000, 1, dtype=int)
size_list = [2700]
lbfgs_history_size = 10
lbfgs_max_iter = 10
n_iter = 10
n_step = 10

name_csv = f"benchmark_results_memory_classic_ni{n_iter}_ns{n_step}_lh{lbfgs_history_size}_li{lbfgs_history_size}.csv"
# name_csv = "benchmark_results_memory_new.csv"

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
# else:
#     raise ValueError("CUDA is not available.")


# Lire le CSV s’il existe
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)
    print(existing_df.keys())
    existing_df['img shape'] = existing_df['img shape'].apply(ast.literal_eval)

else:
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            'img shape',
            "image_mem_size",
            "save_gpu",
            "n_iter",
            "n_step",
            "lbfgs_history_size",
            "lbfgs_max_iter",
            "memory allocated",
            "memory reserved",
            "time_exec",
        ])
    existing_df = pd.DataFrame(columns=[
        "img shape",
        "save gpu",
        "n_iter",
        "n_step",
        "lbfgs_history_size",
        "lbfgs_max_iter",
        "memory allocated",
        "memory reserved",
        "time_exec",
    ])

# %%


for save_gpu in [False, True]:
    for size in size_list:

        # Vérifie si cette config est déjà présente
        already_done = (
            ((existing_df["img shape"] == (1, 1, size, size)) &
             (existing_df["save_gpu"] == save_gpu)) &
            (existing_df["n_iter"] == n_iter) &
            (existing_df["n_step"] == n_step) &
            (existing_df["lbfgs_history_size"] == lbfgs_history_size) &
            (existing_df["lbfgs_max_iter"] == lbfgs_max_iter)
        ).any()

        if already_done:
            print(f"Déjà fait: size=({(size, size, 1)}), save_gpu={save_gpu} → Ignoré.")
            continue

        # Sinon, on lance le benchmark
        print(f"\nLancement: size=({(size, size, 1)}), save_gpu={save_gpu}")
        print("python3",
              "examples/3_benchmark/execute_meta.py",
              "--width", str(size),
              "--height", str(size),
              "--save_gpu", str(save_gpu).lower(),
              "--csv_file", str(csv_file)
              )
        subprocess.run([
            "python3",
            "examples/3_benchmark/execute_meta.py",
            "--width", str(size),
            "--height", str(size),
            "--save_gpu", str(save_gpu).lower(),
            "--csv_file", str(csv_file),
            "--n_iter", str(n_iter),
            "--n_step", str(n_step),
            "--lbfgs_history_size", str(lbfgs_history_size),
            "--lbfgs_max_iter", str(lbfgs_max_iter),
        ], check=True, env=env)

# Lecture des résultats
# %%
df = pd.read_csv(csv_file)
df['img shape'] = df['img shape'].apply(ast.literal_eval)

# Nettoyage / transformation
# df["size"] = df["width"] * df["height"]  # on suppose que width = height
df["size"] = df["img shape"].apply(prod)

df = pd.read_csv(csv_file)
df['img shape'] = df['img shape'].apply(ast.literal_eval)

# Nettoyage / transformation
# df["size"] = df["width"] * df["height"]  # on suppose que width = height
df["size"] = df["img shape"].apply(prod)

df["memory allocated"] = pd.to_numeric(df["memory allocated"], errors='coerce')
df["time_exec"] = pd.to_numeric(df["time_exec"], errors='coerce')

# Étape 4 : Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f" {gpu_name}; Total Memory: {total_memory / (1024 ** 3):.2f} GB")

# %%
import predict_gpu_model_image as p


# Compute théoritical values:
def predict_from_row(row):
    size = int(row["size"] ** 0.5)
    img = torch.rand((size, size), dtype=torch.float32)
    return p.predict_gpu_model_image(
        img,
        int(row["n_step"]),
        int(row["n_iter"]),
        int(row["lbfgs_max_iter"]),
        int(row["lbfgs_history_size"]),
        p.PARAM_CLASSIC_ALL_TRUE if row["save_gpu"] else p.PARAM_CLASSIC_ALL_FALSE
    )

def predict_from_size(bytes, save_gpu):
    size =  int(sqrt(bytes/32))
    img = torch.rand((size, size), dtype=torch.float32)
    return p.predict_gpu_model_image(
        img,
        n_step,
        n_iter,
        lbfgs_max_iter,
        lbfgs_history_size,
        p.PARAM_CLASSIC_ALL_TRUE if save_gpu else p.PARAM_CLASSIC_ALL_FALSE
    )

# Application de la fonction et ajout au DataFrame
df["predicted_mem_model"] = df.apply(predict_from_row, axis=1)

# %%

for save_gpu in [False, True]:
    sub_df = df[df["save_gpu"] == save_gpu]

    # # Ajouter calcul du coef du modèle y = ax + b
    # df_min = sub_df[sub_df["memory allocated"] == sub_df["memory allocated"].min()]
    # df_max = sub_df[sub_df["memory allocated"] == sub_df["memory allocated"].max()]
    # a = ((df_max["memory allocated"].item() - df_min["memory allocated"].item()) /
    #      (df_max["size"].item() - df_min["size"].item()))
    # b = df_min["memory allocated"].item() - a * df_min["size"].item()
    # print(f"modèle checkpoint : {save_gpu} : "
    #       f"a: ({a}, {convert_bytes_size(a)}), "
    #       f"b: ({b}, {convert_bytes_size(b)})")

    label = 'with checkpoint' if save_gpu else 'no checkpoint'
    # size_max = p.max_image_size_for_params(
    #     total_memory,
    #     n_step,
    #     n_iter,
    #     lbfgs_max_iter,
    #     lbfgs_history_size,
    #     p.PARAM_CLASSIC_ALL_TRUE  if save_gpu else p.PARAM_CLASSIC_ALL_FALSE
    # )
    # size_max /= 32 # put back in pixels
    size_max = sub_df["size"].max()
    print('size_max', size_max)
    x = torch.linspace(0, size_max * (1 + .5), 5, dtype=torch.int)
    y = [predict_from_size(size, save_gpu) for size in x]
    # ax[0].plot(x, a*x +b, '--', label=f"model checkpoint : {save_gpu}",
    #            c= 'C1' if save_gpu else 'C2')
    ax[0].plot(sub_df["size"], sub_df["predicted_mem_model"], '--',
               label=f"model checkpoint : {save_gpu}",
               c='C1' if save_gpu else 'C2')
    # ax[0].plot(x/32, y, '--',
    #            label=f"model checkpoint : {save_gpu}",
    #            # c='C1' if save_gpu else 'C2')
    #            c='k')


    ax[0].plot(sub_df["size"], sub_df["memory allocated"],
               label=label,
               marker='o',
               c='C1' if save_gpu else 'C2'
               )

    ax[1].plot(sub_df["size"], sub_df["time_exec"] / 60,
               label=label,
               marker='o',
               c='C1' if save_gpu else 'C2'
               )

ax[0].hlines(total_memory, 0, size_max * (1 + .3),
             label="memory_limit",
             linestyle='--',
             color='k'
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
name =  f"benchmark_plot_memory_meta_ni{n_iter}_ns{n_step}_lh{lbfgs_history_size}_li{lbfgs_history_size}.jpg"
img_path = os.path.join(os.path.dirname(csv_file),name)
plt.savefig(img_path)
plt.show()
print(f"Plot saved to {img_path}")
