"""
In this file we perform a benchmark tracking execution time and memory usage with
or without the option for saving gpu memory.

This file execute the program `examples/3_utils/execute_meta.py`

By defauft I provide the results for my laptop. You can change the path of
`csv_file` to try on your computer.

"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import csv
import os, ast
import pandas as pd
from demeter.constants import ROOT_DIRECTORY
from demeter.utils.toolbox import convert_bytes_size
import torch.cuda as cda
from math import prod
import subprocess

env = os.environ.copy()
env["MKL_THREADING_LAYER"] = "GNU"
SIMPLEX = False
if SIMPLEX:
    name_csv = "understanding_memory_simplex.csv"
else:
    name_csv = "understanding_memory_classic.csv"
    # name_csv = "understanding_memory_results_archive.csv"



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
    print(existing_df["img shape"])
    existing_df['img shape'] = existing_df['img shape'].apply(ast.literal_eval)

else:
    existing_df = pd.DataFrame(columns=[
                "img shape",
                "image mem size",
                "save gpu",
                "n_iter",
                "n_step",
                "lbfgs_history_size",
                "lbfgs_max_iter",
                "memory allocated",
                "memory reserved",
                "exec time sec",
            ])
    existing_df.to_csv(csv_file, index=False)

if SIMPLEX:
    size_list = [80,120,140, 200]
else:
    size_list = [200, 282, 400, 750]
save_gpu_list = [True, False]
n_iter_list = [2,10,15]
n_step_list = [3,5,7,10,12]
lbfgs_history_size_list = [10,20, 50]
lbfgs_max_iter_list = [5,10]




#%%
param_name = ["size" ,"save_gpu", "n_iter", "n_step", "lbfgs_history_size", "lbfgs_max_iter",]
params = [
    {n:pu for pu,n in zip(p,param_name)}
    for p in itertools.product(
        size_list, save_gpu_list, n_iter_list, n_step_list, lbfgs_history_size_list, lbfgs_max_iter_list
    )
]


for c, p in enumerate(params):
    size = p["size"]
    save_gpu = p["save_gpu"]
    n_iter = p["n_iter"]
    n_step = p["n_step"]
    lbfgs_history_size = p["lbfgs_history_size"]
    lbfgs_max_iter = p["lbfgs_max_iter"]

    # Vérifie si cette config est déjà présente
    already_done = (
        (existing_df["img shape"] == (1, 1, size, size)) &
         (existing_df["save gpu"] == save_gpu) &
        (existing_df["n_iter"] == n_iter) &
        (existing_df["n_step"] == n_step) &
        (existing_df["lbfgs_history_size"] == lbfgs_history_size) &
        (existing_df["lbfgs_max_iter"] == lbfgs_max_iter)
    ).any()

    if SIMPLEX:
        exec_file = "examples/3_utils/execute_simplex_pixyl.py"
    else:
        exec_file = "examples/3_utils/execute_meta.py"

    print(f"\nLancement {c+1}/{len(params)}: size=({(size, size, 1)}), save_gpu={save_gpu}")
    print("python3 ",
        exec_file,
        f"\n\t--width {size}" 
        f"\n\t--height {size}" 
        f"\n\t--save_gpu {save_gpu}"
        f"\n\t--n_iter {n_iter}" 
        f"\n\t--n_step {n_step}" 
        f"\n\t--lbfgs_history_size {lbfgs_history_size}" 
        f"\n\t--lbfgs_max_iter {lbfgs_max_iter}" 
        f"\n\t--csv_file {csv_file}"
    )
    if already_done:
        print(f"Déjà fait: → Ignoré.")
        continue

    # Sinon, on lance le benchmark

    subprocess.run([
        "python3",
        exec_file,
        "--width", str(size),
        "--height", str(size),
        "--save_gpu", str(save_gpu).lower(),
        "--n_iter", str(n_iter),
        "--n_step", str(n_step),
        "--lbfgs_history_size", str(lbfgs_history_size),
        "--lbfgs_max_iter", str(lbfgs_max_iter),
        "--csv_file", str(csv_file),
    ], check=True, env=env)


# Étape 4 : Plot
# Lecture des résultats
#%%
df = pd.read_csv(csv_file)
print(df.keys())
try:
    df['img shape'] = df['img shape'].apply(ast.literal_eval)
except ValueError:
    pass

# Nettoyage / transformation
# df["size"] = df["width"] * df["height"]  # on suppose que width = height
df["size"] = df["img shape"].apply(prod)
df["n_step * im_mem"] = df["n_step"] * df["image mem size"]

mem_to_consider = "memory allocated"
# mem_to_consider = "memory reserved"

df[mem_to_consider] = pd.to_numeric(df[mem_to_consider], errors='coerce')
df["exec time sec"] = pd.to_numeric(df["exec time sec"], errors='coerce')

df["M"] =  np.minimum( df["lbfgs_max_iter"]*df["n_iter"], df["lbfgs_history_size"] )
df["M1"] =  df["lbfgs_max_iter"]*df["n_iter"]
df["M1 > hist"] = df["M1"] > df["lbfgs_history_size"]


df["M * im_mem"]= df["M"] * df["image mem size"]

df_200 = df[df["img shape"] == (1,1,200,200)]

df_200 = df_200[df_200["n_iter"]>1]

#%%
from sklearn.linear_model import LinearRegression

f_sa_gpu = True
df_gpu = df[
    (df["save gpu"] == f_sa_gpu)
    # & (df_200["M"]> 20)
].dropna()
X = df_gpu[['M * im_mem',"n_step * im_mem"]]

# X = df_gpu[['lbfgs_history_size * im_mem',"n_step * im_mem"]]
y = df_gpu['memory allocated']
# y = df_gpu['memory reserved']


model = LinearRegression()
model.fit(X, y)

print(f'Prediction allocated memory (save gpu = {f_sa_gpu})')
print("Coefficients :", model.coef_)
print("Intercept    :", model.intercept_," , ",convert_bytes_size(model.intercept_))
print("Score R²     :", model.score(X, y))

a,b = model.coef_
c = model.intercept_
results = {
    'a': float(round(a, 8)),
    'b': float(round(b, 8)),
    'c': float(round(c)),  # en bytes
    'r2': float(round(model.score(X, y), 6)),
    'save_gpu': f_sa_gpu,
}

print(results)


#%%
# calcul de b gpu True
# m_ref = 60
# mask_10 = ((df_200["M"] == m_ref) &
#     (df_200["n_step"] == 10) &
#      (df_200["save gpu"] == True))
# mask_2 = ((df_200["M"] == m_ref) &
#     (df_200["n_step"] == 2) &
#      (df_200["save gpu"] == True))
# try:
#     D = df_200[mask_10]["image mem size"].item()
#     b_true = (df_200[mask_10][mem_to_consider].item() - df_200[mask_2][mem_to_consider].item()) / ((10 -2) * D)
# except ValueError:
#     if ((len(df_200[mask_10]["image mem size"].unique()) ==1)
#         and (len(df_200[mask_10][mem_to_consider].unique()) ==1)
#             and (len(df_200[mask_10]["image mem size"].unique())==1)):
#         D = df_200[mask_10]["image mem size"].to_list()[0]
#         b_true = (df_200[mask_10][mem_to_consider].to_list()[0] - df_200[mask_2][mem_to_consider].to_list()[0]) / ((10 -2) * D)
# # calcul de b gpu False
# mask_10 = ((df_200["M"] == m_ref) &
#     (df_200["n_step"] == 10) &
#      (df_200["save gpu"] == False))
# mask_2 = ((df_200["M"] == m_ref) &
#     (df_200["n_step"] == 2) &
#      (df_200["save gpu"] == False))
# try:
#     D = df_200[mask_10]["image mem size"].item()
#     b_false = (df_200[mask_10][mem_to_consider].item() - df_200[mask_2][mem_to_consider].item()) / ((10 -2) * D)
# except ValueError:
#     if ((len(df_200[mask_10]["image mem size"].unique()) ==1)
#         and (len(df_200[mask_10][mem_to_consider].unique()) ==1)
#             and (len(df_200[mask_10]["image mem size"].unique())==1)):
#         D = df_200[mask_10]["image mem size"].to_list()[0]
#         b_false = (df_200[mask_10][mem_to_consider].to_list()[0] - df_200[mask_2][mem_to_consider].to_list()[0]) / ((10
# print(f"b_true : {b_true}, b_false : {b_false}, M = {m_ref}")
#%%
fig, ax = plt.subplots(2, 2, figsize=(12, 10), constrained_layout = True)
fig.suptitle(f" {gpu_name}; Total Memory: {total_memory / (1024 ** 3):.2f} GB; img size : {df_200["img shape"].iloc[0]}")#, {D/1024**2:.2f}MB")
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_styles = ['-', '--', '-.',':',(0, (3, 5, 1, 5, 1, 5))]
markers = ['.','v', '<', '>', '^', 's' ,"*"]
m_unique = df_200["M"].unique()
i = 0
eps = .1*0
for m in m_unique:
    df_200_m = df_200[df_200["M"] == m].sort_values(by="n_step", ascending=True)
    for save_gpu in [True, False]:
        df_200_m_gpu = df_200_m[df_200_m["save gpu"] == save_gpu]
        # print(len(df_200_m_gpu))
        for ls, lh in enumerate(df_200_m_gpu["M1 > hist"].unique()):
            df_200_m_gpu_lh = df_200_m_gpu[df_200_m_gpu["M1 > hist"] == lh]
            n_ax = 1 if save_gpu else 0

            ax[0,n_ax].plot(df_200_m_gpu_lh["n_step"]+ i*eps,
                            df_200_m_gpu_lh[mem_to_consider], #- a* df_200_m_gpu_lh["M * im_mem"] - c,
                       label=f"M = {m}, {"M1 > hist" if lh else 'M1 <= hist'}",
                       # label=f"M = {m}" if ls == 0 else "",
                       #  linestyle='',
                       linestyle = line_styles[ls],
                        marker= markers[ls],
                       color=default_colors[i] if i < len(default_colors) else default_colors[i- len(default_colors)],
                       )
            ax[0,n_ax].set_xlabel("n integration step")
            ax[0,n_ax].set_ylabel("Mem usage bytes")
            ax[0,n_ax].legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0)
            ax[0,n_ax].set_title(f"save gpu : {save_gpu}")
    i +=1

for save_gpu in [True, False]:
    n_ax = 1 if save_gpu else 0
    x_name = "M"
    # x_name = 'lbfgs_history_size'
    crit = 'n_step'
    # crit_2 = "lbfgs_history_size"
    # crit_2 =
    crit_2 = "M1 > hist"
    # crit_2 =cor "n_iter"
    # crit_2 = "M"
    i = 0
    for pi, p in enumerate(df_200[crit].unique()):
        for ci, p2 in enumerate(df_200[crit_2].unique()):
            # df_200_pi = df_200[(df_200[crit] == p) ]
            df_200_pi = df_200[(df_200[crit] == p) & (df_200[crit_2] == p2)]
            df_200_pi_gpu = df_200_pi[df_200_pi["save gpu"] == save_gpu].sort_values(by=x_name, ascending=True)
            # if p2:
            ax[1,n_ax].plot(df_200_pi_gpu[x_name], df_200_pi_gpu[mem_to_consider],
                            marker=markers[pi],
                            label=f"{crit}:{p}, {crit_2 if not p2 else 'M1 <= hist'}",
                            linestyle = line_styles[ci],
                            color=default_colors[i]
                            )
            ax[1,n_ax].set_xlabel(x_name)
            ax[1,n_ax].set_ylabel("Mem usage bytes")
            ax[1,n_ax].legend(bbox_to_anchor=(1.05, 1),)

        i +=1
plt.show()
fig.savefig(csv_file[:-4] + ".png")

#%%
from mpl_toolkits.mplot3d import Axes3D

# Filtrer les lignes où save_gpu == False
df_filtered = df[
    (df['save gpu'] == False)
    & (df["img shape"] == (1,1,200,200))
    & (df["n_iter"] > 1)
    # & (df["n_step"] == 2)
]
color = df_filtered['M']

pivot_table = df_filtered.pivot_table(
    index='lbfgs_history_size',
    columns='M1',
    values='memory allocated'
)
# Extract grid values
X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
Z = pivot_table.values

# Ajouter une surface rouge semi-transparente là où M1 < lbfgs_history_size
mask = Y > X  # car Y = lbfgs_history_size, X = M1
Z_mask = np.where(mask, Z, np.nan)  # ne garder que les zones valides

# Créer la figure et l'axe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tracer les points
sc = ax.scatter(
    df_filtered['M1'],
    df_filtered['lbfgs_history_size'],
    df_filtered['memory allocated'],
    c=color, marker='o',cmap="jet"
)
highlight = df_filtered[df_filtered['M1'] < df_filtered['lbfgs_history_size']]
ax.scatter(
    highlight['M1'],
    highlight['lbfgs_history_size'],
    highlight['memory allocated'],
    facecolors='none', edgecolors='red', s=80, linewidths=1.5, label='M1 < hist'
)
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k',alpha=0.5)
ax.plot_surface(X, Y, Z_mask, color='red', alpha=0.3, linewidth=0, antialiased=False)

# Ajouter une barre de couleur
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('M')


# Ajouter les étiquettes des axes
ax.set_xlabel('M1')
ax.set_ylabel('lbfgs_history_size')
ax.set_zlabel('memory allocated (MB)')  # ajuste l'unité si besoin

# Titre
ax.set_title('Memory Allocated selon n_step et M (save_gpu = False)')

plt.tight_layout()
plt.show()
