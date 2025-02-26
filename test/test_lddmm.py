import pytest


import os
import csv
import torch

from src.demeter import *

import src.demeter.utils.torchbox as tb
import matplotlib.pyplot as plt
import src.demeter.utils.bspline as bs
import src.demeter.utils.vector_field_to_flow as vff
import src.demeter.utils.reproducing_kernels as rk

# %load_ext autoreload
# %autoreload 2
import src.demeter.metamorphosis as mt

plot = False

#####################
#  UTILS
#####################

import string
import random


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


csv_file = os.path.join(OPTIM_SAVE_DIR, DEFAULT_OPTIM_CSV)

# make wrapper around check_csv to catch the FileNotFoundError
def catch_file_not_found_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError:
            os.mkdir(OPTIM_SAVE_DIR)
            with open(csv_file, "w", newline="") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(
                    DEFAULT_CSV_HEADER
                )
            return func(*args, **kwargs)

    return wrapper

@catch_file_not_found_error
def check_csv(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file, delimiter=";")
        for i, line in enumerate(reader):
            if i == 0:
                first_line = line
            pass
        last_line = line
        n_lines = i + 1

    return first_line, last_line, n_lines


def remove_saved_files_and_csv_entry(saved_file):
    # Lire le fichier CSV et stocker les lignes
    file_path = os.path.join("saved_optim", saved_file)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Fichier supprimé : {file_path}")

    # Supprimer la dernière entrée du fichier CSV
    with open(csv_file, "r") as file:
        reader = csv.reader(file, delimiter=";")
        lines = list(reader)
    lines = lines[:-1]
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        for line in lines:
            writer.writerow(line)


#####################
# END  UTILS
#####################


@pytest.fixture()
def setup_lddmm():
    ic(ROOT_DIRECTORY)
    size = (100, 100)
    source = tb.reg_open("20", size=size)  # .to('cuda:0')

    # build a spline deformation
    # cms = 4 * torch.randn((2, 8, 8))
    cms = torch.tensor(
        [
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.3908, -2.1598, 1.8110, 3.0516, -2.0448, -1.3106, 0.0000],
                [0.0000, -3.4693, -2.9383, 0.2399, 3.2613, -9.0656, 6.6554, 0.0000],
                [0.0000, 7.4708, 5.1128, 7.4947, 2.8691, -0.6619, -0.7874, 0.0000],
                [0.0000, -1.3023, 1.7911, 1.8349, -0.0542, 2.6127, -5.3047, 0.0000],
                [0.0000, -2.8034, -3.0380, 1.4516, -1.8438, -0.4917, -3.4913, 0.0000],
                [0.0000, 5.1222, 0.8622, -1.2726, 0.6390, 7.1037, 0.3399, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, -1.0876, -0.9888, 2.3738, -5.8171, 0.3255, -0.2233, 0.0000],
                [0.0000, 0.1378, -5.6583, 1.2368, 0.6751, 5.0485, -0.4347, 0.0000],
                [0.0000, -5.6404, -10.1700, -2.9721, -0.9263, -0.5875, 3.0433, 0.0000],
                [0.0000, -6.2936, -4.2457, -0.6589, -13.0326, -4.9987, 12.9451, 0.0000],
                [0.0000, -2.7168, 0.8551, -4.4050, -5.0932, -3.1117, 6.3056, 0.0000],
                [0.0000, 2.7018, 0.8891, -1.7447, -0.7347, 1.0819, 1.5808, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
        ]
    )

    field = bs.field2D_bspline(cms, size, degree=(2, 2), dim_stack=-1)[None]
    deformation = vff.FieldIntegrator(method="fast_exp")(field.clone(), forward=True)
    deformator = vff.FieldIntegrator(method="fast_exp")(field.clone(), forward=False)

    # landmarks
    landmarks_source = torch.tensor([[40, 50], [70, 65], [70, 35], [50, 50]])

    landmarks_target = deformation[0, landmarks_source[:, 1], landmarks_source[:, 0]]

    # apply the deformation
    target = tb.imgDeform(source, deformator, dx_convention="pixel")

    # ==========
    if plot:
        # tb.deformation_show(deformation, color="black", step=5)

        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(source[0, 0].cpu(), cmap="gray", origin="lower")
        ax[0].plot(landmarks_source[:, 0], landmarks_source[:, 1], "ro")
        ax[1].imshow(target[0, 0].cpu(), cmap="gray", origin="lower")
        ax[1].plot(landmarks_target[:, 0], landmarks_target[:, 1], "bo")
        ax[2].imshow(tb.imCmp(source, target, "seg"), cmap="gray", origin="lower")
        plt.show()
    # ==========

    kernelOperator = rk.GaussianRKHS(sigma=(3, 3), normalized=True)

    #
    mr = mt.lddmm(
        source,
        target,
        momentum_ini=0,
        kernelOperator=kernelOperator,
        cost_cst=1e-5,
        integration_steps=10,
        n_iter=5 if plot else 2,
        grad_coef=10,
        optimizer_method="LBFGS_torch",
        hamiltonian_integration=True,
        dx_convention="square"
    )
    if plot:
        mr.plot()
        plt.show()
    return mr, landmarks_source, landmarks_target


# #%% # debug
# mr, landmarks_source, landmarks_target = setup_lddmm()
# #%%
# new_file = mr.save('new_test','lddmm',light_save=True)
#
# # %%
#
# file_name = new_file[0]
# mr_2 = mt.load_optimize_geodesicShooting(file_name)
# #%%
# mr_2.plot()


# %%
def test_landmark_computation(setup_lddmm):
    mr, landmarks_source, landmarks_target = setup_lddmm
    mr_def = mr.mp.get_deformation()
    if mr.dx_convention== "square":
        mr_def = tb.square_to_pixel_convention(mr_def,is_grid=True)
    elif mr.dx_convention == "2square":
        mr_def = tb.square2_to_pixel_convention(mr_def,is_grid=True)

    landmarks_reg = mr_def[0, landmarks_source[:, 1], landmarks_source[:, 0]]
    landmarks_reg_2, land_dist, _ = mr.compute_landmark_dist(
        landmarks_source, landmarks_target
    )

    my_dist = (landmarks_reg - landmarks_target).abs().mean()

    assert torch.abs(my_dist - land_dist) < 1e-6, (
        "There might be a problem with the landmark computation:"
        f"mr.dist = {land_dist} and dist computed here is {my_dist}"
    )

    if plot:
        fig, ax = plt.subplots()
        # ax.imshow(target[0, 0].cpu(), cmap="gray",origin='lower')
        ax.imshow(tb.imCmp(mr.mp.image, mr.target, "seg"), cmap="gray", origin="lower")
        ax.plot(landmarks_target[:, 0], landmarks_target[:, 1], "bo", label="target")
        ax.plot(landmarks_reg[:, 0], landmarks_reg[:, 1], "go", label="registered")
        ax.plot(
            landmarks_reg_2[:, 0], landmarks_reg_2[:, 1], "yo", label="registered_2"
        )
        ax.plot(landmarks_source[:, 0], landmarks_source[:, 1], "ro", label="source")
        plt.legend()
        plt.show()


# %%
def test_get_all_parameters(setup_lddmm):
    mr, landmarks_source, landmarks_target = setup_lddmm

    params = mr.get_all_arguments()
    print(params.keys())

    assert set(params.keys()) == {
        'n_step',
        'cost_cst',
        'kernelOperator',
        'hamiltonian_integration',
        'dx_convention',
        'rho',
        'method'
    }, f"The keys of the params dictionary do not match the expected keys. got: {params.keys()}"


def test_check_csv_before_save():
    first_line, last_line_b, n_lines_b = check_csv(csv_file)
    assert first_line == DEFAULT_CSV_HEADER, f"first line of csv is : {first_line}"


def test_check_csv_after_save(setup_lddmm):
    mr, landmarks_source, landmarks_target = setup_lddmm

    rdm_str = id_generator()

    first_line, last_line_b, n_lines_b = check_csv(csv_file)
    file_name, file_path = mr.save("test_lddmm", message=rdm_str)
    first_line, last_line_a, n_lines_a = check_csv(csv_file)

    print("number of columns:", len(last_line_a))
    assert n_lines_a == n_lines_b + 1
    assert last_line_a[2] == "2D"
    assert last_line_a[4] == "Metamorphosis_Shooting"
    assert last_line_a[-1] == rdm_str

    remove_saved_files_and_csv_entry(os.path.join(file_path, file_name))


@pytest.mark.parametrize("light_save", [True, False])
def test_save_load(setup_lddmm, light_save):
    mr, landmarks_source, landmarks_target = setup_lddmm

    file_name, file_path = mr.save("test_lddmm_delete_me", light_save=light_save)



    mr_2 = mt.load_optimize_geodesicShooting(file_name)
    fig_ax_c, fig_ax_i = mr_2.plot()
    fig_c, ax_c = fig_ax_c
    fig_i, ax_i = fig_ax_i
    if plot:
        plt.show()
    assert isinstance(fig_c, plt.Figure), f"type fig_c {fig_c}"
    assert ax_c is not None, f"type ax_c {ax_c}"
    assert isinstance(fig_i, plt.Figure), f"type fig_i {fig_i}"
    assert ax_i is not None, f"type ax_i {ax_i}"

    assert isinstance(
        mr_2.mp.kernelOperator, torch.nn.Module
    ), f"type {type(mr_2.mp.kernelOperator)} should be a torch.nn.Module"

    mr_args = mr.get_all_arguments()
    mr_2_args = mr_2.get_all_arguments()
    for key in mr_args.keys():
        print(f"key: {key}, mr: {mr_args[key]}, mr_2: {mr_2_args[key]}")
    assert mr_args == mr_2_args

    remove_saved_files_and_csv_entry(os.path.join(file_path, file_name))




# %%
"""
csv_file = "saved_optim/saves_overview.csv"


# def get_last_line(csv_file):
def check_csv(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file, delimiter=";")
        for i, line in enumerate(reader):
            if i == 0:
                first_line = line
            pass
        last_line = line
        n_lines = i + 1

    return first_line, last_line, n_lines


first_line, last_line_b, n_lines_b = check_csv(csv_file)
print(first_line)
file_name, file_path = mr.save("test", "lddmm")
first_line, last_line_a, n_lines_a = check_csv(csv_file)
print(last_line_a)

assert first_line == [
    "time",
    "saved_file_name",
    "source",
    "target",
    "n_dim",
    "shape",
    "meta_type",
    "data_cost",
    "kernelOperator",
    "optimizer_method",
    "final_loss",
    "DICE",
    "landmarks",
    "rho",
    "lamb",
    "n_step",
    "n_iter",
    "message",
]
assert n_lines_a == n_lines_b + 1
assert last_line_a[2] == "test"
assert last_line_a[6] == "Metamorphosis_Shooting"
# print(get_last_line(path_save_optim))

# get last line in the saved csv file

# %%
mr_2 = mt.load_optimize_geodesicShooting(file_name)


fig_c, ax_c, fig_i, ax_i = mr_2.plot()
plt.show()
assert isinstance(fig_c, plt.Figure)
assert isinstance(ax_c, plt.Axes)
assert isinstance(fig_i, plt.Figure)
assert isinstance(ax_i, plt.Axes)
"""
