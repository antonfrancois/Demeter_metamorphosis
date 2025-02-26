import os

import matplotlib.pyplot as plt
import torch
import numpy as np
import appdirs
from dotenv import load_dotenv
import csv

from icecream import ic

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

ROOT_DIRECTORY = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

# ROOT_DIRECTORY = os.getcwd()
# while not ROOT_DIRECTORY.endswith("Demeter_metamorphosis"):
#     ROOT_DIRECTORY = os.path.dirname(ROOT_DIRECTORY)
#     if ROOT_DIRECTORY == '/': break
# if not ROOT_DIRECTORY.endswith('Demeter_metamorphosis'):
#     raise ValueError(f'ROOT_DIRECTORY should end with '
#                      f'Demeter_metamorphosis, got {ROOT_DIRECTORY}')


# ====================================================
# SAVE_DIR
# used in saving metamorphosis optimisations
# Define the default saving location with appdirs
default_save_dir = appdirs.user_data_dir("Demeter_metamorphosis", "antonfrancois")

# Utiliser la variable d'environnement si elle est définie, sinon utiliser le répertoire par défaut
OPTIM_SAVE_DIR = os.getenv('DEMETER_OPTIM_SAVE_DIR', default_save_dir)
# OPTIM_SAVE_DIR = ROOT_DIRECTORY + '/saved_optim/'
OPTIM_SAVE_DIR  = os.path.join( OPTIM_SAVE_DIR ,  "saved_optim/")
DEFAULT_OPTIM_CSV = 'saves_overview.csv'
DEFAULT_CSV_HEADER = [
        "time",
        "saved_file_name",
        "n_dim",
        "shape",
        "meta_type",
        "data_cost",
        "kernelOperator",
        "optimizer_method",
        "hamiltonian_integration",
        "dx_convention",
        "final_loss",
        "DICE",
        "landmarks",
        "rho",
        "lamb",
        "n_step",
        "n_iter",
        "message",
    ]
FIELD_TO_SAVE = [
            'mp',
            'source', 'target', 'cost_cst', 'optimizer_method_name','data_term',
            'parameter','ssd', 'norm_v_2', 'norm_l2_on_z',
            'total_cost', 'to_analyse','dice'
        ]

# Vérifier que le répertoire existe, sinon le créer
if not os.path.exists(OPTIM_SAVE_DIR):
    os.makedirs(OPTIM_SAVE_DIR)
    full_path = os.path.join(OPTIM_SAVE_DIR, DEFAULT_OPTIM_CSV)
    with open(full_path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            DEFAULT_CSV_HEADER
        )


def display_env_help():
    env_var_name = 'DEMETER_OPTIM_SAVE_DIR'
    current_value = os.getenv(env_var_name, 'Not defined')

    help_message = f"""
    The environment variable {env_var_name} is currently: {current_value}

    To set this environment variable, you can follow these steps:

    1. Open your terminal.
    2. Add the following line to your `.env` file at the root of the
    demeter project, which is located at:
        {os.path.dirname(os.path.abspath(__file__))}
        
    add the following line into the .env file, you can follow the env.example file
       {env_var_name}=/path/to/your/directory
       

    If you are using a shell like bash, you can also set this variable temporarily using:
       export {env_var_name}=/path/to/your/directory

    To verify that the variable is set, you can run the following command in your terminal:
       echo ${env_var_name}
    """

    print(help_message)


# # Call the function to display the help
# display_env_help()

# =====================================================
# OTHER CONSTANTS


# Default arguments for plots in images and residuals
DLT_KW_IMAGE = dict(cmap='gray',
                      # extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=0,vmax=1)
DLT_KW_RESIDUALS = dict(cmap='RdYlBu_r',
                      extent=[-1,1,-1,1],
                      origin='lower',
                      )#,
DLT_KW_GRIDSAMPLE = dict(padding_mode="border",
                         align_corners=True
                         )


# What info to keep for optimisation in files.
default_optim_csv = 'saves_overview.csv'

def get_freer_gpu():
    cuda = torch.cuda.is_available()
    if not cuda: return 'cpu'
    torch.autograd.set_detect_anomaly(False)
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = torch.tensor([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
    return 'cuda:'+str(int(torch.argmax(memory_available)))

def find_cuda(foo=True):
    if not foo: return 'cpu'
    device = get_freer_gpu()
    print('device used :',device)
    return device

# colors & plots misc.:
GRIDDEF_YELLOW = '#E5BB5F'
source_ldmk_kw = dict(
    marker='o',c='blue', label='source',markersize=15,linestyle = 'None'
)
deform_ldmk_kw = dict(
    marker='s',c='red', label='target',markersize=15,linestyle = 'None'
)
target_ldmk_kw = dict(
    marker='p',c='orange',label='deformed',markersize=15,linestyle = 'None'
)

def set_ticks_off(ax):
    try:
        ax.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                right=False,
                left=False,
                labelbottom=False, # labels along the bottom edge are off
                labelleft=False,
            )
    except AttributeError:
        for a in ax.ravel():
            set_ticks_off(a)

