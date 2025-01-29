import os

import matplotlib.pyplot as plt
import torch
import numpy as np


# ROOT_DIRECTORY = os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.abspath(__file__))
# ))

ROOT_DIRECTORY = os.getcwd()
while not ROOT_DIRECTORY.endswith("Demeter_metamorphosis"):
    ROOT_DIRECTORY = os.path.dirname(ROOT_DIRECTORY)
    if ROOT_DIRECTORY == '/': break
if not ROOT_DIRECTORY.endswith('Demeter_metamorphosis'):
    raise ValueError(f'ROOT_DIRECTORY should end with '
                     f'Demeter_metamorphosis, got {ROOT_DIRECTORY}')

# used in saving metamorphosis saving
OPTIM_SAVE_DIR = ROOT_DIRECTORY + '/saved_optim/'
FIELD_TO_SAVE = [
            'mp',
            'source', 'target', 'cost_cst', 'optimizer_method_name','data_term',
            'parameter','ssd', 'norm_v_2', 'norm_l2_on_z',
            'total_cost', 'to_analyse','dice'
        ]

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

