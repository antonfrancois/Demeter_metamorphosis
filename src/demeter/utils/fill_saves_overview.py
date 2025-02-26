#!/usr/bin/python
# import sys
# from datetime import datetime
import re
import csv

from .. import metamorphosis as mt
from demeter.constants import *

print(DEFAULT_OPTIM_CSV)
DEFAULT_PATH = OPTIM_SAVE_DIR

def rec(s):
    """remove_escape_characters"""
    return re.sub(r'[\n\t\r]', '', s)

def _optim_to_state_dict_(optim,file_name,write_dict=None,message=None):
    """ load and store needed variables from object inherited from Optimize_GeodesicShooting

    :param optim: child of Optimize_geodesicShooting
    :param file_name: (str) name of file to store
    :return: (dict) containing all relevant information to store.
    """

    if write_dict is None:
        write_dict = dict(
            time = file_name[3:13].replace('_','/'),                                              # All this informations are found in the
            saved_file_name = file_name,
            # source= re.search(r'.+?(?=_to_)',file_name[14:]).group(),                   # file name that is saved with the convention :
            # target = re.search(r'(?<=_to_).*?(?=_(\d\d\d).pk1)',file_name[14:]).group(), # {n_dim}_{time}_{source}_to_{target}_{\d\d\d}.pk1
            n_dim = file_name[:2]
        )
    else: write_dict['saved_file_name'] = file_name

    # flag for discriminating against different kinds of Optimizers
    # try:
    #     isinstance(optim.mp.rf,mt.Residual_norm_function)
    #     modifier_str = optim.mp.rf.__repr__()
    # except AttributeError:
    #     modifier_str = 'None'

    state_dict = {
        "shape": tuple(optim.source.shape),
        # "modifier": modifier_str,
        "meta_type": optim.__class__.__name__,
        "data_cost": optim.data_term.__class__.__name__,
        "kernelOperator": rec(optim.mp.kernelOperator.__repr__()),
        "optimizer_method": optim.optimizer_method_name,
        "hamiltonian_integration": optim.flag_hamiltonian_integration,
        "dx_convention": optim.dx_convention,
        "final_loss": float(optim.total_cost.detach()),
        "DICE": optim.get_DICE(),
        "landmarks": optim.get_landmark_dist(),
        "rho": optim._get_rho_(),
        "lamb": optim.cost_cst,
        "n_step": optim.mp.n_step,
        "n_iter": len(optim.to_analyse[1]),
        "message": '' if message is None else message
    }
    return  {**write_dict , **state_dict}

def _write_dict_to_csv(dict,csv_file = None,path=None):
    if csv_file is None: csv_file = DEFAULT_OPTIM_CSV
    if path is None: path = DEFAULT_PATH
    if not os.path.isdir(path):
        os.mkdir(path)
    wh = False if os.path.isfile(path+csv_file) else True
    with open(path+csv_file,mode='a') as csv_f:
        writer = csv.DictWriter(csv_f,dict.keys(),delimiter=';')
        if wh: writer.writeheader()
        writer.writerow(dict)
def append_to_csv_new(file_name,message=''):

    mr = mt.load_optimize_geodesicShooting(file_name)
    state_dict = _optim_to_state_dict_(mr,file_name,message=message)
    _write_dict_to_csv(state_dict)
    # with open(path+csv_file,m

def update_csv():

    # make list of all files listed in csv
    with open(DEFAULT_PATH+DEFAULT_CSV_FILE) as csv_f:
        csv_reader = csv.DictReader(csv_f,delimiter=';')
        file_list_csv = []
        for row in csv_reader:
            file_list_csv.append(row["saved_file_name"])

    # make list of all file in saved_optim
    file_list = []
    for f in os.listdir(DEFAULT_PATH):
        if  '.pk1' in f:
            file_list.append(f)

    for f in file_list:
        if not f in file_list_csv:
            print(f"\n Adding {f}")
            append_to_csv_new(f,'from update')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        file_name,message = sys.argv[1],sys.argv[2]
        append_to_csv_new(file_name,message)
    elif len(sys.argv) == 2:
        file_name = sys.argv[1]
        if 'update' in file_name:
            update_csv()
        else:
            append_to_csv_new(file_name)
    else:
        print("Usage : bad arguments")
