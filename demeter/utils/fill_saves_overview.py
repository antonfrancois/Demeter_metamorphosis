#!/usr/bin/python
# import sys
# from datetime import datetime
import re
import csv
import metamorphosis as mt
from utils.constants import *

DEFAULT_CSV_FILE,DEFAULT_PATH = 'saves_overview.csv', ROOT_DIRECTORY+'/saved_optim/'

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
            source= re.search(r'.+?(?=_to_)',file_name[14:]).group(),                   # file name that is saved with the convention :
            target = re.search(r'(?<=_to_).*?(?=_(\d\d\d).pk1)',file_name[14:]).group(), # {n_dim}_{time}_{source}_to_{target}_{\d\d\d}.pk1
            n_dim = file_name[:2]
        )
    else: write_dict['saved_file_name'] = file_name

    # flag for discriminating against different kinds of Optimizers
    try:
        isinstance(optim.mp.rf,mt.Residual_norm_function)
        modifier_str = optim.mp.rf.__repr__()
    except AttributeError:
        modifier_str = 'None'

    state_dict = dict(
            shape=optim.source.shape.__str__()[10:],
            modifier=modifier_str,
            method=optim.optimizer_method_name,
            final_loss=float(optim.total_cost.detach()),
            DICE = optim.get_DICE(),
            mu=optim._get_mu_(),
            rho=optim._get_rho_(),
            lamb=optim.cost_cst,
            sigma_v=optim.mp.sigma_v.__str__(),
            n_step=optim.mp.n_step,
            n_iter=len(optim.to_analyse[1]),
            message= '' if message is None else message
        )
    return  {**write_dict , **state_dict}

def _write_dict_to_csv(dict,csv_file = DEFAULT_CSV_FILE):
    with open(DEFAULT_PATH+csv_file,mode='a') as csv_f:
        writer = csv.DictWriter(csv_f,dict.keys(),delimiter=';')
        # writer.writeheader()
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
