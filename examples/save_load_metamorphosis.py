import torch

import __init__
import metamorphosis as mt
import my_torchbox as tb
from constants import *
import fill_saves_overview as fso

#%%
print("Performing simple optimisation and saving it")
size = (200,200)
source_name,target_name = '01','02'
S = tb.reg_open(source_name,size = size).to('cpu') #,location='bartlett')
T = tb.reg_open(target_name,size = size).to('cpu')
residuals = torch.zeros(S.shape[2:])
mr = mt.metamorphosis(S,T,residuals,.03,1,5,0.0001,15,10,100)

mr.plot()
file,_ = mr.save(source_name,target_name,'Testing save')

mr_2 = mt.load_optimize_geodesicShooting(file)

#%%
print(f"Please Check that the file {file} was indeed written in the csv file. We will now try to write an externally written"
      f"file to the csv table. To do so"
      f", delete the good line in the csv table (should be the last) and press enter.")
_ = input('Press Enter to proceed')
#%%

fso.append_to_csv_new(file,"testing write csv")

