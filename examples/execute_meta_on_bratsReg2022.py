import csv
import itertools
import os
from mailbox import Error

import nibabel as nib

import torch
# import torch
# import nibabel as nib
from datetime import datetime
import pandas as pd
from datetime import date

from demeter.constants import *
import demeter.utils.torchbox as tb
from demeter.utils.image_3d_plotter import imshow_3d_slider, Visualize_GeodesicOptim_plt
from demeter.utils.toolbox import get_size
from demeter.utils.decorators import time_it
import demeter.metamorphosis as mt
import demeter.utils.reproducing_kernels as rk

import brats_utils as bu


@time_it
def exe_lddmm(img_1,img_2,seg_1,seg_2):
    sigma = [3,7]
    lamb = 1e-7
    n_step = 5
    n_iter = 1000 if at_serv else 3
    mr = mt.lddmm(img_2,img_1,0,sigma,lamb,n_step,n_iter=n_iter,grad_coef=1,sharp=True)
    return mr

class ExeMeta():

    def __init__(self,n_steps = None,
                 rho = None,
                 lamb = None,
                 kernelOperator =  None,
                 sharp = None,
                 n_iter=None,
                 dx_convention = None,
                 ):
        self.n_steps = 10   if n_steps is None else n_steps
        self.lamb = 1e-7    if lamb is None else lamb
        self.rho = 10  if rho is None else rho
        self.sharp = True   if sharp is None else sharp
        self.n_iter = 1000 if n_iter is None else n_iter
        if kernelOperator is None:
            sigma = [(3,3),(7,7)]
            self.kernelOperator = rk.Multi_scale_GaussianRKHS(
                sigma,
                normalized=False,
            )
        else:
            self.kernelOperator = kernelOperator
        self.dx_convention = "pixel" if dx_convention is None else dx_convention

    def __call__(self,img_1,img_2,seg_1,seg_2):
        n_iter = self.n_iter if at_serv else 3
        mr = mt.metamorphosis(img_2,img_1,0,
                              self.rho,self.lamb,self.n_steps,
                              kernelOperator=self.kernelOperator,
                              n_iter = n_iter,grad_coef=.1,sharp=self.sharp,
                             dx_convention=self.dx_convention
                              )
        return mr


class ExeCM():

    def __init__(self,n_steps = None,
                 rho = None,
                 p = None,
                 lamb = None,
                 # sigma = None,
                 kernelOperator = None,
                 sharp = None,
                 n_iter=None,
                 dx_convention = 'pixel'
                 ):
        self.n_steps = 10   if n_steps is None else n_steps
        self.lamb = 1e-7    if lamb is None else lamb
        if p is None:
            self.rho = 10  if rho is None else rho
            self.p = None
        else:
            self.p = p
            self.rho = self.p*self.mu
        # self.gamma = self.rho*self.k if self.mu == 1 else self.p*self.k
        # self.sigma = [(3,3,3),(7,7,7)]     if sigma is None else sigma
        self.kernelOperator = kernelOperator
        self.sharp = True   if sharp is None else sharp
        self.n_iter = 1000 if n_iter is None else n_iter
        self.flag_mr_mask = False
        self.dx_convention = dx_convention

        print("|||||||||||| ExeCM |||||||||||||||||||||||")
        print(f"n_steps = {self.n_steps}, "
        f"lamb = {self.lamb},"      
        # f"mu = {self.mu}, "      
        f"rho = {self.rho}, " 
        # f"gamma = {self.gamma}, " 
        f"p = {self.p}, " 
        f"sharp = {self.sharp}. "
        f"n_iter = {self.n_iter}"
        f"dx_convention = {self.dx_convention}"
              )

    def set_mr_mask(self,mr_mask):
        self.flag_mr_mask = True
        self.mr_mask = mr_mask

    @time_it
    def __call__(self,img_1,img_2,seg_1,seg_2):
        # Parameters
        # n_steps = self.n_steps
        # gamma = self.gamma
        sharp = self.sharp
        sharp = False

        # seg_1_2 = tb.imCmp(seg_1,seg_2, 'compose')
        # imshow_3d_slider(seg_1_2,title= "Seg 1 and 2")
        # plt.show()
        # Make temporal masks
        seg_1_1 = tb.imCmp(seg_1,seg_1, 'compose')
        imshow_3d_slider(seg_1_1)
        val_o,val_n = .5,1
        if not self.flag_mr_mask:
            print("\n mask not found => Computing ...")
            n_iter = 10 #if at_serv else 3

            kwargs_meta = dict(
                 momentum_ini=0,
                rho=1,
                kernelOperator=self.kernelOperator,
                cost_cst=0.1,
                integration_steps=self.n_steps, # if at_serv else 3,
                n_iter=n_iter,
                grad_coef=.1,
                sharp=False,
                save_gpu_memory=True
            )

            # -------------------------------------------------------
            # mr weighted
            seg_resi_1 = torch.zeros_like(seg_1)
            seg_resi_2 = torch.zeros_like(seg_2)
            e = 2e-1
            seg_resi_1[torch.logical_and(seg_1 <= val_o + e, seg_1 >= val_o-e)] = val_o
            seg_resi_2[torch.logical_and(seg_2 <= val_o + e, seg_2 >= val_o-e)] = val_o

            # seg_1_2 = tb.imCmp(seg_resi_1,seg_resi_2, 'compose')
            # imshow_3d_slider(seg_1_2,title= "Seg 1 and 2, after")
            # plt.show()

            print("Mask weighted:")
            mr_weighted = mt.metamorphosis(
                    seg_resi_2.to(device),
                    seg_resi_1.to(device),
                **kwargs_meta
            )

            # -------------------------------------------------------
            # mr oriented
            seg_oriented_1 = torch.zeros_like(seg_1)
            seg_oriented_2 = torch.zeros_like(seg_2)
            e = 2e-1
            seg_oriented_1[seg_1 >= val_n-e] = val_n
            seg_oriented_2[seg_2 >= val_o-e] = val_n

            # seg_1_2 = tb.imCmp(seg_oriented_1,seg_oriented_2, 'compose')
            # imshow_3d_slider(seg_1_2,title= "Seg 1 and 2, after")
            # plt.show()
            print("Mask oriented:")
            mr_oriented = mt.metamorphosis(seg_oriented_2.to(device),seg_oriented_1.to(device),**kwargs_meta)

            ic.disable()
            # mr_oriented.plot_cost()
            # Visualize_GeodesicOptim_plt(mr_oriented, 'masks')
            # plt.show()

            # mr_mask.save('mask_exeCM')
        else:
            mr_mask = self.mr_mask


        # raise Error("Arrete toi ici programme")

        mask_o = mr_weighted.mp.image_stock.clone()
        mask_o[mask_o > val_o ] = 1

        mask_n = mr_oriented.mp.image_stock.clone()
        mask_n[mask_n <= val_o + .05] = 0
        mask_n[mask_n > 0] = 1

        orienting_field = mr_oriented.mp.field_stock.to(device)
        # norm_2_om = (orienting_field**2).sum(axis=-1).sqrt().unsqueeze(1)
        # norm_2_om = norm_2_om/norm_2_om.max()
        orienting_mask = mask_n.to(device)
        weigthed_mask = mask_o.to(device)

        # Make computation
        residuals = torch.zeros(img_2.shape,device=device)
        # residuals = mr_sl_m.to_analyse[0]
        residuals.requires_grad = True

        mr_cm = mt.constrained_metamorphosis(img_1, img_2, 0,
                                     orienting_mask,
                                     orienting_field,
                                     weigthed_mask,
                                     kernelOperator=self.kernelOperator,
                                     cost_cst=1e-10,
                                     grad_coef=.1,
                                    n_iter=self.n_iter if at_serv else 3,
                                     dx_convention=self.dx_convention,
                                    save_gpu_memory=True
                                        # optimizer_method='adadelta',
                                     )
        self.flag_mr_mask = False
        return mr_cm

class ExeJoinedCM:

    def __init__(self,n_steps = None,
                 mu_I = None,
                 rho_I = None,
                 p_I = None,
                 mu_M = None,
                 rho_M = None,
                 p_M = None,
                 lamb = None,
                 sigma = None,
                 # sharp = None,
                 n_iter=None,
                 precompute_mask = False):
        self.n_steps = 10   if n_steps is None else n_steps
        self.mu_I = 1      if mu_I is None else mu_I
        self.mu_M = 0      if mu_M is None else mu_M
        self.lamb = 1e-7    if lamb is None else lamb
        if p_I is None:
            self.rho_I = 1  if rho_I is None else rho_I
            self.p_I = None
        else:
            self.p_I = p_I
            self.rho_I = self.p_I*self.mu_I
        if p_M is None:
            self.rho_M = 1  if rho_M is None else rho_M
            self.p_M = None
        else:
            self.p_M = p_M
            self.rho_M = self.p_M*self.mu_M
        self.sigma = [(3,3,3),(7,7,7)]     if sigma is None else sigma
        # self.sharp = True   if sharp is None else sharp
        self.n_iter = 1000 if n_iter is None else n_iter
        self.precompute_mask = precompute_mask

    def __call__(self,image_source,image_target,
                         mask_source,mask_target
                 ):
        n_iter = self.n_iter if at_serv else 3

        if self.precompute_mask:
            eps = .1
            # i3v.compare_3D_images_vedo(mask_source,mask_target)
            print("\n>> Precomputing mask")
            seg_oedeme_source = mask_source.clone()
            seg_oedeme_source[seg_oedeme_source > eps] = 1
            seg_oedeme_target = mask_target.clone()
            seg_oedeme_target[seg_oedeme_target > eps] = 1

            # mask necrosis
            mask_source = mask_source.clone()
            mask_source[mask_source < 1 - eps] = 0
            mask_target = mask_target.clone()
            mask_target[mask_target < 1 - eps] = 0

            # i3v.compare_3D_images_vedo(mask_source,mask_target)
            # i3v.compare_3D_images_vedo(seg_oedeme_source,seg_oedeme_target)

            mr_lddmm = mt.lddmm(seg_oedeme_source,seg_oedeme_target,0,
                                self.sigma,
                                self.lamb,
                                self.n_steps,
                                n_iter=400 if at_serv and not test else 3,
                                grad_coef=10,
                                sharp=True)
            mr_lddmm.save('mask','joinedCM')


            precomp_mask = mr_lddmm.mp.image_stock.detach()
            mask_function = mt.mask_sum(precomp_mask)
            print("Precomputed mask done")
        else:
            print("\n>> No precomputing mask")
            mask_function = None
        mr = mt.joined_metamorphosis(image_source,image_target,
                                        mask_source,mask_target,0,
                                        mu_I=self.mu_I,
                                        rho_I=self.rho_I,
                                        mu_M=self.mu_M,
                                        rho_M=self.rho_M,
                                        sigma=self.sigma,
                                        n_steps=self.n_steps,
                                        cost_cst=self.lamb,
                                        n_iter=n_iter,
                                        grad_coef=10,
                                        mask_function=mask_function,
                                     )
        return mr

class ExeMiniseg():

    def __init__(self,n_steps = None,mu = None,p = None,k = None,sigma = None, sharp = None):
        self.n_steps = 10   if n_steps is None else n_steps
        self.mu = 1         if mu is None else mu
        self.p = 10         if p is None else p
        self.k = .4         if k is None else k
        self.rho = self.p*self.mu
        self.sigma = [(3,3,3),(7,7,7)]     if sigma is None else sigma
        self.sharp = True   if sharp is None else sharp

        print("|||||||||||| ExeMiniseg |||||||||||||||||||||||")
        print(f"self.n_steps = {self.n_steps}, "
        f"self.mu = {self.mu}, " 
        f"self.p = {self.p}, " 
        f"self.k = {self.k}, "       
        f"self.rho = {self.rho}, " 
        f"self.sigma = {self.sigma}, " 
        f"self.sharp = {self.sharp}. "
              )

    @time_it
    def __call__(self,img_1,img_2,seg_1,seg_2):

        class cmp_field_ssd(mt.DataCost):
            def __init__(self,mask_N_S,mask_N_T,target):
                super(cmp_field_ssd, self).__init__()
                # self.cost_mask_O = mt.SimiliSegs(mask_O_S,mask_O_T)
                self.cost_mask_N = mt.SimiliSegs(mask_N_S,mask_N_T)
                self.ssd = mt.Ssd(target)

            def __repr__(self):
                return f"{self.__class__.__name__}"

            def set_optimizer(self, optimizer):
                super(cmp_field_ssd, self).set_optimizer(optimizer)
                self.ssd.set_optimizer(optimizer)
                self.cost_mask_N.set_optimizer(optimizer)

            def to_device(self,device):
                self.cost_mask_N.to_device(device)
                # super(cmp_field_ssd, self).to_device(device)
                self.ssd.to_device(device)

            def __call__(self):
                # cost_O = self.cost_mask_O()
                cost_N = self.cost_mask_N()
                ssd = self.ssd()
                return cost_N + ssd

        n_iter = 400 if at_serv else 3
        seg_all_1 = seg_1.clone()
        seg_all_2 = seg_2.clone()
        seg_all_1[seg_1> 0] = 1
        seg_all_2[seg_2> 0] = 1

        n_iter = 500 if at_serv else 3

        mr_mask = mt.metamorphosis(seg_all_2, seg_all_1, 0, mu=1, rho=5,
                                   sigma=[3,6], cost_cst=.0001,
                                   integration_steps=self.n_steps if at_serv else 3,
                                   n_iter=n_iter, grad_coef=1,
                                   sharp=True)
        mr_mask.to_device('cpu')
        mr_mask.save('mask','similiseg')
        mask_w = mr_mask.mp.image_stock.clone()

       # Make computation
        residuals = torch.zeros(img_2.shape,device=device)
        # residuals = mr_sl_m.to_analyse[0]
        residuals.requires_grad = True

        seg_n_1,seg_n_2 = seg_1,seg_2
        seg_n_1[seg_n_1 <1 ] = 0
        seg_n_2[seg_n_2 < 1] = 0
        data_cost = cmp_field_ssd(seg_n_2,seg_n_1,img_1)
        rf = mt.Residual_norm_identity(mask_w.to(device),self.mu,self.rho)
        mp_meta = mt.Constrained_meta_path(residual_function=rf,
                                           sigma_v=self.sigma,
                                           sharp=self.sharp
                                        # n_step=20 # n_step is defined from mask.shape[0]
                                        )
        mr_meta = mt.Constrained_Optim(img_2,img_1,mp_meta,
                                            cost_cst=.0001,
                                            data_term=data_cost,
                                            # optimizer_method='LBFGS_torch')
                                            optimizer_method='adadelta')
        n_iter = 1500 if at_serv else 3
        mr_meta.forward_safe_mode(residuals,n_iter=n_iter,grad_coef=1)
        return mr_meta

class ExeMasks():

    def __init__(self,n_steps = None,mu = None,rho = None,sigma = None, sharp = None, lamb = None,n_iter=None):
        self.n_steps = 10   if n_steps is None else n_steps
        self.mu = 1         if mu is None else mu
        self.rho = 1        if rho is None else rho
        self.sigma = [3,7]     if sigma is None else sigma
        self.sharp = True   if sharp is None else sharp
        self.lamb = 1e-5    if lamb is None else lamb
        self.n_iter = 1000 if n_iter is None else n_iter

        print("|||||||||||| ExeMasks |||||||||||||||||||||||")
        print(f"self.n_steps = {self.n_steps}, "
        f"mu = {self.mu}, " 
        f"rho = {self.rho}, " 
        f"sigma = {self.sigma}, " 
        f"sharp = {self.sharp}, "
        f"lamb = {self.lamb}."
              )

    @time_it
    def __call__(self,img_1,img_2,seg_1,seg_2):
        n_iter = self.n_iter if at_serv else 3

        seg_1[seg_1 > 0] = 1
        seg_2[seg_2 > 0] = 1

        mr_mask = mt.metamorphosis(seg_2, seg_1, 0, mu=self.mu, rho=self.rho,
                                   sigma=self.sigma, cost_cst=self.lamb,
                                   integration_steps=self.n_steps if at_serv else 3,
                                   n_iter=n_iter, grad_coef=.05,
                                   sharp=True)
        seg_2[seg_2 > 0] = 1
        seg_1[seg_1 > 0] = 1
        mr_mask.compute_DICE(seg_2.cpu(),seg_1.cpu())
        print(f"DICE : {mr_mask.get_DICE()}")
        return mr_mask

def init_save_landmark():
    save_path =  ROOT_DIRECTORY + "/examples/results/"
    date_time = datetime.now()

    folder_name = 'bratsReg2022_'+date_time.strftime("%Y%m%d_%H%M")+'/'
    save_path = os.path.join(save_path,folder_name)
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass

    return save_path

def save_valid_landmarks(mr,brats_name,save_path,scale_img,message=''):
    ldmk = mr.deform_landmark
    save_path = save_path+'/'
    mr.save(brats_name,'valid',destination=save_path,file='optim_overview.csv',message=message)

    with open(save_path+brats_name+'.csv',mode='w') as csv_l:
        writer = csv.writer(csv_l)
        writer.writerow(['Landmark','X','Y','Z'])
        for i,l in enumerate(ldmk):
            ll = [
                i+1,
                int((l[2]/scale_img).round()),
                -int((239 - l[1]/scale_img).round()),
                int((l[0]/scale_img).round())
            ]
            writer.writerow(ll)

    # save jacobian matrix
    if brats_name == 'BraTSReg_141':
        det_jaco = tb.checkDiffeo( mr.mp.get_deformation().cpu())
        det_jaco = tb.resize_image([det_jaco[None]],1/scale_img)
        print(f"\n Jacobian have {(det_jaco < 0).sum()} negatives values")
        nib.save(nib.Nifti1Image(det_jaco.numpy(),pb.affine),
                 save_path+'BraTSReg_141_detJacobian.nii.gz')


# def get_mask_info(file):
    # with open(file) as f:
    #     csv_reader = csv.DictReader(f, delimiter=';')
    #     mask_info = []
    #     for row in csv_reader:
    #         if row['saved_file_name'] != '':
    #             mask_info.append(
    #                 {'saved_file_name':row['saved_file_name'],
    #                  'n_steps':row['n_step']}
    #             )
    #             # brats_list.append(row['source'])
    #     return mask_info
    df_mask_info =  pd.load_csv(file)

def try_to_load_mask_optim(function,mask_info,brats_name):
    if not hasattr(function,'set_mr_mask'):
        return 0
    # try:
    #     mask_name = [mi["saved_file_name"] for mi in mask_info if brats_name in mi["saved_file_name"]][0]
    # except IndexError:
    #     return 0
    # choose a file index that contains the brats name and with the right n_step
    file_index = mask_info['saved_file_name'].str.contains(brats_name)
    file_index = file_index & (mask_info['n_step'] == function.n_steps)

    if file_index.sum() == 0:
        return 0
    mask_name = mask_info[file_index]['saved_file_name'].values[0]
    print(mask_name)
    mr_mask = mt.load_optimize_geodesicShooting(mask_name,verbose=False)
    print(f"\n mask found => {mask_name} Loaded")
    function.set_mr_mask(mr_mask)


def exe_i_list(function, pb,
               i_list,
               valid,
               modality,
               save_path = None,
               scale_img = .5,
               save_file=None,
               light_save=True,
               try_to_load_mask=True,
               device = 'cuda:0'):
    if save_path is None:
        save_path = init_save_landmark()
    elif save_path == 'results':
        save_path = ROOT_DIRECTORY + "/examples/results/bratsReg2022/"
    elif not ROOT_DIRECTORY in save_path:
        save_path = ROOT_DIRECTORY + "/examples/results/bratsReg2022/"+save_path+'/'
    if save_file is None: save_file = 'optim_overview.csv'
    print(f'\nResults will be saved at :{save_path+save_file}\n')

    if try_to_load_mask:
        # mask_info = get_mask_info(MASK_TO_USE_CSVFILE)
        df_mask_info =  pd.read_csv(MASK_TO_USE_CSVFILE,sep=';')
    print('i list', i_list, type(i_list))
    for n,i in enumerate(i_list):
        print(f"\n==== openning {i}:{pb.brats_list[i]}  {n+1}/{len(i_list)}======")
        print("scale_img : ",scale_img)
        #open images :
        img_1,img_2,seg_1,seg_2,landmarks = pb(i,to_torch=True,scale=scale_img,modality=modality)

        print('image shape :',img_1.shape)
        img_1 = img_1.to(device)
        seg_1 = seg_1.to(device)
        img_2 = img_2.to(device)
        seg_2 = seg_2.to(device)


        if try_to_load_mask:
            try_to_load_mask_optim(function,df_mask_info,pb.brats_list[i])
        # make optimisation
        mr = function(img_1,img_2,seg_1,seg_2)
        ic("mr computed")
        ic(get_size(mr))
        # if valid save landmarks in new file
        if valid:
            b_n = pb.brats_list[i]
            b_n += '_mask' if 'ExeMasks' in function.__class__.__name__ else ''

            mr.compute_landmark_dist(landmarks[1],landmarks[0],verbose=True)
            save_valid_landmarks(mr,b_n,save_path,scale_img)
        else:
            ic("Land_dist will be computed")
            _,landDist,landDistbefore = mr.compute_landmark_dist(landmarks[1],landmarks[0],verbose=True)
            landDistDiff = landDist - landDistbefore
            # save_valid_landmarks(mr,pb.brats_list[i],save_path+'_'+modality,scale_img)
            if 'mask' in save_file:
                file_name = pb.brats_list[i]+'Mask'+'_search_param'
            else:
                file_name = f'{pb.brats_list[i]}_train_{modality}'
            ic(light_save)
            ic("I will save",file_name,'at',save_path,save_file)
            mr.save(file_name,
                    light_save=light_save,
                    destination=save_path+'/',
                    file_csv=save_file,
                    message= f"{float(landDistDiff):0.3f}")

#%%
def rerun_valid_in_existing_folder(folder):
    save_path = ROOT_DIRECTORY + "/my_metamorphosis/results/"+folder
    file_in_save_path = os.listdir(save_path)
    brats_list = os.listdir(ROOT_DIRECTORY+'/../data/bratsreg_2022/BraTSReg_Validation_Data/')
    brats_list.remove('ReadMe_validation.txt')
    for f in file_in_save_path:
        print(f[:-4])
        if f[:-4] in brats_list:
            brats_list.remove(f[:-4])
    return brats_list

#%%

if __name__ == '__main__':
    at_serv = False
    # device = find_cuda()
    device = 'cuda:0'
    # device = 'cpu'
    torch.cuda.empty_cache()
    # if at_serv:
    #     import os
    #     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"

    test = False
    valid = False
    n_img_to_test = 3
    # scale_img = .8 if at_serv and not test else .15
    scale_img = .5
    light_save = True

    # brats_list = [
    # "BraTSReg_140",
    # # "BraTSReg_057",  # Le masque de segmentation n'est pas très bon
    # # "BraTSReg_083",
    # # "BraTSReg_078",
    # # "BraTSReg_014",
    # # "BraTSReg_060",
    # # "BraTSReg_070",
    # # "BraTSReg_134",
    # # "BraTSReg_129",
    # # "BraTSReg_058"
    # ]
    brats_list = [
        "BraTSReg_086",
        #"BraTSReg_090","BraTSReg_084",
        # "BraTSReg_046",
        # "BraTSReg_002","BraTSReg_021",
        # "BraTSReg_040","BraTSReg_118","BraTSReg_114","BraTSReg_132",

        # "BraTSReg_101","BraTSReg_073","BraTSReg_025","BraTSReg_022","BraTSReg_068","BraTSReg_120","BraTSReg_031","BraTSReg_088","BraTSReg_006","BraTSReg_003","BraTSReg_024","BraTSReg_035","BraTSReg_076","BraTSReg_012","BraTSReg_123",
    # 'BraTSReg_034',
        # 'BraTSReg_048', 'BraTSReg_055', 'BraTSReg_082', 'BraTSReg_045', 'BraTSReg_089', 'BraTSReg_057',
        # 'BraTSReg_096', 'BraTSReg_083', 'BraTSReg_042', 'BraTSReg_061', 'BraTSReg_074', 'BraTSReg_097', 'BraTSReg_056', 'BraTSReg_033', 'BraTSReg_136', 'BraTSReg_119', 'BraTSReg_108', 'BraTSReg_054', 'BraTSReg_091', 'BraTSReg_100', 'BraTSReg_030', 'BraTSReg_126', 'BraTSReg_133', 'BraTSReg_138', 'BraTSReg_053', 'BraTSReg_110', 'BraTSReg_079',
    # 'BraTSReg_008', 'BraTSReg_131', 'BraTSReg_001', 'BraTSReg_023', 'BraTSReg_064', 'BraTSReg_067', 'BraTSReg_115', 'BraTSReg_029', 'BraTSReg_093', 'BraTSReg_129', 'BraTSReg_005',
    #  'BraTSReg_140',
        #'BraTSReg_036', 'BraTSReg_071'
    ]

    # brats_list = None

    # to rerun in existing folder because something went wrong
    # # save_folder = "valid_bratsReg2022_20220722_2357_gpu3_CM_nstep5"
    # save_folder = "bratsReg2022_20220724_0127_gpu3"
    # if valid: brats_list = rerun_valid_in_existing_folder(save_folder)
    save_folder = None

    # MASK_TO_USE_CSVFILE = ROOT_DIRECTORY + "/my_metamorphosis/results/bratsReg2022_mask_to_use_meta.csv"
    MASK_TO_USE_CSVFILE = ROOT_DIRECTORY + "/examples/results/bratsReg2022_mask_to_use_lddmm.csv"

    brats_folder= '2022_valid' if valid else '2022_train'
    modality = 'flair'
    pb = bu.parse_brats(brats_list=brats_list,brats_folder=brats_folder,modality=modality)

    print(" -_ "*20)
    print(f"\n\n\nExecuting at date :{date.today()} at : {datetime.now().strftime('%H:%M:%S')}")
    print(f"BratS list is {len(pb.brats_list)} long")
    if valid:
        print('valid')
        # function = exe_lddmm
        # function = ExeCM(mu = 1,p = 5,k =.4,n_iter=2000,n_steps=5)
        # function = ExeMeta(rho=5)
        # function = ExeMasks(sigma=[3,7],rho=5,lamb=1e-7,n_iter=2000,n_steps=5)

        print("\n========Batch 1/4 =========")
        i_list = torch.arange(len(pb.brats_list))
        function = ExeCM(mu = 1,p = 5,n_iter=2000,n_steps=7,sigma=[3,7,20])
        exe_i_list(function,i_list,valid,modality,save_path=save_folder)
        # function = ExeCM(mu = 1,p = 5,k =.6,n_iter=2000,n_steps=5)
        # exe_i_list(function,i_list,valid,modality,save_path=save_folder)


        # print("\n|||||||||||||||||||||| FLAIR ||||||||||||||||||||||||||||||||\n")
        # exe_i_list(function,i_list,valid,'flair')
    else:
        # save_file = 'bratsReg2022_train_CM_paramSearch_optim_overview.csv'
        # save_file = 'bratsReg2022_train_Miniseg_paramSearch_optim_overview.csv'
        # save_file = 'bratsReg2022_mask_paramSearch_optim_overview.csv'
        # save_file = 'bratsReg_2022_param_search.py'
        # save_file = 'bratsReg2022_train_20230919_mask_newmodel_paramSearch_optim_overview.csv'
        # save_file = 'bratsReg2022_train_20230926_CM_newmodel_paramSearch_optim_overview.csv'
        # save_file = 'bratsReg2022_train_20231012_CM_newmodel_overview.csv'
        # save_file = 'bratsReg2022_train_20230924_LDDMM_newmodel_paramSearch_optim_overview.csv'
        # save_file = 'bratsReg2022_train_20240115_JM_twoResi_overview.csv'
        # save_file = 'bratsReg2022_train_20240201_JM_twoResi_twoMask_overview.csv'
        # save_file = 'bratsReg2022_train_20240201_flair_JM_twoResi_twoMask_overview.csv'
        # save_file = 'bratsReg2022_train_20240520_JWM_twoMask_overview.csv'
        save_file = "bratsReg2022_train_trash_JWM_overview.csv"
        save_file = "bratsReg2022_train_trash_LDDMM_overview.csv"
        save_file = "bratsReg2022_train_trash_CM_overview.csv"


        # save_file = MASK_TO_USE_CSVFILE.split('/')[-1]
        # save_file = 'bratsReg2022_train_20240115_JM_twoResi_overview.csv'

        print(f"img_size : {pb.get_img_size()}")
        resized_shape = (1,1)+ tuple(i * scale_img for i in pb.get_img_size()[2:])
        print("resized shape :",resized_shape)
        # for JM,we set mu_I = mu, and rho_I = rho as we want to set mu_M = rho_M = 0.
        # ps = [5,10,25,50]
        granularity_list = [
            #
            # [3,5],
            # [3,7],
            # [3,13],
            # [3,16],
            # [3,5,16],
            # [3,7,20],
            # [3,13,30],

            # [10,20,40]
            [10,20]
            # [10,20,40,50]
            # [10]
            ]
        resized_shape = (1,1)+ tuple(i * scale_img for i in pb.get_img_size()[2:])
        sigmas = rk.get_sigma_from_img_ratio(resized_shape,granularity_list, all_equal=True)
        ic(sigmas)

        kernelOperator = rk.Multi_scale_GaussianRKHS(
            sigmas[0],
            normalized=True
        )

        lambs = [1e-5]
        rhos = [(1,1)]
        # ks = [.5]
        n_iters = [25]# if not test else [3]
        n_steps = [10]
        precompute_mask = False

        param_name = ['rho','sigma','n_iter','n_steps','lamb']
        params = [
            {n:pu for pu,n in zip(p,param_name)}
            for p in itertools.product(rhos,sigmas,n_iters,n_steps,lambs )
        ]
        # print(params)


        i_list = torch.arange(len(pb.brats_list))
        # i_list  = torch.randperm(len(pb.brats_list))[:n_img_to_test]
        if 'LDDMM' in save_file:
            function = ExeMeta(rho=0,n_steps=15,n_iter=20, kernelOperator=kernelOperator,)
            exe_i_list(function, i_list, valid, modality, save_path='results', save_file=save_file)
        elif 'mask' in save_file:

            function = ExeMasks(sigma=sigmas,
                                  mu= 0,
                                  rho=0,
                                  lamb=0,
                                  n_steps=15,
                                  n_iter=3000)
            exe_i_list(function, i_list, valid, modality, save_path='results', save_file=save_file, try_to_load_mask=False)

        else:
            print(f"Number of parameters to test : {len(params)}")
            for count,p in enumerate(params):
                print(f"\n << Batch integration  {count+1} on {len(params)} >>")
                print(f"Parameters : {p}")
                if 'CM' in save_file:
                    print('CMM//§MCMCMCM')
                    function = ExeCM(
                           rho = p['rho'][0],
                            kernelOperator = kernelOperator,
                           lamb = p['lamb'],
                           n_iter=p['n_iter'],
                           n_steps=p['n_steps']
                    )

                elif 'JWM' in save_file: function = ExeJoinedCM(sigma = p['sigma'],
                                                       mu_I = p['mu'][0],
                                                       mu_M = p['mu'][1],
                                                       rho_I=p['rho'][0],
                                                       rho_M=p['rho'][1],
                                                       # rho_I = p['rho'][0],                                          ],
                                                       # rho_M = p['rho'][1],
                                                       lamb = p['lamb'],
                                                       n_iter = p['n_iter'],
                                                       n_steps = p['n_steps'],
                                                       precompute_mask=precompute_mask
                                                       )
                elif 'Miniseg' in save_file: function = ExeMiniseg(mu = p['mu'],
                                                                   k =p['k'],sigma=p['sigma'])

                else: function = ExeMeta(sigma=p['sigma'], rho=p['rho'],n_steps=p['n_steps'],n_iter=p['n_iter'])
                print(getattr(function, "__name__", str(function)))
                exe_i_list(function, pb,  i_list, valid, modality,
                           light_save = light_save,
                           save_path='results',
                           save_file=save_file,
                           try_to_load_mask=False
                           )
        # print("\n|||||||||||||||||||||| FLAIR ||||||||||||||||||||||||||||||||\n")
        #
        # i_list = indices = torch.randperm(len(pb.brats_list))[:n_img_to_test]
        # exe_i_list(function,valid,'flair')
        print("\n ======== End of execution ========\n")
        print(f"to get csv file :{save_file} copy paste the following line in a terminal :")
        print(f"sh shell/pull_csv_from_server.sh -l ipnp -f {save_file}")




