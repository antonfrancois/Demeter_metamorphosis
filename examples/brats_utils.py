import nrrd
import torch
import os
import numpy as np
import warnings
import csv
from skimage.exposure import match_histograms
from demeter.utils.torchbox import resize_image
from nibabel import load as nib_load


ROOT_DIRECTORY = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

# TODO : Test and document this file.

def nib_normalize(img,method='mean'):
    """
    MR Image intensity normalization. The method can be either 'mean' or 'min_max'

    Parameters
    ----------
    img : np.array (D,H,W)
        Image to normalize.
    method : str, optional
        Normalization method, by default 'mean'
    """
    if method == 'mean' or method is None:
        print("using mean")
        img = (img -img.mean())/(img.std() + 1e-30)
        img = np.clip(img,a_min=0,a_max=1)
    elif method == 'min_max':
        img += img.min()
        img /= img.max()
    else:
        raise ValueError(f"method must be 'mean' or 'min_max' got {method}")
    return img

def open_nib(folder_name,irm_type,data_base,format= '.nii.gz',normalize=True, to_torch =True):

    if data_base == 'brats':
        path = ROOT_DIRECTORY+ '/data/brats/'
    elif data_base == 'brats_2021':
        path = ROOT_DIRECTORY+ '/data/brats_2021/'
    elif data_base == 'bratsreg_2022':
        path = ROOT_DIRECTORY+'/data/bratsreg_2022/BraTSReg_Training_Data_v3/'
    else:
        path = data_base
    img_nib = nib_load(path+folder_name+'/'+folder_name+'_'+irm_type+format)
    img = img_nib.get_fdata()
    method = None
    if isinstance(normalize,str):
        method = normalize
        normalize = True
    if normalize:
        print(f"in open_nib() >I am normalizing {normalize}")
        img = nib_normalize(img,method=method)
    if to_torch: return torch.Tensor(img)[None,None]
    else: return img_nib


def open_template_icbm152(ponderation = 't1',normalize= True):
    path = ROOT_DIRECTORY+ '/data/template/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/'
    format = '.mgz' if ponderation == 'segV' else '.nii'
    file_name = 'mni_icbm152_'+ponderation+'_tal_nlin_asym_09c'+format
    img = nib_load(path+file_name).get_fdata()
    seg = nib_load(path+'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii').get_fdata()
    seg = seg/seg.max() # segmentation value is exaclty one ...
    # take off the skull and all non brain elements
    img[seg == 0] = 0
    method = None
    if isinstance(normalize,str):
        method = normalize
        normalize = True
    if normalize: img = nib_normalize(img,method=method)
    return torch.Tensor(img)[None,None]

class parse_brats:

    def __init__(self,brats_list=None,
                 template_folder=None,
                 brats_folder=None,
                 get_template=True,
                 modality='T1',
                 device = 'cpu'):
        """

        :param brats_list: list of stings containing the name of the folders
        :param template_folder: path to template folder
        :param brats_folder: path to brats db
        :param modality: modality of the IRM ex: `'T1'`,`'T2'`
        """
        self.flag_brats_2021 = False
        self.flag_bratsReg_2022 = False
        if template_folder is None:
            template_folder = ROOT_DIRECTORY+'/data/template/sri_spm8/templates/'
        if brats_folder is None or '2021' in brats_folder:
            self.brats_folder = ROOT_DIRECTORY+'/data/brats_2021/'
            self.flag_brats_2021 = True
        elif "2022_valid" in brats_folder:
            self.brats_folder = ROOT_DIRECTORY+'/data/bratsreg_2022/BraTSReg_Validation_Data/'
            self.flag_bratsReg_2022 = True
        elif "2022" in brats_folder:
            self.brats_folder = ROOT_DIRECTORY+'/data/bratsreg_2022/BraTSReg_Training_Data_v3/'
            # print(f"\n!!!!!! {self.flag_bratsReg_2022} <<<<<<<<\n")
            self.flag_bratsReg_2022 = True

        self.after_first_call = False

        # TODO : check that the list is correct by parsing with os.get_dir ...
        if brats_list is None:
            if self.flag_brats_2021:
                warnings.warn("It is not recommended to set brats_list to None with BraTS2021"
                              "database. It can lead to errors because ventricule segmentations "
                              "where not made for all data.")
            self._make_brats_list(self.brats_folder)
        else:
            self.brats_list = brats_list
        self.modality = modality
        self.device = device
        self.flag_get_template = get_template
        if not self.flag_bratsReg_2022 and get_template:
            template_nib = nib_load(template_folder+modality+"_brain.nii")
            self.template_affine = template_nib.affine
            self.template_img = template_nib.get_fdata()[:,::-1,:,0]

            self.template_seg = nib_load(template_folder+'seg_sri24.mgz').get_fdata()


    def get_img_size(self,scale = None):
        if self.flag_bratsReg_2022:
            if scale is None:
                scale = self.scale if self.after_first_call else 1
            img,_,_,_,_ = self._call_bratsReg_2022(0,True,scale)
            if scale == 1:
                return img.shape
            else:
                return tuple([int(max(s*scale,1)) for s in img.shape])
        elif self.flag_brats_2021:
            return self.template_img.shape

    def _make_brats_list(self,folder):
        self.brats_list = []
        for obj in os.listdir(folder):
            if self.flag_bratsReg_2022 and 'BraTSReg_' in obj:
                self.brats_list.append(obj)
            if self.flag_brats_2021 and 'BraTS2021' in obj:
                self.brats_list.append(obj)

    def get_template(self,normalised=True):
        if normalised:
            img_norm = (self.template_img - self.template_img.min()) / (self.template_img.max() - self.template_img.min())
            return torch.Tensor(img_norm,device=self.device)[None,None]
        else:
            return torch.Tensor(self.template_img.copy(),device=self.device)

    def get_template_vesi(self):
        vesi_seg = torch.zeros(self.template_seg.shape)
        vesi_seg[self.template_seg==4] = 1
        vesi_seg[self.template_seg==43] = 1
        return vesi_seg.flip(1)

    def get_template_whiteMatter(self):
        whmtr_seg = torch.zeros(self.template_seg.shape)
        whmtr_seg[self.template_seg==2] = 1
        whmtr_seg[self.template_seg==41] = 1
        return whmtr_seg.flip(1)

    def get_vesicule_seg(self,index,mask_correction=None):
        path = ROOT_DIRECTORY+ '/data/brats_2021/'
        brats_name = self.brats_list[index]
        brats_img_size = (240,240,155)
        vesi,header = nrrd.read(path+brats_name+'/'+brats_name+'_segV.seg.nrrd')
        space_origin = header['space origin']
        vesi_s = vesi.shape
        if vesi_s == brats_img_size: return torch.tensor( vesi)[None,None]

        vesi_pad = np.zeros(brats_img_size)
        vesi_pad[
        int(space_origin[0]):int(space_origin[0]+vesi_s[0]),
        int(space_origin[1]):int(space_origin[1]+vesi_s[1]),
        int(space_origin[2]):int(space_origin[2]+vesi_s[2])
        ] = vesi
        # vesi = open_nib(brats_name,'segV','brats_2021',
        #                   format='.nii.gz',normalize=False,to_torch=True)
        # img_1 = torch.zeros(vesi.shape)
        # img_1[vesi==4] = 1
        # img_1[vesi==43] = 1
        if not mask_correction is None:
            vesi[mask_correction>0] =0
        return torch.tensor( vesi_pad)[None,None]

    def _read_landmarks_csv_(self,file):
        with open(file) as csv_f:
            csv_reader = csv.DictReader(csv_f,delimiter=',')
            landmarks = []
            for row in csv_reader:
                try:
                    listv = [
                        float(row["Z"]),
                        239 + float(row["Y"]),
                        float(row["X"])
                    ]
                except KeyError:
                    listv = [
                        float(row[" Z"]),
                        239 + float(row[" Y"]),
                        float(row[" X"])
                    ]
                landmarks.append(listv)
        return torch.Tensor(landmarks)

    def _get_landmarks(self,path,file_list):
        file_ldk_1 = [f for f in file_list if '_01_' in f and '_landmarks.csv' in f][0]
        # print(file_ldk_1)
        ldk_1 = self._read_landmarks_csv_(path+file_ldk_1)
        if 'Training_Data' in path:
            file_ldk_0 = [f for f in file_list if '_00_' in f and '_landmarks.csv' in f][0]
            # print(file_ldk_0)
            ldk_0 = self._read_landmarks_csv_(path+file_ldk_0)
            return (ldk_0,ldk_1)
        return (None,ldk_1)

    def _call_brats_2021_(self,index,to_torch,normalize=False):
        brats_name = self.brats_list[index]
        self.after_first_call = True
        print("mean")
        gliom = open_nib(brats_name,self.modality.lower(),'brats_2021',normalize='min_max',to_torch=False)
        segmentation_tumor = open_nib(brats_name,'seg','brats_2021',normalize=False,to_torch=to_torch)
        if to_torch:
            segmentation_tumor[segmentation_tumor== 2] = .5
            segmentation_tumor[segmentation_tumor == 4] = 1
        else: segmentation_tumor = segmentation_tumor.get_fdata()
        # gliom = nib.load("/Users/maillard/Downloads/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/BraTS2021_00008/BraTS2021_000_T1.nii.gz")
        # histogram normalisation
        gliom_img = gliom.get_fdata()
        if self.flag_get_template and normalize:
            gliom_img[gliom_img > 0] = match_histograms(gliom_img[gliom_img > 0], self.template_img[self.template_img > 0])
            gliom_img = (gliom_img - gliom_img.min()) / (gliom_img.max() - gliom_img.min())
        self.scale = 1
        if to_torch:
            gliom_img = torch.Tensor(gliom_img,device=self.device)[None,None]
        return (gliom_img,
                segmentation_tumor)

    def _call_bratsReg_2022(self,index,to_torch,scale=0,rigidly_reg = False):
        """

        :param index: (int)
        :return: The two brains data to get.
        !!! Do not use rigidly_reg it does not work
        """
        path = self.brats_folder
        self.after_first_call = True
        self.scale = scale

        folder_name = self.brats_list[index]
        path += folder_name+'/'
        file_list = os.listdir(path)
        file_0 = [f for f in file_list if '_00_' in f  and self.modality.lower()+'.' in f][0]
        if rigidly_reg:
            file_1 = [
                f for f in file_list
                if '_01_' in f and self.modality.lower() in f and 'resampled' in f ][0]
        else:
            file_1 = [f for f in file_list if '_01_' in f  and self.modality.lower()+'.' in f][0]
        #print(file_0,file_1)
        img_nib_0 = nib_load(path + file_0)
        self.affine = img_nib_0.affine
        img_0 = img_nib_0.get_fdata()

        img_nib_1 = nib_load(path + file_1)
        img_1 = img_nib_1.get_fdata()

        # img_1[img_1 > 0] = match_histograms(img_1[img_1 > 0], img_0[img_0 > 0])
        img_0 = (img_0 - img_0.min()) / (img_0.max() - img_0.min())
        img_1 = (img_1 - img_1.min()) / (img_1.max() - img_1.min())
        # v_min, v_max = min(img_0.min(), img_1.min()), max(img_0.max(),img_1.max() )
        # v_min, v_max = img_0.min(),img_0.max()
        #
        # img_0 = (img_0 - v_min) / (v_max - v_min)
        # img_1 = (img_1 - v_min) / (v_max - v_min)



        landmarks = self._get_landmarks(path,file_list)

        # Segmentation !
        if 'Training_Data' in path:
            seg_path = ROOT_DIRECTORY+"/data/bratsreg_2022/Train_seg/"
        elif 'Validation' in path:
            seg_path = ROOT_DIRECTORY+"/data/bratsreg_2022/Valid_seg/"
        else:
            raise ValueError("Something went wrong.")
        seg_img_0 = nib_load(seg_path+folder_name+'_seg_00_.nii.gz').get_fdata()
        seg_img_0[seg_img_0 == 1] = 3
        seg_img_0[seg_img_0 == 2] = 1.5
        seg_img_0 = seg_img_0/3

        seg_img_1 = nib_load(seg_path+folder_name+'_seg_01_.nii.gz').get_fdata()
        seg_img_1[seg_img_1 == 1] = 3
        seg_img_1[seg_img_1 == 2] = 1.5
        seg_img_1 = seg_img_1/3

        # TODO : Check if we should normalize
        if to_torch:
            img_0 = torch.Tensor(img_0,device=self.device)[None,None]
            img_1 = torch.Tensor(img_1,device=self.device)[None,None]

            seg_img_0 =torch.Tensor(seg_img_0,device=self.device)[None,None]
            seg_img_1 =torch.Tensor(seg_img_1,device=self.device)[None,None]
            if scale != 1:
                img_0,img_1,seg_img_0,seg_img_1 = resize_image((img_0,img_1,seg_img_0,seg_img_1),scale)
                landmarks = (
                    landmarks[0]*scale if not landmarks[0] is None else None,
                    landmarks[1]*scale
                )

        return img_0,img_1,seg_img_0,seg_img_1,landmarks

    def __call__(self, index,
                 to_torch = True,
                 modality = None,
                 scale=0,
                 rigidly_reg=False,
                 normalize=False):
        """ Open the brats folder in self.brats_list at the desired index

        :param index: must be int < len(brats_list)
        :param to_torch: (bool) return image as torch.Tensor if True, else a numpy array.
        :return: image at the index of the brats_list
        """
        if index >= len(self.brats_list):
            raise ValueError(f"You asked for a too high value, index is : {index} and len(brat_list) is : {len(self.brats_list)}")
        if not modality is None:
            self.modality = modality
        if self.flag_brats_2021:
            return self._call_brats_2021_(index,to_torch,normalize=normalize)
        if self.flag_bratsReg_2022:
            return self._call_bratsReg_2022(index,to_torch,scale,rigidly_reg=rigidly_reg)
        # source = nib.Nifti1Image(source_img, self.template_affine)
        # return source.get_fdata()
