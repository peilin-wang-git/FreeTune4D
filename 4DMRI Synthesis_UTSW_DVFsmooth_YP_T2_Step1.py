# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:35:20 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:10:03 2022
S217615@Gu6kRYxuF1

@author: user
"""

try:
    import sys # Just in case
    start = sys.version.index('|') # Do we have a modified sys.version?
    end = sys.version.index('|', start + 1)
    version_bak = sys.version # Backup modified sys.version
    sys.version = sys.version.replace(sys.version[start:end+1], '') # Make it legible for platform module
    import platform
    platform.python_implementation() # Ignore result, we just need cache populated
    platform._sys_version_cache[version_bak] = platform._sys_version_cache[sys.version] # Duplicate cache
    sys.version = version_bak # Restore modified version string
except ValueError: # Catch .index() method not finding a pipe
    pass

import SimpleITK as sitk
import tensorflow as tf
import torch
import torch.nn.functional as F
import pydicom
import os
from scipy.ndimage import zoom
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import voxelmorph as vxm
import scipy.interpolate as interpolate
import glob
from skimage.metrics import structural_similarity as SSIM
import peilin
import argparse

start = time.time()

import sys
cwd = r'/mnt/sda/Academics/Code/MyCode/UltraRecon-4D/DDEM.Liver/uq4d_scripts'
sys.path.append(os.path.join(cwd,'voxelmorph-master','pytorch'))
from model import SpatialTransformer
from model_C2F import Unet

bases = (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter)
p = argparse.ArgumentParser(
    formatter_class=type('formatter', bases, {}),
    description=f'FreeTune4D for UTSouthWestern',
)

p.add_argument('--base-path', type=str, default='/mnt/sda/Academics/Code/MyCode/UltraRecon-4D/DDEM.Liver/ReProductionDataset03Case0008/', help="base path of 3D/4D image")
p.add_argument('--MR-number', type=str, default='raw', help="MRN")
p.add_argument('--st_date', type=str, default="StDate", help='StDate')
p.add_argument('--net_path', type=str, default="coarse.h5", help='path to network')
p.add_argument('--name_3d', type=str, default="T2_AX_MVXD", help='name of 3D image')
p.add_argument('--reference_file', type=str, default="IM-301-0001.dcm", help="reference file for header of dicom")
arg = p.parse_args()


net_path = arg.net_path
#HyperMorph
gpu = '0'
device = 'cuda'
vol_size=[128,128,128]
reg_args = dict(
    int_steps=5,
    reg_field = 'warp',
    inshape = vol_size,
    int_resolution=2,
    svf_resolution=2,
    nb_unet_features=([64] * 4, [64] * 6)
)


device_vxm, _ = vxm.tf.utils.setup_device(gpu)
with tf.device(device_vxm):
    model = vxm.networks.HyperVxmDense(**reg_args)
    model = peilin.inspect_weight_loading(model, net_path)
        
frame_limit = 64

amplifier = 1.0

base_path = arg.base_path
MRN=arg.MR_number
StDate=arg.st_date
ThreeDImg=arg.name_3d
ImgName=arg.reference_file

HKU_header_path = os.path.join(base_path, MRN, StDate, ThreeDImg, ImgName)
#HKU_header_path = os.path.join(base_path, MRN, ThreeDImg, ImgName)
HKU_header = pydicom.dcmread(HKU_header_path)
T2_path = os.path.join(base_path, MRN, StDate, ThreeDImg)
mat_path = os.path.join(base_path, MRN, StDate, "phase_T2.mat") 
Output_path = os.path.join(base_path, MRN, StDate, "UQ_4D_T2_Step1") 
patient_name = MRN
patient_id = MRN


def dvf_interp(dvf, vol_size):
    orig_shape = dvf.shape[2:]
    DVF_0 = F.interpolate(torch.unsqueeze(dvf[:,0,:,:,:],1), vol_size, mode = 'trilinear')
    DVF_1 = F.interpolate(torch.unsqueeze(dvf[:,1,:,:,:],1), vol_size, mode = 'trilinear')
    DVF_2 = F.interpolate(torch.unsqueeze(dvf[:,2,:,:,:],1), vol_size, mode = 'trilinear')
    DVF_0 = DVF_0 * (vol_size[0]/orig_shape[0])
    DVF_1 = DVF_1 * (vol_size[1]/orig_shape[1])
    DVF_2 = DVF_2 * (vol_size[2]/orig_shape[2])
    DVF = torch.cat([DVF_0,DVF_1,DVF_2],1)
    return DVF

def img_norm(img):
    max_val = np.max(img)
    min_val = np.min(img)
    img = img * np.array((1/(max_val-min_val)))
    return img

def img_restore(img, orig_size):
    img_restored = np.zeros(list(orig_size))
    pad = np.array(orig_size) - np.array(img.shape)
    if pad[0] % 2 == 0:
        dim0_start = int(pad[0]/2)
        dim0_end = -int(pad[0]/2)
    else:
        dim0_start = int(pad[0]/2)+1
        dim0_end = -int(pad[0]/2)
    if pad[1] % 2 == 0:
        dim1_start = int(pad[1]/2)
        dim1_end = -int(pad[1]/2)
    else:
        dim1_start = int(pad[1]/2)+1
        dim1_end = -int(pad[1]/2)
    if pad[2] % 2 == 0:
        dim2_start = int(pad[2]/2)
        dim2_end = -int(pad[2]/2)
    else:
        dim2_start = int(pad[2]/2)+1
        dim2_end = -int(pad[2]/2)
    img_restored[dim0_start:dim0_end, dim1_start:dim1_end, dim2_start:dim2_end] = img
    return img_restored

def compute_cc(img1, img2):
    mean_1 = np.mean(img1[:])
    mean_2 = np.mean(img2[:])
    std_1 = np.std(img1[:])
    std_2 = np.std(img2[:])
    top = np.mean((img1-mean_1)*(img2-mean_2))
    bot = std_1 * std_2
    cc = top/bot
    return cc

def DVF_smooth(DVF, no_phase):
    # input a concatenation of DVF_x, DVF_y, or DVF_z in n phases, i.e., 128x128
    # x64xn. Return a smoothed DVF, also 128x128x64xn
    # Bspline fitting
    dims,width,length,height = DVF.shape[1:]
    phase_axis = np.arange(no_phase)
    DVF_numpy = DVF.detach().cpu().numpy()
    DVF_smooth = np.zeros_like(DVF_numpy)
    for i in range(width):
        for j in range(length):
            for h in range(height):
                for dim in range(dims):
                    DVF_string = DVF_numpy[:,dim,i,j,h]
                    t, c, k = interpolate.splrep(phase_axis, DVF_string, s=0.1, k=3)
                    spline = interpolate.BSpline(t, c, k, extrapolate=False)
                    DVF_smooth[:,dim,i,j,h] = spline(phase_axis)
                    # N = 100
                    # xmin, xmax = phase_axis.min(), phase_axis.max()
                    # xx = np.linspace(xmin, xmax, N)
                    # yy = spline(xx)
                    # plt.plot(phase_axis, DVF_string, 'bo', label='Original points')
                    # plt.plot(xx, spline(xx), 'r', label='BSpline')
                    # plt.grid()
                    # plt.legend(loc='best')
                    # plt.show()
    return DVF_smooth

def Elastix(moving_image, fixed_image, path, mode="BSpline"):
    import subprocess
    import nibabel as nib

    moving = {"max":np.max(moving_image), "min":np.min(moving_image)}
    fixed = {"max":np.max(fixed_image), "min":np.min(fixed_image)}

    moving_image = (moving_image - moving["min"])/(moving["max"] - moving["min"])
    fixed_image = (fixed_image - fixed["min"])/(fixed["max"] - fixed["min"])

    moving_image_out = sitk.GetImageFromArray(moving_image*255)
    fixed_image_out = sitk.GetImageFromArray(fixed_image*255)
        
    sitk.WriteImage(moving_image_out,'./{}/moving_image.nii.gz'.format(path))
    sitk.WriteImage(fixed_image_out,'./{}/fixed_image.nii.gz'.format(path))

    if mode == "BSpline":
    #    elastix_command = "elastix -f {} -m {} -p ./parameters_BSpline.txt -out {}".format("./{}/fixed_image.nii.gz".format(path), "./{}/moving_image.nii.gz".format(path), path)
       elastix_command = "elastix -f {} -m {} -p ./Par0020bspline2-MI-lesswarp.txt -out {}".format("./{}/fixed_image.nii.gz".format(path), "./{}/moving_image.nii.gz".format(path), path)
        
        # parameter_path = os.path.join(os.getcwd(), 'parameters_BSpline.txt')
    elif mode == "Affine":
       elastix_command = "elastix -f {} -m {} -p ./parameters_Affine.txt -out {}".format("./{}/fixed_image.nii.gz".format(path), "./{}/moving_image.nii.gz".format(path), path)
        # parameter_path = os.path.join(os.getcwd(), 'parameters_Rigid.txt')
    elif mode == "Rigid":
       elastix_command = "elastix -f {} -m {} -p ./parameters_Rigid.txt -out {}".format("./{}/fixed_image.nii.gz".format(path), "./{}/moving_image.nii.gz".format(path), path)
        # parameter_path = os.path.join(os.getcwd(), 'parameters_Rigid.txt')

    try:
       subprocess.run(elastix_command, shell=True, check=True)
       print("Elastix runs successfully!")
    except subprocess.CalledProcessError as e:
       print("Elastix runs with error: ", e)

    warped = torch.tensor(np.squeeze(nib.load("./{}/result.0.nii.gz".format(path)).dataobj).astype(float).transpose((2,1,0)))
    warped[warped<0] = 0
    warped = (warped -  torch.min(warped)) / (torch.max(warped) - torch.min(warped))

    return (warped * (moving["max"] - moving["min"]) + moving["min"]).numpy()
        
FourD = sio.loadmat(mat_path)['FourD_ave_save']
T2 = sio.loadmat(mat_path)['T2_save']

plt.imshow(T2[..., 80], cmap='gray', interpolation=None, aspect=None)
plt.show()
# Load T2 images
T2_dicoms = glob.glob(os.path.join(T2_path, 'IM-*'))
T2_files = []
T2_x = np.array([])
T2_y = np.array([])
T2_z = np.array([])
print('Loading T2w MRI...\n')
count_num = 1
for fname in T2_dicoms:
    print("\rloading: %d/%d" % (count_num, len(T2_dicoms)), end=' ')
    T2_files.append(pydicom.dcmread(os.path.join(T2_path,fname)))
    T2_x = np.append(T2_x, T2_files[count_num-1].ImagePositionPatient[0])
    T2_y = np.append(T2_y, T2_files[count_num-1].ImagePositionPatient[1])
    T2_z = np.append(T2_z, T2_files[count_num-1].ImagePositionPatient[2])    
    count_num = count_num + 1 
print('\n')

origsize_FourD = np.shape(FourD)[:3]
FourD_num_frames = np.shape(FourD)[3]

#Find the closest frame
SSIM_T2 = np.zeros(FourD_num_frames)
for frame_no in range(FourD_num_frames):
    FourD_frame_np = FourD[:,:,:,frame_no]
    SSIM_T2[frame_no] = compute_cc(FourD_frame_np, T2)
    
FourD_T2_frame = FourD[:,:,:,np.argmax(SSIM_T2)]
print(f"Matched Frame is {np.argmax(SSIM_T2)}")

#---------- Registration preparation ----------
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import shutil
import oct2py
# os.environ['OCTAVE_EXECUTABLE'] = shutil.which(r'C:\Users\S217615\AppData\Local\Programs\GNU Octave\Octave-6.4.0\mingw64\bin\octave-cli.exe')
oc = oct2py.Oct2Py()
elastix_path = os.path.join(os.getcwd(), 'elastix-5.0.1-win64')
matlab_elastix_path = os.path.join(os.getcwd(), 'matlab_elastix-master')
octave_tablicious_path = os.path.join(os.getcwd(), 'octave-tablicious-master')
yamlmatlab_path = os.path.join(os.getcwd(), 'yamlmatlab-master')
addMfile_path = os.path.join(os.getcwd(), 'addMfile')
temp_path_1 = os.path.join(os.getcwd(), 'temp_folder_1')
temp_path_2 = os.path.join(os.getcwd(), 'temp_folder_2')
parameter_path_0 = os.path.join(os.getcwd(), 'Par0020rigid.txt')
parameter_path_1 = os.path.join(os.getcwd(), 'Par0020affine.txt')
parameter_path_1_1 = os.path.join(os.getcwd(), 'Par0020affine - 2.txt')
parameter_path_2 = os.path.join(os.getcwd(), 'Par0020bspline2 - MI.txt')
parameter_path_3 = os.path.join(os.getcwd(), 'Par0020bspline2 - MI - 2.txt')
parameter_path_4 = os.path.join(os.getcwd(), 'parameters_BSpline.txt')

oc.addpath(oc.genpath(elastix_path))
oc.addpath(oc.genpath(octave_tablicious_path))
oc.addpath(oc.genpath(matlab_elastix_path))
oc.addpath(oc.genpath(yamlmatlab_path))
oc.addpath(oc.genpath(addMfile_path))

#---------- Register T2 to the selected frame----------
os.makedirs(temp_path_1, exist_ok=True)
os.makedirs(temp_path_2, exist_ok=True)
T2_def_1 = Elastix(T2, FourD_T2_frame, "temp_folder_1", "Rigid")
T2_def = Elastix(T2_def_1, FourD_T2_frame, "temp_folder_2", "BSpline")
# T2_def_1 = oc.elastix(T2, FourD_T2_frame, temp_path_1, [parameter_path_0])
# T2_def = oc.elastix(T2_def_1, FourD_T2_frame, temp_path_2, [parameter_path_4])
peilin.plot_3DLiver(T2, name="T2", path = "./tmp_plot", min=0, max=1)
peilin.plot_3DLiver(FourD_T2_frame, name="4D_selected", path = "./tmp_plot", min=0, max=1)
peilin.plot_3DLiver(T2_def, name="T2_warped", path = "./tmp_plot", min=0, max=1)
## shutil.rmtree(temp_path, ignore_errors=True)


# import sys
# sys.path.append(os.path.join(cwd,'voxelmorph-master','pytorch'))
# from model import SpatialTransformer
# from model_C2F import Unet
# ### Load trained DL model
# vol_size=[128,128,64] 
# nf_enc=[64,64,64,64]
# nf_dec=[64,64,64,64,64,64]
# gpu = '0'
# device = 'cuda'
# torch.backends.cudnn.benchmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu
# model = Unet(nf_enc, nf_dec, 1)
# ST = SpatialTransformer(vol_size)   
# model.to(device)
# ST.to(device)
# model.load_state_dict(torch.load(net_path))

slice_count = 0

## Deform 4D images - T2
T2_crop = img_norm(T2_def)
origsize_T2 = T2_crop.shape
ST_orig_T2 = SpatialTransformer(origsize_T2)   
ST_orig_T2.to(device)
T2_crop = T2_crop[np.newaxis, np.newaxis, ...]
T2_crop = np.array(T2_crop, dtype=np.float32)
T2_crop = torch.from_numpy(T2_crop).to(device).float()
input_T2 = F.interpolate(T2_crop, size = vol_size, mode='trilinear')
FourD_T2_frame = img_norm(FourD_T2_frame)
volmov_T2 = FourD_T2_frame[np.newaxis, np.newaxis, ...]
volmov_T2 = np.array(volmov_T2, dtype=np.float32)
volmov_T2 = torch.from_numpy(volmov_T2).to(device).float()
input_mov_T2 = F.interpolate(volmov_T2, size = vol_size, mode='trilinear')
series_number_base = int(np.random.randint(low = 1500, high = 3000, size = 1))
end_dataloading = time.time()
np.random.seed(int(time.time()))
random_seeds = np.random.choice(10000, size = 10000, replace = False)

rand_study = np.random.randint(10, size = 29)
study_UID_suffix = str()
for i in range(len(rand_study)):
    study_UID_suffix += str(int(rand_study[i]))
study_number = str(int(np.random.randint(low = 1500, high = 3000, size = 1)))

rand_frame = np.random.randint(10, size = 29)
frame_UID_suffix = str()
for i in range(len(rand_frame)):
    frame_UID_suffix += str(int(rand_frame[i]))

DVF_collect = torch.zeros([FourD.shape[3],3,*vol_size]).to(device)
for frame in range(FourD.shape[3]):

    if frame > frame_limit:
        break
    volfix = FourD[:,:,:,frame]
    volfix = img_norm(volfix)
    volfix = volfix[np.newaxis,np.newaxis,...]
    volfix = np.array(volfix, dtype = np.float32)
    volfix = torch.from_numpy(volfix).to(device).float()
    input_fix = F.interpolate(volfix, size = vol_size, mode='trilinear')
    input_diff = input_mov_T2 - input_fix

    peilin.plot_3DLiver(input_mov_T2.squeeze().cpu().numpy(), name="tmp{}".format(frame), path = "./tmp_plot", min=0, max=1)
    with torch.no_grad():
        # flow_up = model(input_mov_T2, input_fix, input_diff) ##DDEM
        flow_up = peilin.inference_HyperMorph(input_mov_T2.cpu(), input_fix.cpu(), model, f"frame_{frame}")
        # flow_up = peilin.inference_SynthMorph(input_mov_T2.cpu(), input_fix.cpu(), model, f"frame_{frame}")
        # flow_up = peilin.inference_SynthMorphDDEM(input_mov_T2.cpu(), input_fix.cpu(), model, f"frame_{frame}")

        peilin.plot_3DLiver(flow_up.squeeze()[0].detach().cpu().numpy(), name="tmp_dvf_1dim_{}".format(frame), path = "./tmp_plot", min=0, max=1)
        peilin.plot_3DLiver(flow_up.squeeze()[1].detach().cpu().numpy(), name="tmp_dvf_2dim_{}".format(frame), path = "./tmp_plot", min=0, max=1)
        peilin.plot_3DLiver(flow_up.squeeze()[2].detach().cpu().numpy(), name="tmp_dvf_3dim_{}".format(frame), path = "./tmp_plot", min=0, max=1)
        flow_orig = dvf_interp(flow_up, origsize_T2)
    DVF_collect[frame, :, :, :, :] = flow_up * amplifier

DVF_smoothed = DVF_smooth(DVF_collect, FourD.shape[3])

for frame in range(FourD.shape[3]):
    flow_smooth = torch.unsqueeze(torch.from_numpy(DVF_smoothed[frame,:,:,:,:]),0).to(device) 
    flow_orig = dvf_interp(flow_smooth, origsize_T2)
    T2_interp_dvf = ST_orig_T2(T2_crop, flow_orig)
    volfix = FourD[:,:,:,frame]
    volfix = img_norm(volfix)
    volfix = volfix[np.newaxis,np.newaxis,...]
    volfix = np.array(volfix, dtype = np.float32)
    volfix = torch.from_numpy(volfix).to(device).float()
    FourD_np = F.interpolate(volfix, origsize_T2).cpu().numpy()[0,0,:]
    T2_FourD_np = T2_interp_dvf[0,0,:].cpu().numpy()
    n1=HKU_header.PixelSpacing[0]
    n2=HKU_header.PixelSpacing[1]
    n3=HKU_header.SliceThickness
    T2_FourD_np_restored = F.interpolate(T2_interp_dvf, list([int(tmp) for tmp in np.int16(np.array(T2_def.shape)/[n1,n2,n3])]))[0,0,:].cpu().numpy()
    mat_name = Output_path + '/UQ_T2_' + str(frame) + '.mat'
    mat_var_name_T2 = 'UQ_T2_' + str(frame)
    mat_var_name_4D = 'FourD_' + str(frame)
    sio.savemat(mat_name,{mat_var_name_T2:T2_FourD_np, mat_var_name_4D:FourD_np})

    peilin.plot_3DLiver(T2_def, name="T2_def_{}".format(frame), path = "./tmp_plot", min=0, max=FourD_np.max(), titles = ["Coronal Plane", "Sagittal Plane", "Axial Plance"])
    peilin.plot_3DLiver(T2_crop.detach().to("cpu").numpy()[0,0], name="T2_crop_{}".format(frame), path = "./tmp_plot", min=0, max=FourD_np.max(), titles = ["Coronal Plane", "Sagittal Plane", "Axial Plance"])
    peilin.plot_3DLiver(T2_FourD_np, name="UQ4D_{}".format(frame), path = "./tmp_plot", min=0, max=T2_FourD_np.max(), titles = ["Coronal Plane", "Sagittal Plane", "Axial Plance"])
    peilin.plot_3DLiver(FourD_np, name="LQ4D_{}".format(frame), path = "./tmp_plot", min=0, max=FourD_np.max(), titles = ["Coronal Plane", "Sagittal Plane", "Axial Plance"])
#---------- Write to Dicom ----------
### Use T2 coordinate and information
    np.random.seed(int(start)+slice_count) # Problem. It will make all the image with the same frame number get the same UID.
    time.sleep(1)
    rand_series = np.random.randint(10, size = 29)
    series_UID_suffix = str()
    for i in range(len(rand_series)):
        series_UID_suffix += str(int(rand_series[i]))
    series_number = pydicom.valuerep.IS(series_number_base + frame)
    for slice_index in range(T2_FourD_np_restored.shape[2]):
    # for slice_index in range(len(T2_files)):
        # if slice_index <= pad-1:
        #     continue
        # slice_index = slice_index - pad
        dicom_header = T2_files[-slice_index]
        SeriesInstanceUID = pydicom.uid.UID(dicom_header.SeriesInstanceUID[:27] + series_UID_suffix)
        np.random.seed(int(time.time() + random_seeds[slice_count]))
        rand_instance = np.random.randint(10, size = 29)
        instance_UID_suffix = str()
        for i in range(len(rand_instance)):
            instance_UID_suffix += str(int(rand_instance[i]))
        SOPInstanceUID = pydicom.uid.UID(dicom_header.SOPInstanceUID[:27] + str(frame) + str(slice_index) + instance_UID_suffix[:-len(str(frame) + str(slice_index))])
        StudyInstanceUID = pydicom.uid.UID(dicom_header.StudyInstanceUID[:27] + study_UID_suffix)
        FrameOfReferenceUID = pydicom.uid.UID(dicom_header.FrameOfReferenceUID[:27] + frame_UID_suffix)
### Change dicom file
        HKU_header.PatientName = patient_name
        HKU_header.PatientID = patient_id
        T2_FourD_np_restored[T2_FourD_np_restored<0] = 0
        HKU_header.PixelData = np.uint16(T2_FourD_np_restored[:,:,slice_index]*500).tobytes() # float? int?
        HKU_header.SOPInstanceUID = SOPInstanceUID
        HKU_header.FrameOfReferenceUID = FrameOfReferenceUID
        HKU_header.SeriesInstanceUID = SeriesInstanceUID
        HKU_header.SeriesNumber = series_number
        HKU_header.StudyID = study_number
        HKU_header.Rows = T2_FourD_np_restored.shape[0]
        HKU_header.Columns = T2_FourD_np_restored.shape[1]
        HKU_header.SliceThickness = dicom_header.SliceThickness
        HKU_header.SpacingBetweenSlices = dicom_header.SliceThickness #5.02 dicom_header.SpacingBetweenSlices
        HKU_header.ImagePositionPatient = [np.min(T2_x), np.min(T2_y), np.max(T2_z)-slice_index*dicom_header.SliceThickness]
        HKU_header.ImageOrientationPatient = dicom_header.ImageOrientationPatient
        HKU_header.PixelSpacing = dicom_header.PixelSpacing
        HKU_header.InstanceNumber = dicom_header.InstanceNumber
        HKU_header.SliceLocation = np.max(T2_z)-slice_index*dicom_header.SliceThickness
        HKU_header.SeriesDescription = 'UQ-T2w 4D-MRI frame ' + str(frame) 
        # dicom_header.FrameOfReferenceUID = FrameOfReferenceUID
        # dicom_header.FrameOfReferenceUID = FrameOfReferenceUID
        # dicom_header.SeriesInstanceUID = SeriesInstanceUID
        # dicom_header.SeriesNumber = series_number
        # dicom_header.StudyInstanceUID = StudyInstanceUID
        # dicom_header.StudyID = study_number
        # dicom_header.PixelData = np.uint16(T1_FourD_np_restored[:,:,slice_index]*1000).tobytes() # float? int?
        file_name = Output_path + '/T2w_frame' + str(frame) + '_' + str(slice_index) + '.dcm'
        HKU_header.save_as(file_name)
        slice_count = slice_count + 1
    print('UQ-T2w 4D-MRI Frame %d generated. \n' %(frame))
    time.sleep(1)

for frame in range(FourD.shape[3]):
    volfix = FourD[:,:,:,frame]
    volfix = img_norm(volfix)
    volfix = volfix[np.newaxis,np.newaxis,...]
    volfix = np.array(volfix, dtype = np.float32)
    volfix = torch.from_numpy(volfix).to(device).float().cpu()
    # FourD_np = F.interpolate(volfix, origsize_T2).cpu().numpy()[0,0,:]
    n1=HKU_header.PixelSpacing[0]
    n2=HKU_header.PixelSpacing[1]
    n3=HKU_header.SliceThickness
    FourD_np_restored = F.interpolate(volfix, list([int(tmp) for tmp in np.int16(np.array(T2_def.shape)/[n1,n2,n3])]))[0,0,:].cpu().numpy()
    # mat_name = Output_path + '/LQ_T2_' + str(frame) + '.mat'
    # mat_var_name_T2 = 'LQ_T2_' + str(frame)
    # mat_var_name_4D = 'FourD_' + str(frame)
    # sio.savemat(mat_name,{mat_var_name_T2:T2_FourD_np, mat_var_name_4D:FourD_np})

    # peilin.plot_3DLiver(T2_FourD_np, name="LQ4D_{}".format(frame), path = "./tmp_plot", min=0, max=T2_FourD_np.max(), titles = ["Coronal Plane", "Sagittal Plane", "Axial Plance"])
    # peilin.plot_3DLiver(FourD_np, name="LQ4D_{}".format(frame), path = "./tmp_plot", min=0, max=FourD_np.max(), titles = ["Coronal Plane", "Sagittal Plane", "Axial Plance"])
#---------- Write to Dicom ----------
### Use T2 coordinate and information
    np.random.seed(int(start)+slice_count) # Problem. It will make all the image with the same frame number get the same UID.
    time.sleep(1)
    rand_series = np.random.randint(10, size = 29)
    series_UID_suffix = str()
    for i in range(len(rand_series)):
        series_UID_suffix += str(int(rand_series[i]))
    series_number = pydicom.valuerep.IS(series_number_base + frame)
    for slice_index in range(FourD_np_restored.shape[2]):
    # for slice_index in range(len(T2_files)):
        # if slice_index <= pad-1:
        #     continue
        # slice_index = slice_index - pad
        dicom_header = T2_files[-slice_index]
        SeriesInstanceUID = pydicom.uid.UID(dicom_header.SeriesInstanceUID[:27] + series_UID_suffix)
        np.random.seed(int(time.time() + random_seeds[slice_count]))
        rand_instance = np.random.randint(10, size = 29)
        instance_UID_suffix = str()
        for i in range(len(rand_instance)):
            instance_UID_suffix += str(int(rand_instance[i]))
        SOPInstanceUID = pydicom.uid.UID(dicom_header.SOPInstanceUID[:27] + str(frame) + str(slice_index) + instance_UID_suffix[:-len(str(frame) + str(slice_index))])
        StudyInstanceUID = pydicom.uid.UID(dicom_header.StudyInstanceUID[:27] + study_UID_suffix)
        FrameOfReferenceUID = pydicom.uid.UID(dicom_header.FrameOfReferenceUID[:27] + frame_UID_suffix)
### Change dicom file
        HKU_header.PatientName = patient_name
        HKU_header.PatientID = patient_id
        # T2_FourD_np_restored[T2_FourD_np_restored<0] = 0
        HKU_header.PixelData = np.uint16(FourD_np_restored[:,:,slice_index]*500).tobytes() # float? int?
        HKU_header.SOPInstanceUID = SOPInstanceUID
        HKU_header.FrameOfReferenceUID = FrameOfReferenceUID
        HKU_header.SeriesInstanceUID = SeriesInstanceUID
        HKU_header.SeriesNumber = series_number
        HKU_header.StudyID = study_number
        HKU_header.Rows = FourD_np_restored.shape[0]
        HKU_header.Columns = FourD_np_restored.shape[1]
        HKU_header.SliceThickness = dicom_header.SliceThickness
        HKU_header.SpacingBetweenSlices = dicom_header.SliceThickness #5.02 dicom_header.SpacingBetweenSlices
        HKU_header.ImagePositionPatient = [np.min(T2_x), np.min(T2_y), np.max(T2_z)-slice_index*dicom_header.SliceThickness]
        HKU_header.ImageOrientationPatient = dicom_header.ImageOrientationPatient
        HKU_header.PixelSpacing = dicom_header.PixelSpacing
        HKU_header.InstanceNumber = dicom_header.InstanceNumber
        HKU_header.SliceLocation = np.max(T2_z)-slice_index*dicom_header.SliceThickness
        HKU_header.SeriesDescription = 'LQ-T2w 4D-MRI frame ' + str(frame) 
        # dicom_header.FrameOfReferenceUID = FrameOfReferenceUID
        # dicom_header.FrameOfReferenceUID = FrameOfReferenceUID
        # dicom_header.SeriesInstanceUID = SeriesInstanceUID
        # dicom_header.SeriesNumber = series_number
        # dicom_header.StudyInstanceUID = StudyInstanceUID
        # dicom_header.StudyID = study_number
        # dicom_header.PixelData = np.uint16(T1_FourD_np_restored[:,:,slice_index]*1000).tobytes() # float? int?
        file_name = Output_path + '/LQ_T2w_frame' + str(frame) + '_' + str(slice_index) + '.dcm'
        HKU_header.save_as(file_name)
        slice_count = slice_count + 1
    print('LQ-T2w 4D-MRI Frame %d generated. \n' %(frame))
    time.sleep(1)

# # Original 4D
# gif_orig = np.flip(
#     np.rot90(FourD[240, :, :, :], k=1, axes=(0,1)),
#     axis=0
# )
# gif_orig_resized = zoom(gif_orig, (FourD_vs[2], FourD_vs[1], 1), order=1)
# GIFplot2(gif_orig_resized,
#          os.path.join(patient_path, '4D_AX_T2.gif'),
#          duration=0.3,
#          intensity_range=(0, 0.8*gif_orig_resized.max()))
    