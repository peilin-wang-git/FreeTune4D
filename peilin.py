import numpy as np
import h5py
import torch
import pandas as pd
import math
import peilin_loss as loss
import matplotlib.pyplot as plt
import os
import pydicom

def plot_3DLiver(img, name="Test", titles = ["Axial Plance", "Coronal Plane", "Sagittal Plane"], path = "TestResult", max=1, min=0, dpi=300):
    half_size = np.array(img.shape)
    half_size = half_size/2
    half_size = half_size.astype(int)

    # plt.subplots_adjust(hspace=0.3, wspace=0.3)
    img1 = img[half_size[0], :, :]
    img2 = img[:, half_size[1], :]#####改了
    img3 = img[:, :, int(half_size[2]/2)]#####改了

    img1[0,0] = max
    img2[0,0] = max
    img3[0,0] = max

    img1[0,1] = min
    img2[0,1] = min
    img3[0,1] = min

    plt.figure(figsize=(12, 4), dpi=dpi)  # 可以调整尺寸以适应子图
    plt.subplot(131)
    plt.imshow(img1, cmap='gray')
    plt.title(titles[0])
    cb = plt.colorbar(fraction=0.045)
    cb.ax.tick_params(labelsize=7)
    # cb.ax.tick_params()
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(img2, cmap='gray')
    plt.title(titles[1])
    cb = plt.colorbar(fraction=0.045)
    cb.ax.tick_params(labelsize=7)
    # cb.ax.tick_params()
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(img3, cmap='gray')
    plt.title(titles[2])
    cb = plt.colorbar(fraction=0.045)
    cb.ax.tick_params(labelsize=7)
    # cb.ax.tick_params()
    plt.axis('off')

    try:
        os.makedirs(path)
    except:
        print("Path {} has already exists.".format(path))
    
    plt.savefig('{}/{}.jpg'.format(path, name), dpi=dpi)
    plt.close()  # 关闭图形以释放资源

def inspect_weight_loading(model, weights_path):

    # 1. 获取模型中所有可赋值权重的层名
    model_layers = []
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            model_layers.append(layer.name)
    
    # 2. 读取 .h5 权重文件中的层名
    weight_layers = []
    with h5py.File(weights_path, 'r') as f:
        # 获取所有 group（通常是层名）
        weights = f["model_weights"]
        
        for key in weights.keys():
            if isinstance(weights[key], h5py.Group):
                # 检查 group 下是否有权重数据（dataset）
                for subkey in weights[key].keys():
                    if isinstance(weights[key][subkey], h5py.Group):
                        if any(isinstance(weights[key][subkey][subsubkey], h5py.Dataset) for subsubkey in weights[key][subkey].keys()):
                            weight_layers.append(subkey)

    # print("权重文件中的层名（group 名）:", weight_layers)
    
    # 3. 对比if isinstance(f[key][subkey], h5py.Dataset)
    matched = set(model_layers) & set(weight_layers)
    only_in_model = set(model_layers) - set(weight_layers)
    only_in_weights = set(weight_layers) - set(model_layers)

    print("✅ 匹配的层（将被加载）:", matched)
    print("❌ 模型中有但权重中没有:", only_in_model)
    print("ℹ️  权重中有多余的层（将被忽略）:", only_in_weights)

    # 4. 加载权重
    model.load_weights(weights_path, by_name=True)
    print("权重加载完成。")
    return model

def ObtainSubMatrixGlobal(warped, fixed, TempColumns):
    matrix = {}
    for resultcolumn in TempColumns:
        if "LCC" == resultcolumn:
            # LCC = loss.NCC().loss(warped, fixed))
            LCC = loss.NCC().loss(warped, fixed)
            matrix[resultcolumn] = LCC
        if "SSIM" == resultcolumn:
            SSIM = loss.SSIM().loss(warped, fixed)
            matrix[resultcolumn] = SSIM
            # matrix.append(SSIM)
        if "PSNR" == resultcolumn:
            PSNR = loss.PSNR().loss(warped, fixed)
            matrix[resultcolumn] = PSNR
            # matrix.append(PSNR)
        if "NMI" == resultcolumn:
            NMI = loss.NMI().loss(warped, fixed)
            matrix[resultcolumn] = NMI
            # matrix.append(NMI)
        if "MSE" == resultcolumn:
            MSE = loss.MSE().loss(warped, fixed)
            matrix[resultcolumn] = MSE
            # matrix.append(MSE)
        if "ROIErr" == resultcolumn:
            pass
            # ROIErr = loss.ROIErr().loss(warped, fixed))#####定六个点吧
            # matrix.append(ROIErr)
        if "CNR" == resultcolumn:
            pass
            # CNR = loss.CNR().loss(warped, fixed))
            # matrix.append(CNR)
        if "PBM" == resultcolumn:
            PBM = loss.PBM().loss(warped, fixed)
            matrix[resultcolumn] = PBM
            # matrix.append(PBM)
        if "FWHM" == resultcolumn:#edge full-width-at-half maximum
            pass

    return matrix

def decide_hyp(img_list, model, save_path, name = "ResultHyp"):
    ResultColumnsHyp = ["HyperParameter","4DTo4D_LCC"]
    [img4D_slice, img4D_moving] = img_list

    ResultHyp = pd.DataFrame(columns=ResultColumnsHyp)
    total_num = 11
    hyp_list = np.linspace(0, 1, total_num)

    batch_size = 20
    warped_4D_np = np.zeros([total_num]+list(img4D_slice.shape))
    for i in range(math.ceil(total_num / batch_size)):
        if i < math.ceil(total_num / batch_size)-1:
            hyp = np.array(hyp_list[i*batch_size:(i+1)*batch_size])[..., None]
            warped_4D_tmp, warp_4DT4D = model.predict([np.repeat(img4D_moving[None,...,None],batch_size,0), 
                                                       np.repeat(img4D_slice[None,...,None],batch_size,0), hyp])
            
            warped_4D_np[i*batch_size:(i+1)*batch_size, ...] = warped_4D_tmp[..., 0]
        else:
            hyp = np.array(hyp_list[i*batch_size:])[..., None]
            warped_4D_tmp, warp_4DT4D = model.predict([np.repeat(img4D_moving[None,...,None],total_num % batch_size,0), 
                                                       np.repeat(img4D_slice[None,...,None],total_num % batch_size,0), hyp])
            
            warped_4D_np[i*batch_size:, ...] = warped_4D_tmp[..., 0]
        print("hyp: ", hyp)
    
    matrix = ObtainSubMatrixGlobal(torch.Tensor(warped_4D_np[:, None, ...]), 
                                   torch.Tensor(np.repeat(img4D_slice[None, None, ...], total_num, 0)), ["LCC"])

    return hyp_list[np.argmin(matrix['LCC'].cpu().numpy())]

def inference_HyperMorph(img4D_moving_resized, img4D_fixed_resized, reg_model, name):
    plot_3DLiver(img4D_moving_resized.squeeze(), name="tmp_raw_{}".format(name), path = "./tmp_plot", min=0, max=1)
    img4D_fixed_resized = img4D_fixed_resized.squeeze().permute([2,0,1]).numpy()
    img4D_moving_resized = img4D_moving_resized.squeeze().permute([2,0,1]).numpy()

    # warped_imgs_pre, DVF_3DTo4D_resized = reg_model.predict([img4D_moving_resized[None, ...,None], img4D_fixed_resized[None,...,None]])
            
    hyp = decide_hyp([img4D_fixed_resized, img4D_moving_resized], reg_model, name)
    hyp_4DTo4D = np.array([hyp])[None, ...]

    plot_3DLiver(img4D_moving_resized, name="tmp_moving_{}".format(name), path = "./tmp_plot", min=0, max=1)
    plot_3DLiver(img4D_fixed_resized, name="tmp_fixed_{}".format(name), path = "./tmp_plot", min=0, max=1)
    _, DVF_3DTo4D_resized = reg_model.predict([img4D_moving_resized[None,...,None], img4D_fixed_resized[None,...,None], hyp_4DTo4D])
    plot_3DLiver(_.squeeze(), name="tmp_warped_{}".format(name), path = "./tmp_plot", min=0, max=1)

    DVF_3DTo4D_resized = torch.Tensor(DVF_3DTo4D_resized).permute([0,4,2,3,1])
    DVF_3DTo4D_resized = DVF_3DTo4D_resized[:, [1,2,0], ...]

    return DVF_3DTo4D_resized
            
def inference_SynthMorphDDEM(img4D_moving_resized, img4D_fixed_resized, reg_model, name):
    plot_3DLiver(img4D_moving_resized.squeeze(), name="tmp_raw_{}".format(name), path = "./tmp_plot", min=0, max=1)
    img4D_fixed_resized = img4D_fixed_resized.squeeze().numpy()
    img4D_moving_resized = img4D_moving_resized.squeeze().numpy()

    # warped_imgs_pre, DVF_3DTo4D_resized = reg_model.predict([img4D_moving_resized[None, ...,None], img4D_fixed_resized[None,...,None]])
            
    # hyp = decide_hyp([img4D_fixed_resized, img4D_moving_resized], reg_model, name)
    # hyp_4DTo4D = np.array([hyp])[None, ...]

    plot_3DLiver(img4D_moving_resized, name="tmp_moving_{}".format(name), path = "./tmp_plot", min=0, max=1)
    plot_3DLiver(img4D_fixed_resized, name="tmp_fixed_{}".format(name), path = "./tmp_plot", min=0, max=1)
    _, DVF_3DTo4D_resized = reg_model.predict([img4D_moving_resized[None,...,None], img4D_fixed_resized[None,...,None]])
    plot_3DLiver(_.squeeze(), name="tmp_warped_{}".format(name), path = "./tmp_plot", min=0, max=1)

    DVF_3DTo4D_resized = torch.Tensor(DVF_3DTo4D_resized).permute([0,4,1,2,3])

    return DVF_3DTo4D_resized
            
def inference_SynthMorph(img4D_moving_resized, img4D_fixed_resized, reg_model, name):
    plot_3DLiver(img4D_moving_resized.squeeze(), name="tmp_raw_{}".format(name), path = "./tmp_plot", min=0, max=1)
    img4D_fixed_resized = img4D_fixed_resized.squeeze().permute([2,0,1]).numpy()
    img4D_moving_resized = img4D_moving_resized.squeeze().permute([2,0,1]).numpy()

    # warped_imgs_pre, DVF_3DTo4D_resized = reg_model.predict([img4D_moving_resized[None, ...,None], img4D_fixed_resized[None,...,None]])
            
    # hyp = decide_hyp([img4D_fixed_resized, img4D_moving_resized], reg_model, name)
    # hyp_4DTo4D = np.array([hyp])[None, ...]

    plot_3DLiver(img4D_moving_resized, name="tmp_moving_{}".format(name), path = "./tmp_plot", min=0, max=1)
    plot_3DLiver(img4D_fixed_resized, name="tmp_fixed_{}".format(name), path = "./tmp_plot", min=0, max=1)
    _, DVF_3DTo4D_resized = reg_model.predict([img4D_moving_resized[None,...,None], img4D_fixed_resized[None,...,None]])
    plot_3DLiver(_.squeeze(), name="tmp_warped_{}".format(name), path = "./tmp_plot", min=0, max=1)

    DVF_3DTo4D_resized = torch.Tensor(DVF_3DTo4D_resized).permute([0,4,2,3,1])
    DVF_3DTo4D_resized = DVF_3DTo4D_resized[:, [1,2,0], ...]

    return DVF_3DTo4D_resized
            
# def inference_TransMorph(img4D_fixed, img4D_moving, seg4D_fixed, vert4D_intp, img3D_moving_resized, img3D_moving, seg3D_moving, reg_model, ImageShape_step3_tmp, ImageShape_step2_tmp, sizerate_tmp):

#     [img4D_fixed_resized, img4D_moving_resized], _, step2_4Dsize = img_process_4D_step2([img4D_fixed, img4D_moving], [seg4D_fixed, vert4D_intp], ImageShape_step2_tmp)
                                                      
#     outputs = reg_model(torch.cat((img4D_moving_resized[None,None,...].to(gl_device).),
#                                        img4D_fixed_resized[None,None,...].to(gl_device).)), dim=1))
#     DVF_3DTo4D_resized =  outputs[1]
#     enc_feature = outputs[2]
#     dec_feature = outputs[3]

#     enc_feature_ = [feat.cpu().detach().numpy() for feat in enc_feature]
#     dec_feature_ = [feat.cpu().detach().numpy() for feat in dec_feature]
#     enc_feature = [feat.cpu().detach() for feat in enc_feature]
#     dec_feature = [feat.cpu().detach() for feat in dec_feature]

#     np.savez("./result/Num{}/cka_matrix_frame{}.npz".format(i, j), enc0 = enc_feature_[0], enc1 = enc_feature_[1],
#          enc2 = enc_feature_[2], enc3 = enc_feature_[3], enc4 = enc_feature_[4],
#          dec0 = dec_feature_[0], dec1 = dec_feature_[1], dec2 = dec_feature_[2],
#          dec3 = dec_feature_[3], dec4 = dec_feature_[4])
            
#     if j == 0:
#         features_frame = enc_feature + dec_feature
#     else:
#         features_frame = [torch.concat([features_frame[feat_num], (enc_feature + dec_feature)[feat_num]], dim=0) for feat_num in range(len(features_frame))]

#     DVF_3DTo4D = warp_resized([DVF_3DTo4D_resized.permute([0,2,3,4,1]).detach().to("cpu").numpy()], list([i) for i in sizerate_tmp]))
#     DVF_3DTo4D = torch.Tensor(DVF_3DTo4D[0].numpy()).permute([0,4,1,2,3]).to(gl_device)
#     [img4D_fixed], [seg4D_fixed, vert4D_intp] = img_process_4D_step3([img4D_fixed], [seg4D_fixed, vert4D_intp], ImageShape_step3_tmp)
            
#     transform_model = MyCode.SpatialTransformer(ImageShape_step3_tmp).to(gl_device)
#     transform_model_nearest = MyCode.SpatialTransformer(ImageShape_step3_tmp, mode="nearest").to(gl_device)    
#     img3D_warped = transform_model(img3D_moving.to(gl_device).), DVF_3DTo4D.)).squeeze().to("cpu")
#     seg3D_warped = transform_model_nearest(seg3D_moving.to(gl_device).), DVF_3DTo4D.)).squeeze().to("cpu")
            
#     return img3D_warped, seg3D_warped
import glob
def load_dicom_4d(input_dicom_path):
    """
    读取指定路径下所有符合命名规则的T2w DICOM文件，并将它们整理成一个四维数组 [frame, slice, H, W]。
    
    参数:
    input_dicom_path (str): DICOM文件夹路径，包含命名规则为 T2w_frame{frame}_{slice}.dcm 的文件。
    
    返回:
    np.ndarray: 四维数组，形状为 [frame, slice, H, W]。
    """
    # 获取所有符合命名规则的DICOM文件
    dicom_files = sorted(glob.glob(os.path.join(input_dicom_path, 'T2w_frame*.dcm')))

    # 用于存储每个frame的每个slice的图像数据
    dicom_data_dict = {}
    headers = []

    # 解析文件名中的frame和slice，并将图像数据整理到字典中
    for dicom_file in dicom_files:
        # 提取文件名并解析frame和slice
        filename = os.path.basename(dicom_file)
        parts = filename.split('_')
        frame = int(parts[1].replace('frame', ''))  # 提取frame编号
        slice_index = int(parts[2].replace('.dcm', ''))  # 提取slice编号

        # 读取DICOM图像
        dicom_header = pydicom.dcmread(dicom_file)
        img_data = dicom_header.pixel_array
        headers.append(dicom_header)

        # 按照frame和slice编号组织图像数据
        if frame not in dicom_data_dict:
            dicom_data_dict[frame] = {}
        
        dicom_data_dict[frame][slice_index] = img_data

    # 获取frame和slice的最大值，以确定四维数组的大小
    num_frames = len(dicom_data_dict)
    num_slices = max([len(dicom_data_dict[frame]) for frame in dicom_data_dict])
    img_height, img_width = dicom_data_dict[0][0].shape  # 假设所有图像的尺寸相同

    # 初始化一个四维数组，用于存储所有图像数据
    fourD_array = np.zeros((num_frames, num_slices, img_height, img_width), dtype=np.uint16)

    # 填充四维数组
    for frame in dicom_data_dict:
        for slice_index in dicom_data_dict[frame]:
            fourD_array[frame, slice_index, :, :] = dicom_data_dict[frame][slice_index]

    # 返回四维数组
    return fourD_array, headers
