import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings
import pystrum.pynd.ndutils as nd

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, device='cpu', win=None):
        self.win = win
        self.device = device

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc, dim=[i for i in range(1, len(cc.shape))])

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        # print("MSE Loss: ", torch.max(y_true), torch.min(y_true), torch.max(y_pred), torch.min(y_pred))
        return torch.mean((y_true - y_pred) ** 2)

class Dice:
    def loss(self, y_true, y_pred, labels=None, include_zero=False):

        y_true = np.round(y_true, decimals=0)
        y_pred = np.round(y_pred, decimals=0)
        if labels is None:
            labels = np.concatenate([np.unique(a) for a in [y_true, y_pred]])
            labels = np.sort(np.unique(labels))
        if not include_zero:
            labels = np.delete(labels, np.argwhere(labels == 0)) 
    
        dicem = np.zeros(len(labels))
        for idx, label in enumerate(labels):
            top = 2 * np.sum(np.logical_and(y_true == label, y_pred == label))
            bottom = np.sum(y_true == label) + np.sum(y_pred == label)
            bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
            dicem[idx] = top / bottom
        return np.mean(dicem)

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()

class NMI:
    def __init__(self):
        pass
    
    def loss(self,  img1, img2, bins=32):
        shape = img1.shape[2:]
        img1 = (img1-torch.min(img1))/(torch.max(img1)-torch.min(img1))
        img2 = (img2-torch.min(img2))/(torch.max(img2)-torch.min(img2))

        # 将图像转换为numpy数组
        img1_np = img1.numpy()
        img2_np = img2.numpy()

        # 计算直方图
        hist_2d, x_edges, y_edges = np.histogram2d(img1_np.ravel(), img2_np.ravel(), bins=bins)

        # 计算概率密度
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1) + 1/np.prod(np.array(shape))/10
        py = np.sum(pxy, axis=0) + 1/np.prod(np.array(shape))/10

        # 计算互信息
        px_py = px[:, None] * py[None, :]
        non_zero_indices = pxy > 0
        mi = np.sum(pxy[non_zero_indices] * np.log(pxy[non_zero_indices] / px_py[non_zero_indices]))

        # 计算归一化互信息
        entropy_x = -np.sum(px * np.log(px))
        entropy_y = -np.sum(py * np.log(py))
        nmi = mi / np.sqrt(entropy_x * entropy_y)

        return nmi

class PSNR:
    def __init__(self):
        pass
    
    def loss(self,  img1, img2):
        img1 = (img1-torch.min(img1))/(torch.max(img1)-torch.min(img1))
        img2 = (img2-torch.min(img2))/(torch.max(img2)-torch.min(img2))

        img1 = img1.cpu().detach().numpy()
        img2 = img2.cpu().detach().numpy()

        # 计算均方误差
        mse = max(np.mean((img1 - img2) ** 2), 1e-7)

        # 如果图像的像素范围是 [0, 1]，则 PSNR 计算公式为 20 * log10(MAX) - 10 * log10(MSE)
        # 其中 MAX 是像素的最大值（例如对于数据范围为 [0, 1] 的图像，MAX = 1）
        # 注意：如果像素范围是 [0, 255]，则 MAX = 255
        max_pixel = 1.0  # 假设图像像素范围是 [0, 1]
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
        if mse == 1e-7:
            warnings.warn("MSE less than 1e-7.")

        return psnr

class SSIM:
    def __init__(self):
        pass
    
    def loss_old(self,  y_true, y_pred):
        img_shape = y_true.shape
        if len(img_shape)==4:
            pixel_num = img_shape[2]*img_shape[3]
        elif len(img_shape)==5:
            pixel_num = img_shape[2]*img_shape[3]*img_shape[4]

        y_true = (y_true-torch.min(y_true))/(torch.max(y_true)-torch.min(y_true))
        y_pred = (y_pred-torch.min(y_pred))/(torch.max(y_pred)-torch.min(y_pred))

        mean_true = torch.mean(y_true)
        mean_pred = torch.mean(y_pred)

        mu_true = (torch.sum((y_true-mean_true)**2)/(pixel_num-1))**0.5
        mu_pred = (torch.sum((y_pred-mean_pred)**2)/(pixel_num-1))**0.5
        mu_xy = torch.sum((y_true-mean_true)(y_pred-mean_pred))/(pixel_num-1)

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        index = ((2*mean_true*mean_pred+c1)*(2*mu_xy+c2))/(((mean_true**2)+(mean_pred**2)+c1)*((mu_true**2)+(mu_pred**2)+c2))
        return index
    
    def loss(self, img1, img2):
        if len(img1.shape)==4:
            return self.loss2D(img1, img2)
        elif len(img1.shape)==5:
            return self.loss3D(img1, img2)
    
    def loss2D(self, img1, img2, window_size=11, sigma=1.5):
        # 数据归一化处理
        img1 = (img1-torch.min(img1))/(torch.max(img1)-torch.min(img1))
        img2 = (img2-torch.min(img2))/(torch.max(img2)-torch.min(img2))
    
        # 创建高斯权重
        channel = img1.size(1)
        window = torch.FloatTensor(window_size, window_size).fill_(1)
        gaussian = window.unsqueeze(0).unsqueeze(0)
        gaussian = gaussian.expand(channel, 1, window_size, window_size).contiguous()
        gaussian = gaussian.to(img1.device, img1.dtype)

        # 计算均值
        mu1 = F.conv2d(img1, gaussian, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, gaussian, padding=window_size // 2, groups=channel)
    
        # 计算方差
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, gaussian, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, gaussian, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, gaussian, padding=window_size // 2, groups=channel) - mu1_mu2

        # SSIM公式
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        # 结构相似性指数
        return float(ssim_map.mean())
    
    def loss3D(self, img1, img2, window_size=11, sigma=1.5):
        # 数据归一化处理
        img1 = (img1-torch.min(img1))/(torch.max(img1)-torch.min(img1))
        img2 = (img2-torch.min(img2))/(torch.max(img2)-torch.min(img2))
    
        # 创建高斯权重
        channel = img1.size(1)
        window = torch.FloatTensor(window_size, window_size, window_size).fill_(1)
        gaussian = window.unsqueeze(0).unsqueeze(0)
        gaussian = gaussian.expand(channel, 1, window_size, window_size, window_size).contiguous()
        gaussian = gaussian.to(img1.device, img1.dtype)

        pixel = 1
        for i in gaussian.shape:
            pixel = pixel*i
        gaussian = gaussian/pixel
        
        # 计算均值
        mu1 = F.conv3d(img1, gaussian, padding=window_size // 2, groups=channel)
        mu2 = F.conv3d(img2, gaussian, padding=window_size // 2, groups=channel)
    
        # 计算方差
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # sigma1_sq = F.conv3d((img1 - mu1)**2, gaussian/pixel, padding=window_size // 2, groups=channel)**0.5
        # sigma2_sq = F.conv3d((img2 - mu2)**2, gaussian/pixel, padding=window_size // 2, groups=channel)**0.5
        # sigma12 = F.conv3d((img1 - mu1) * (img2 - mu2), gaussian/pixel, padding=window_size // 2, groups=channel)**0.5

        sigma1_sq = F.conv3d(img1 * img1, gaussian, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, gaussian, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, gaussian, padding=window_size // 2, groups=channel) - mu1_mu2

        # SSIM公式
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        # ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq**2 + sigma2_sq**2 + c2))

        # 结构相似性指数
        return ssim_map.mean()
    
def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    # disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def SDLogJ(DVF):
    DVF = DVF.numpy()
    # DVF = np.transpose(DVF, (0, 2, 3, 4, 1))
    jacobian_det = jacobian_determinant_vxm(np.squeeze(DVF))
    return np.std(np.log(jacobian_det))

def PercentJ(DVF):
    DVF = DVF.numpy()
    # DVF = np.transpose(DVF, (0, 2, 3, 4, 1))
    jacobian_det = jacobian_determinant_vxm(np.squeeze(DVF))
    return np.sum(jacobian_det<0)/jacobian_det.size