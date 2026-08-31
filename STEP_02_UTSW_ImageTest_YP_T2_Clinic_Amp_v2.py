import os
import re
import glob
import numpy as np
import pydicom
from scipy import io
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import RectangleSelector
import imageio
import time
import peilin


bases = (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter)
p = argparse.ArgumentParser(
    formatter_class=type('formatter', bases, {}),
    description=f'FreeTune4D for UTSouthWestern',
)

p.add_argument('--phase_num', type=int, default=5, help='phase number')
p.add_argument('--base_path', type=str, default='/mnt/sda/Academics/Code/MyCode/UltraRecon-4D/DDEM.Liver/ReProductionDataset03Case0008/', help="base path of 3D/4D image")
p.add_argument('--MR_number', type=str, default='raw', help="MRN")
p.add_argument('--st_date', type=str, default="StDate", help='StDate')
arg = p.parse_args()

def BA_based_amplitude_sorting(FourD, opts):
    """
    基于体积面积（Body Area）的相位排序。
    FourD: 4D numpy array shape (X,Y,Z,frames)
    opts: dict with keys:
      coronal_index, ROI_cor_x, ROI_cor_y, val_threshold, no_phase_bins
    返回: FourD_ave of shape (X,Y,Z,Phase)
    """
    X, Y, Z, frames = FourD.shape
    ci = opts['coronal_index']
    ROIx = opts['ROI_cor_x']
    ROIy = opts['ROI_cor_y']
    thr = opts['val_threshold']
    Phase_num = opts['no_phase_bins']

    BodyArea = np.zeros(frames)
    for i in range(frames):
        cor_slice = FourD[ci, :, :, i]  # shape (Y,Z)
        roi = cor_slice[ROIx[:, None], ROIy]  # broadcast to 2D
        mask = (roi >= thr)
        BodyArea[i] = mask.sum()

    Rescaled = BodyArea - BodyArea.mean()
    NormBA = Rescaled / (abs(Rescaled.max()) + abs(Rescaled.min()) + 1e-12)
    plt.figure(); plt.plot(NormBA); plt.title('Normalized Body Area')

    AM_max, AM_min = NormBA.max(), NormBA.min()
    seq = np.linspace(AM_min, AM_max, Phase_num*2 + 1)
    PD = np.zeros_like(NormBA, dtype=int)
    for ph in range(Phase_num*2):
        mask = (NormBA >= seq[ph]) & (NormBA <= seq[ph+1])
        PD[mask] = ph + 1

    FourD_ave = np.zeros((X, Y, Z, Phase_num))
    for p in range(Phase_num):
        sel = np.where((PD == 2*p+1) | (PD == 2*p+2))[0]
        if sel.size:
            FourD_ave[:, :, :, p] = FourD[:, :, :, sel].mean(axis=3)
    plt.figure(); plt.hist(np.floor((PD+1)/2)); plt.title('Phase Histogram')
    return FourD_ave

# def load_image(path, time_index=0, pattern='IM-*', dims=None, nTP=None):
#     """
#     读取 DICOM 序列，返回：
#       - img: numpy 数组，shape 为 (n1, n2, n3[, nTP])
#       - pos: numpy 数组，存储每帧的物理坐标，shape 为 (3, nSlices * nTP)
#       - vs: 体素大小 [dx, dy, dz]
#     """
#     files = sorted(glob.glob(os.path.join(path, pattern + '.dcm')))
#     if not files:
#         raise FileNotFoundError(f"No DICOM files match {pattern} in {path}")

#     ds0 = pydicom.dcmread(files[0])
#     dx, dy = [float(x) for x in ds0.PixelSpacing]
#     dz = float(getattr(ds0, 'SliceThickness', 1.0))
#     vs = np.array([dx, dy, dz], dtype=float)

#     if nTP is not None and nTP > 1:
#         N = len(files)
#         n1, n2 = int(ds0.Rows), int(ds0.Columns)
#         n3 = N // nTP
#         img = np.zeros((n1, n2, n3, nTP), dtype=ds0.pixel_array.dtype)
#         pos = np.zeros((3, N), dtype=float)
#         for idx, f in enumerate(files):
#             ds = pydicom.dcmread(f)
#             sl = idx // nTP
#             tp = idx % nTP
#             img[:, :, sl, tp] = ds.pixel_array
#             pos[:, idx] = ds.ImagePositionPatient
#     else:
#         N = len(files)
#         n1, n2 = int(ds0.Rows), int(ds0.Columns)
#         n3 = N
#         img = np.zeros((n1, n2, n3), dtype=ds0.pixel_array.dtype)
#         pos = np.zeros((3, n3), dtype=float)
#         for sl, f in enumerate(files):
#             ds = pydicom.dcmread(f)
#             img[:, :, sl] = ds.pixel_array
#             pos[:, sl] = ds.ImagePositionPatient

#     if dims is not None and img.shape[:3] != tuple(dims):
#         raise ValueError(f"Expected image shape {dims}, got {img.shape[:3]}")

#     return img, pos, vs

def load_image(path, time_index=0, pattern='IM-*', dims=None, nTP=None):
    """
    读取 DICOM 序列，返回：
      - img: numpy 数组，shape 为 (n1, n2, n3[, nTP])
      - pos: numpy 数组，存储每帧的物理坐标，shape 为 (3, nSlices * nTP)
      - vs: 体素大小 [dx, dy, dz]
    """
    print("\n[load_image] --------------------------------------------------", flush=True)
    print(f"[load_image] path      = {path}", flush=True)
    print(f"[load_image] pattern   = {pattern}", flush=True)
    print(f"[load_image] nTP input = {nTP}", flush=True)

    files = sorted(glob.glob(os.path.join(path, pattern + '.dcm')))
    if not files:
        raise FileNotFoundError(f"No DICOM files match {pattern} in {path}")

    print(f"[load_image] number of dicom files = {len(files)}", flush=True)
    print(f"[load_image] first file = {files[0]}", flush=True)
    print(f"[load_image] last  file = {files[-1]}", flush=True)

    ds0 = pydicom.dcmread(files[0])
    dx, dy = [float(x) for x in ds0.PixelSpacing]
    dz = float(getattr(ds0, 'SliceThickness', 1.0))
    vs = np.array([dx, dy, dz], dtype=float)

    print(f"[load_image] voxel spacing = {vs}", flush=True)
    print(f"[load_image] rows, cols    = ({int(ds0.Rows)}, {int(ds0.Columns)})", flush=True)

    if nTP is not None and nTP > 1:
        N = len(files)
        n1, n2 = int(ds0.Rows), int(ds0.Columns)
        n3 = N // nTP
        img = np.zeros((n1, n2, n3, nTP), dtype=ds0.pixel_array.dtype)
        pos = np.zeros((3, N), dtype=float)

        print(f"[load_image] detected 4D series -> shape will be ({n1}, {n2}, {n3}, {nTP})", flush=True)

        for idx, f in enumerate(files):
            ds = pydicom.dcmread(f)
            sl = idx // nTP
            tp = idx % nTP
            img[:, :, sl, tp] = ds.pixel_array
            pos[:, idx] = ds.ImagePositionPatient

            if idx == 0 or idx == len(files)-1:
                print(f"[load_image] reading idx={idx}, slice={sl}, tp={tp}", flush=True)
    else:
        N = len(files)
        n1, n2 = int(ds0.Rows), int(ds0.Columns)
        n3 = N
        img = np.zeros((n1, n2, n3), dtype=ds0.pixel_array.dtype)
        pos = np.zeros((3, n3), dtype=float)

        print(f"[load_image] detected 3D series -> shape will be ({n1}, {n2}, {n3})", flush=True)

        for sl, f in enumerate(files):
            ds = pydicom.dcmread(f)
            img[:, :, sl] = ds.pixel_array
            pos[:, sl] = ds.ImagePositionPatient

            if sl == 0 or sl == len(files)-1:
                print(f"[load_image] reading slice={sl}", flush=True)

    if dims is not None and img.shape[:3] != tuple(dims):
        raise ValueError(f"Expected image shape {dims}, got {img.shape[:3]}")

    print(f"[load_image] final image shape = {img.shape}", flush=True)
    print(f"[load_image] final pos shape   = {pos.shape}", flush=True)
    print("[load_image] done.", flush=True)

    return img, pos, vs

def imcrop(img, output_size):
    pad = np.array(img.shape) - np.array(output_size)
    start = np.round(pad / 2).astype(int)
    end = start + np.array(output_size)
    return img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

def GIFplot2(volume, filepath, duration, intensity_range, cmap='gray'):
    """
    volume: 3D numpy array (H, W, T)
    duration: seconds per frame
    intensity_range: (vmin, vmax)
    """
    vmin, vmax = intensity_range
    frames = []
    for i in range(volume.shape[2]):
        frame = volume[:, :, i]
        frame = np.clip(frame, vmin, vmax)
        frame = ((frame - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        frames.append(frame)
    imageio.mimsave(filepath, frames, format='GIF', duration=duration)

def robust_normalize(img, p_low=1, p_high=99):
    """
    为显示做稳健归一化，避免极端值影响可视化。
    返回归一化到 [0,1] 的图像，以及原始显示范围 lo/hi。
    """
    arr = img.astype(np.float32)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)

    if hi <= lo:
        lo = arr.min()
        hi = arr.max()

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return arr, float(lo), float(hi)


def save_plot_3d(img, name, path, dpi=300):
    """
    用 peilin.plot_3DLiver 从三个方向保存一张 3D 体数据展示图。
    注意：peilin.plot_3DLiver 内部会改动切片像素，所以这里传 copy。
    """
    img_norm, lo, hi = robust_normalize(img)
    print(f"[vis] save_plot_3d -> {name}, shape={img.shape}, display_range=({lo:.3f}, {hi:.3f})", flush=True)

    peilin.plot_3DLiver(
        img_norm.copy(),
        name=name,
        titles=["Axial Plane", "Coronal Plane", "Sagittal Plane"],
        path=path,
        max=1,
        min=0,
        dpi=dpi
    )


def save_abs_diff_plot(ref_img, mov_img, name, path, dpi=300):
    """
    保存两个 3D 图像的绝对差值图。
    """
    ref_norm, _, _ = robust_normalize(ref_img)
    mov_norm, _, _ = robust_normalize(mov_img)

    diff = np.abs(ref_norm - mov_norm)
    print(f"[vis] save_abs_diff_plot -> {name}, shape={diff.shape}, diff_range=({diff.min():.6f}, {diff.max():.6f})", flush=True)

    peilin.plot_3DLiver(
        diff.copy(),
        name=name,
        titles=["Axial Diff", "Coronal Diff", "Sagittal Diff"],
        path=path,
        max=1,
        min=0,
        dpi=dpi
    )


def save_match_triplet(ref_img, mov_img, prefix, path, score=None, dpi=300):
    """
    同时保存：
      1) ref 图
      2) mov 图
      3) abs diff 图
    """
    score_str = "" if score is None else f"_LCC_{score:.4f}"

    save_plot_3d(ref_img, f"{prefix}_FourDRef{score_str}", path, dpi=dpi)
    save_plot_3d(mov_img, f"{prefix}_T2{score_str}", path, dpi=dpi)
    save_abs_diff_plot(ref_img, mov_img, f"{prefix}_AbsDiff{score_str}", path, dpi=dpi)


def plot_lcc_curve(scores, path, name="axial_lcc_curve", dpi=200):
    """
    保存 axial 滑窗搜索时每个位置的 LCC 曲线。
    """
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(8, 4), dpi=dpi)
    plt.plot(np.arange(len(scores)), scores, marker='o', linewidth=1)
    plt.xlabel("Axial Candidate Index")
    plt.ylabel("LCC Score")
    plt.title("Axial Sliding LCC Curve")
    plt.grid(True, linestyle='--', alpha=0.4)
    save_path = os.path.join(path, f"{name}.jpg")
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"[vis] saved LCC curve -> {save_path}", flush=True)

def center_crop_or_pad_2d_to_shape(vol, target_x, target_y):
    print("\n[center_crop_or_pad_2d_to_shape] ------------------------------", flush=True)
    print(f"[center_crop_or_pad_2d_to_shape] input shape  = {vol.shape}", flush=True)
    print(f"[center_crop_or_pad_2d_to_shape] target shape = ({target_x}, {target_y}, {vol.shape[2]})", flush=True)

    x, y, z = vol.shape
    out = np.zeros((target_x, target_y, z), dtype=vol.dtype)

    if x >= target_x:
        xs0 = (x - target_x) // 2
        xs1 = xs0 + target_x
        xd0 = 0
        xd1 = target_x
    else:
        xs0 = 0
        xs1 = x
        xd0 = (target_x - x) // 2
        xd1 = xd0 + x

    if y >= target_y:
        ys0 = (y - target_y) // 2
        ys1 = ys0 + target_y
        yd0 = 0
        yd1 = target_y
    else:
        ys0 = 0
        ys1 = y
        yd0 = (target_y - y) // 2
        yd1 = yd0 + y

    print(f"[center_crop_or_pad_2d_to_shape] source x: [{xs0}:{xs1}], dest x: [{xd0}:{xd1}]", flush=True)
    print(f"[center_crop_or_pad_2d_to_shape] source y: [{ys0}:{ys1}], dest y: [{yd0}:{yd1}]", flush=True)

    out[xd0:xd1, yd0:yd1, :] = vol[xs0:xs1, ys0:ys1, :]

    print(f"[center_crop_or_pad_2d_to_shape] output shape = {out.shape}", flush=True)
    return out

def lcc_score_3d(vol1, vol2, device="cuda:0", eps=1e-8):
    """
    对两个 shape 相同的 3D 体数据计算 LCC/NCC 分数。
    注意：
      - peilin_loss.NCC(...).loss() 当前返回的是“负数 loss”
      - 因此这里乘以 -1，转成“越大越相似”的 score
    """
    assert vol1.shape == vol2.shape, f"Shape mismatch: {vol1.shape} vs {vol2.shape}"

    v1 = torch.tensor(vol1.astype(np.float32))[None, None, ...]
    v2 = torch.tensor(vol2.astype(np.float32))[None, None, ...]

    # 归一化，减少强度量纲差异影响
    v1 = (v1 - v1.min()) / (v1.max() - v1.min() + eps)
    v2 = (v2 - v2.min()) / (v2.max() - v2.min() + eps)

    loss = peilin_loss.NCC(device=device).loss(v1.to(device), v2.to(device))
    loss = float(np.mean(loss.detach().cpu().numpy()))

    # 关键修改：loss 是负数，乘 -1 变成“越大越好”的相似性 score
    score = -loss
    return score

def lcc_score_overlap_only(fd_ref, t2_xy_aligned, t2_offset, device="cuda:0", min_overlap_ratio=0.5):
    """
    只对 4D 和 T2 当前真正重合的 axial 部分计算 LCC。

    参数
    ----
    fd_ref : np.ndarray
        shape = (X, Y, Zf)
    t2_xy_aligned : np.ndarray
        shape = (X, Y, Zt)
    t2_offset : int
        T2 在全局 axial 坐标中的起点（相对于 4D 的 global z=0）
        例如：
          t2_offset = 0   -> T2 的第 0 层与 4D 的第 0 层对齐
          t2_offset = -5  -> T2 比 4D 更“靠前”，前 5 层在 4D 外面
          t2_offset = 10  -> T2 从 4D 的第 10 层位置开始重叠
    min_overlap_ratio : float
        最小重合比例。这里用 0.5，即至少达到两者中较小 FOV 的 1/2。

    返回
    ----
    score, fd_z0, fd_z1, t2_z0, t2_z1, overlap_len, min_required
    若当前 offset 不满足最小重合长度要求，则返回：
        -np.inf, None, None, None, None, overlap_len, min_required
    """
    zf = fd_ref.shape[2]
    zt = t2_xy_aligned.shape[2]

    # 两者中更小 FOV 的一半，作为最小合法重合长度
    min_required = int(np.ceil(min(zf, zt) * min_overlap_ratio))

    # 全局坐标下：
    # 4D 占据 [0, zf)
    # T2 占据 [t2_offset, t2_offset + zt)
    fd_global_start = 0
    fd_global_end = zf

    t2_global_start = t2_offset
    t2_global_end = t2_offset + zt

    overlap_start = max(fd_global_start, t2_global_start)
    overlap_end = min(fd_global_end, t2_global_end)
    overlap_len = overlap_end - overlap_start

    # 新规则：若重合部分不到两者中较小 FOV 的 1/2，则直接丢弃
    if overlap_len < min_required:
        return -np.inf, None, None, None, None, overlap_len, min_required

    # 映射回各自局部坐标
    fd_z0 = overlap_start - fd_global_start
    fd_z1 = overlap_end - fd_global_start

    t2_z0 = overlap_start - t2_global_start
    t2_z1 = overlap_end - t2_global_start

    fd_part = fd_ref[:, :, fd_z0:fd_z1]
    t2_part = t2_xy_aligned[:, :, t2_z0:t2_z1]

    if fd_part.shape != t2_part.shape:
        return -np.inf, None, None, None, None, overlap_len, min_required

    score = lcc_score_3d(fd_part, t2_part, device=device)
    return score, fd_z0, fd_z1, t2_z0, t2_z1, overlap_len, min_required
def axial_match_by_lcc(fd_crop, t2_xy_aligned, device="cuda:0", vis_path=None, prefix="axial_match"):
    print("\n[axial_match_by_lcc] ==========================================", flush=True)
    print(f"[axial_match_by_lcc] fd_crop shape       = {fd_crop.shape}", flush=True)
    print(f"[axial_match_by_lcc] t2_xy_aligned shape = {t2_xy_aligned.shape}", flush=True)

    fd_ref = fd_crop.mean(axis=3)
    zf = fd_ref.shape[2]
    zt = t2_xy_aligned.shape[2]

    min_required = int(np.ceil(0.5 * min(zf, zt)))

    print(f"[axial_match_by_lcc] fd_ref shape = {fd_ref.shape}", flush=True)
    print(f"[axial_match_by_lcc] axial length -> fd={zf}, t2={zt}", flush=True)
    print(f"[axial_match_by_lcc] minimum required overlap = {min_required}", flush=True)

    if vis_path is not None:
        os.makedirs(vis_path, exist_ok=True)
        save_plot_3d(fd_ref, f"{prefix}_00_FourDRef_before_match", vis_path)
        save_plot_3d(t2_xy_aligned, f"{prefix}_01_T2_before_match_full", vis_path)

    best_score = -np.inf
    best_offset = None
    best_fd_z0, best_fd_z1 = None, None
    best_t2_z0, best_t2_z1 = None, None

    all_scores = []
    all_offsets = []

    # 合法 offset 范围：
    # 至少要保证 overlap_len >= min_required
    offset_min = -(zt - min_required)
    offset_max = zf - min_required

    total_candidates = offset_max - offset_min + 1
    print(f"[axial_match_by_lcc] offset range = [{offset_min}, {offset_max}]", flush=True)
    print(f"[axial_match_by_lcc] total candidates = {total_candidates}", flush=True)

    # baseline：让 T2 和 4D 在 axial 上尽量居中对齐
    center_offset = int(np.clip((zf - zt) // 2, offset_min, offset_max))

    baseline_score, fd_z0, fd_z1, t2_z0, t2_z1, overlap_len, _ = lcc_score_overlap_only(
        fd_ref, t2_xy_aligned, center_offset, device=device, min_overlap_ratio=0.5
    )

    if fd_z0 is not None and vis_path is not None:
        baseline_ref = fd_ref[:, :, fd_z0:fd_z1]
        baseline_mov = t2_xy_aligned[:, :, t2_z0:t2_z1]

        print(
            f"[axial_match_by_lcc] baseline center offset = {center_offset}, "
            f"overlap fd[{fd_z0}:{fd_z1}] vs t2[{t2_z0}:{t2_z1}], "
            f"overlap_len = {overlap_len}, score = {baseline_score:.6f}",
            flush=True
        )

        save_match_triplet(
            baseline_ref,
            baseline_mov,
            f"{prefix}_02_before_match_centerCandidate",
            vis_path,
            score=baseline_score
        )

    for offset in range(offset_min, offset_max + 1):
        score, fd_z0, fd_z1, t2_z0, t2_z1, overlap_len, min_required = lcc_score_overlap_only(
            fd_ref, t2_xy_aligned, offset, device=device, min_overlap_ratio=0.5
        )

        all_offsets.append(offset)
        all_scores.append(score)

        if fd_z0 is None:
            print(
                f"[axial_match_by_lcc] offset = {offset}, "
                f"overlap_len = {overlap_len} < min_required = {min_required}, skipped.",
                flush=True
            )
            continue

        print(
            f"[axial_match_by_lcc] offset = {offset}, "
            f"overlap fd[{fd_z0}:{fd_z1}] vs t2[{t2_z0}:{t2_z1}], "
            f"overlap_len = {overlap_len}, score = {score:.6f}",
            flush=True
        )

        if score > best_score:
            best_score = score
            best_offset = offset
            best_fd_z0, best_fd_z1 = fd_z0, fd_z1
            best_t2_z0, best_t2_z1 = t2_z0, t2_z1

            print(
                f"[axial_match_by_lcc] --> new best: score={best_score:.6f}, "
                f"offset={best_offset}, "
                f"fd[{best_fd_z0}:{best_fd_z1}] vs t2[{best_t2_z0}:{best_t2_z1}]",
                flush=True
            )

    if best_fd_z0 is None:
        raise RuntimeError(
            f"No valid axial overlap found. Need overlap >= {min_required}, "
            f"but no candidate satisfied the condition."
        )

    # 最终输出：两者都裁成“最佳重合区域”
    fd_final = fd_crop[:, :, best_fd_z0:best_fd_z1, :]
    t2_final = t2_xy_aligned[:, :, best_t2_z0:best_t2_z1]

    fd_final_ref = fd_final.mean(axis=3)

    if vis_path is not None:
        save_match_triplet(
            fd_final_ref,
            t2_final,
            f"{prefix}_03_after_match_bestCandidate",
            vis_path,
            score=best_score
        )
        plot_lcc_curve(all_scores, vis_path, name=f"{prefix}_04_lcc_curve")

    print(f"[axial_match_by_lcc] final best_score    = {best_score:.6f}", flush=True)
    print(f"[axial_match_by_lcc] best_offset         = {best_offset}", flush=True)
    print(f"[axial_match_by_lcc] best overlap fd     = [{best_fd_z0}:{best_fd_z1}]", flush=True)
    print(f"[axial_match_by_lcc] best overlap t2     = [{best_t2_z0}:{best_t2_z1}]", flush=True)
    print(f"[axial_match_by_lcc] final overlap len   = {best_fd_z1 - best_fd_z0}", flush=True)
    print(f"[axial_match_by_lcc] fd_final shape      = {fd_final.shape}", flush=True)
    print(f"[axial_match_by_lcc] t2_final shape      = {t2_final.shape}", flush=True)

    # 为了兼容你原先主程序的接收变量名，这里仍然返回 5 个值
    # best_fd_start / best_t2_start 现在表示“最终裁剪区域在各自 volume 中的起始 z”
    best_fd_start = best_fd_z0
    best_t2_start = best_t2_z0

    return fd_final, t2_final, best_score, best_fd_start, best_t2_start

import torch
import math
from sklearn.cluster import AgglomerativeClustering, KMeans
import peilin_loss
def clustering(imgs_tmp, class_num=3, path="./tmp"):
    print("\n[clustering] ==================================================", flush=True)
    print(f"[clustering] input shape = {imgs_tmp.shape}", flush=True)
    print(f"[clustering] class_num   = {class_num}", flush=True)
    print(f"[clustering] save path   = {path}", flush=True)

    imgs_tmp = torch.tensor(imgs_tmp.astype(np.float32))
    imgs = (imgs_tmp - torch.min(imgs_tmp)) / (torch.max(imgs_tmp) - torch.min(imgs_tmp) + 1e-8)

    os.makedirs(path, exist_ok=True)
    batch_size = 20
    gl_device = "cuda:0"

    image_num = imgs.shape[-1]
    matrix = np.ones((image_num, image_num))
    index1 = []
    index2 = []
    index_for_index = []

    for i in range(image_num):
        for j in range(i, image_num):
            index1.append(i)
            index2.append(j)
            index_for_index.append((i, j))

    total_pairs = len(index2)
    total_batches = math.ceil(len(index1) / batch_size)

    print(f"[clustering] image_num   = {image_num}", flush=True)
    print(f"[clustering] total_pairs = {total_pairs}", flush=True)
    print(f"[clustering] batch_size  = {batch_size}", flush=True)
    print(f"[clustering] total_batches = {total_batches}", flush=True)

    metrics = np.ones(len(index2))

    for i in range(total_batches):
        st = i * batch_size
        ed = min((i + 1) * batch_size, len(index1))

        print(f"[clustering] computing batch {i+1}/{total_batches}, pair index [{st}:{ed})", flush=True)

        imgs1 = imgs[..., index1[st:ed]].permute(3, 0, 1, 2).float()
        imgs2 = imgs[..., index2[st:ed]].permute(3, 0, 1, 2).float()

        LCC = peilin_loss.NCC(device=gl_device).loss(
            imgs1[:, None, ...].to(gl_device),
            imgs2[:, None, ...].to(gl_device)
        ).to("cpu").numpy()

        metrics[st:ed] = -LCC

        print(f"[clustering] batch {i+1} finished, metric range = ({metrics[st:ed].min():.6f}, {metrics[st:ed].max():.6f})", flush=True)

    print("[clustering] filling symmetric matrix...", flush=True)
    for i in range(image_num):
        for j in range(i, image_num):
            index = index_for_index.index((i, j))
            matrix[i, j] = float(metrics[index])
            matrix[j, i] = float(metrics[index])

    np.savez(f"./{path}/matrix.npz", matrix=matrix, metrics=metrics)
    print(f"[clustering] saved matrix to ./{path}/matrix.npz", flush=True)

    print("[clustering] running KMeans...", flush=True)
    clustering_model = KMeans(n_clusters=class_num, random_state=0, n_init=10)
    labels = clustering_model.fit_predict(matrix)

    clusters = {i: [] for i in range(class_num)}
    for index, label in zip(range(image_num), labels):
        clusters[label].append(index)

    print("[clustering] raw clusters:", flush=True)
    for k, v in clusters.items():
        print(f"  cluster {k}: size={len(v)}, members={v}", flush=True)

    sorting_matrix = np.zeros((class_num, class_num))
    for i in range(class_num):
        for j in range(i+1, class_num):
            lcc = 0
            for index_i in clusters[i]:
                for index_j in clusters[j]:
                    lcc += matrix[index_i, index_j]
            lcc /= (len(clusters[i]) * len(clusters[j]))
            sorting_matrix[i, j] = lcc
            sorting_matrix[j, i] = lcc

    print(f"[clustering] sorting_matrix:\n{sorting_matrix}", flush=True)

    sorted_clusters = []
    while len(sorted_clusters) < class_num:
        if len(sorted_clusters) == 0:
            sum_sorting_matrix = sorting_matrix.sum(axis=0)
            lcc_index = np.argmin(sum_sorting_matrix)
        else:
            sum_sorting_matrix = np.zeros(class_num)
            for i in range(len(sorted_clusters)):
                sum_sorting_matrix += sorting_matrix[:, sorted_clusters[i]]
            sum_sorting_matrix /= len(sorted_clusters)
            lcc_index = np.argmax(sum_sorting_matrix)
            assert lcc_index not in sorted_clusters, "{} clustering... lcc_index has been included...".format(time.ctime())

        sorted_clusters.append(lcc_index)
        print(f"[clustering] sorted_clusters now = {sorted_clusters}", flush=True)

    sorted_imgs = []
    selected_rep_indices = []

    for i, index in enumerate(clusters.keys()):
        cluster_id = sorted_clusters[index]
        members = clusters[cluster_id]

        if i == 0 or i == (len(clusters.keys()) - 1):
            lcc = []
            for j in range(len(members)):
                lcc.append(np.mean(matrix[members[j], ...]))
            rep_idx = members[lcc.index(min(lcc))]
        else:
            rep_idx = members[0]

        selected_rep_indices.append(rep_idx)
        sorted_imgs.append(imgs_tmp[..., rep_idx, None])

        print(f"[clustering] output phase {i}: from cluster {cluster_id}, representative frame = {rep_idx}", flush=True)

    sorted_imgs = torch.concat(sorted_imgs, dim=-1)

    np.savez(f"./{path}/collected_4d.npz", sorted=sorted_imgs.numpy(), clusters=clusters)
    print(f"[clustering] saved collected 4D to ./{path}/collected_4d.npz", flush=True)
    print(f"[clustering] final sorted shape = {sorted_imgs.shape}", flush=True)
    print(f"[clustering] representative indices = {selected_rep_indices}", flush=True)
    print("[clustering] done.", flush=True)

    return clusters, sorted_imgs.numpy()

if __name__ == '__main__':
    print("\n==================== Script Start ====================", flush=True)
    print(f"[main] phase_num = {arg.phase_num}", flush=True)
    print(f"[main] base_path = {arg.base_path}", flush=True)
    print(f"[main] MR_number = {arg.MR_number}", flush=True)
    print(f"[main] st_date   = {arg.st_date}", flush=True)

    # --- User Input and Paths ---
    MRN = arg.MR_number
    script_dir = arg.base_path
    os.chdir(script_dir)

    data_base = arg.base_path
    patient_path0 = os.path.join(data_base, MRN)
    StDate = arg.st_date
    patient_path = os.path.join(patient_path0, StDate)
    os.chdir(patient_path)

    print(f"[main] patient_path0 = {patient_path0}", flush=True)
    print(f"[main] patient_path  = {patient_path}", flush=True)
    
    vis_root = os.path.join(patient_path, "debug_vis")
    os.makedirs(vis_root, exist_ok=True)
    print(f"[main] visualization root = {vis_root}", flush=True)

    # --- Load 4D DICOM Images ---
    print("\n[main] searching THRIVE folder...", flush=True)
    thr_dirs = glob.glob(os.path.join(patient_path, '*THRIVE*'))
    if not thr_dirs:
        raise FileNotFoundError('No THRIVE folder found')
    FourD_path = thr_dirs[0]
    print(f"[main] FourD_path = {FourD_path}", flush=True)

    dcm_files = sorted(glob.glob(os.path.join(FourD_path, '*.dcm')))
    print(f"[main] number of THRIVE DICOM files = {len(dcm_files)}", flush=True)
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in dynamic DICOM directory: {FourD_path}")

    info = pydicom.dcmread(dcm_files[0])
    nTP = int(info.NumberOfTemporalPositions)
    print(f"[main] NumberOfTemporalPositions = {nTP}", flush=True)
    if nTP <= 0:
        raise ValueError(f"NumberOfTemporalPositions must be positive; got {nTP}.")
    if len(dcm_files) % nTP != 0:
        raise ValueError(
            f"Dynamic DICOM file count {len(dcm_files)} is not divisible by "
            f"NumberOfTemporalPositions {nTP}; slice/time ordering is incomplete."
        )

    FourD, FourD_pos, FourD_vs = load_image(FourD_path, time_index=0, pattern='IM-*', dims=None, nTP=nTP)
    print(f"[main] FourD shape = {FourD.shape}", flush=True)
    print(f"[main] FourD_pos shape = {FourD_pos.shape}", flush=True)
    print(f"[main] FourD_vs = {FourD_vs}", flush=True)

    # Lightweight diagnostics before the first CUDA operation (NCC clustering).
    print("\n[preflight] dynamic DICOM / CUDA validation", flush=True)
    print(f"[preflight] selected modality       = T2", flush=True)
    print(f"[preflight] DICOM files             = {len(dcm_files)}", flush=True)
    print(f"[preflight] temporal positions      = {nTP}", flush=True)
    print(f"[preflight] loaded volume shape     = {FourD.shape}", flush=True)
    print(f"[preflight] loaded volume dtype     = {FourD.dtype}", flush=True)
    print(f"[preflight] loaded intensity range  = ({np.min(FourD)}, {np.max(FourD)})", flush=True)
    print(f"[preflight] requested output phases = {arg.phase_num}", flush=True)
    print(f"[preflight] CUDA device             = cuda:0", flush=True)
    if FourD.ndim != 4 or any(size <= 0 for size in FourD.shape):
        raise ValueError(f"Dynamic DICOM must produce a non-empty 4D volume; got shape {FourD.shape}.")
    if FourD.shape[-1] != nTP:
        raise ValueError(f"Loaded temporal dimension {FourD.shape[-1]} does not match NumberOfTemporalPositions {nTP}.")
    if FourD.shape[-1] < arg.phase_num:
        raise ValueError(f"Cannot create {arg.phase_num} phases from only {FourD.shape[-1]} temporal frames.")
    if not np.isfinite(FourD).all():
        raise ValueError("Dynamic DICOM volume contains NaN or Inf values before CUDA clustering.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device cuda:0 is required by preprocessing but is not available.")
    print(f"[preflight] CUDA device name        = {torch.cuda.get_device_name(0)}", flush=True)
    print("[preflight] dynamic input is valid for NCC clustering", flush=True)

    FourD_all = FourD.copy()
    full_data_path = os.path.join(patient_path, 'Full_data.mat')
    io.savemat(full_data_path, {'FourD': FourD_all}, do_compression=True)
    print(f"[main] saved Full_data.mat -> {full_data_path}", flush=True)

    slice_idx = int(FourD.shape[0] / 2)
    opts = {
        'coronal_index': slice_idx,
        'val_threshold': 30,
        'no_phase_bins': arg.phase_num
    }
    print(f"[main] opts = {opts}", flush=True)

    # --- Clustering ---
    FourD_work = FourD.copy()
    clusted_index, FourD_ave = clustering(FourD_work, class_num=opts['no_phase_bins'], path="./clustering")
    print(f"[main] clustering done. FourD_ave shape = {FourD_ave.shape}", flush=True)
    print(f"[main] clustering index summary:", flush=True)
    for k, v in clusted_index.items():
        print(f"  cluster {k}: size={len(v)}, members={v}", flush=True)

    # --- Load the 3D T2 Image ---
    print("\n[main] searching T2 folder...", flush=True)
    t2_dirs = glob.glob(os.path.join(patient_path, '*T2_AX_MVXD*'))
    if not t2_dirs:
        raise FileNotFoundError('No T2_AX_MVXD folder found')
    T2_path = t2_dirs[0]
    print(f"[main] T2_path = {T2_path}", flush=True)

    T2, T2_pos, T2_vs = load_image(T2_path, time_index=2, pattern='IM-*')
    print(f"[main] T2 shape = {T2.shape}", flush=True)
    print(f"[main] T2_pos shape = {T2_pos.shape}", flush=True)
    print(f"[main] T2_vs = {T2_vs}", flush=True)
    print(f"[preflight] static volume shape     = {T2.shape}", flush=True)
    print(f"[preflight] static volume dtype     = {T2.dtype}", flush=True)
    print(f"[preflight] static intensity range  = ({np.min(T2)}, {np.max(T2)})", flush=True)
    if T2.ndim != 3 or any(size <= 0 for size in T2.shape):
        raise ValueError(f"Static DICOM must produce a non-empty 3D volume; got shape {T2.shape}.")
    if not np.isfinite(T2).all():
        raise ValueError("Static DICOM volume contains NaN or Inf values before CUDA axial matching.")

    # --- Save GIFs for Demo ---
    print("\n[main] generating GIFs...", flush=True)

    gif_orig = np.flip(
        np.rot90(FourD[opts['coronal_index'], :, :, :], k=1, axes=(0,1)),
        axis=0
    )
    gif_orig_resized = zoom(gif_orig, (FourD_vs[2], FourD_vs[1], 1), order=1)
    gif_orig_path = os.path.join(patient_path, '4D_AX_T2.gif')
    GIFplot2(
        gif_orig_resized,
        gif_orig_path,
        duration=0.3,
        intensity_range=(0, 0.8 * gif_orig_resized.max())
    )
    print(f"[main] saved original GIF -> {gif_orig_path}, shape={gif_orig_resized.shape}", flush=True)

    gif_avg = np.flip(
        np.rot90(FourD_ave[opts['coronal_index'], :, :, :], k=1, axes=(0,1)),
        axis=0
    )
    gif_avg_resized = zoom(gif_avg, (FourD_vs[2], FourD_vs[1], 1), order=1)
    gif_avg_path = os.path.join(patient_path, '4D_AX_ave_T2.gif')
    GIFplot2(
        gif_avg_resized,
        gif_avg_path,
        duration=0.3,
        intensity_range=(0, 0.8 * gif_avg_resized.max())
    )
    print(f"[main] saved averaged GIF -> {gif_avg_path}, shape={gif_avg_resized.shape}", flush=True)

    # --- Image Alignment & Cropping ---
    print("\n[main] computing physical ranges...", flush=True)
    n1, n2, n3, _ = FourD.shape
    slice_idxs = np.arange(0, FourD_pos.shape[1], nTP)
    coords = FourD_pos[:, slice_idxs]

    FourD_range_min = np.array([coords[0].min(), coords[1].min(), coords[2].min()])
    FourD_range_max = np.array([
        coords[0].min() + FourD_vs[0] * n1,
        coords[1].min() + FourD_vs[1] * n2,
        coords[2].max()
    ])

    coords2 = T2_pos
    T2_range_min = np.array([coords2[0].min(), coords2[1].min(), coords2[2].min()])
    T2_range_max = np.array([
        coords2[0].min() + T2_vs[0] * T2.shape[0],
        coords2[1].min() + T2_vs[1] * T2.shape[1],
        coords2[2].max()
    ])

    print(f"[main] FourD_range_min = {FourD_range_min}", flush=True)
    print(f"[main] FourD_range_max = {FourD_range_max}", flush=True)
    print(f"[main] T2_range_min    = {T2_range_min}", flush=True)
    print(f"[main] T2_range_max    = {T2_range_max}", flush=True)

    common_min = np.maximum.reduce([FourD_range_min, T2_range_min])
    common_max = np.minimum.reduce([FourD_range_max, T2_range_max])

    print(f"[main] common_min = {common_min}", flush=True)
    print(f"[main] common_max = {common_max}", flush=True)

    # --- Resample FourD_ave to Isotropic Voxels ---
    print("\n[main] resampling FourD_ave to isotropic...", flush=True)
    FourD_ave_iso = np.stack([
        zoom(FourD_ave[..., i], FourD_vs, order=1)
        for i in range(FourD_ave.shape[3])
    ], axis=3)
    print(f"[main] FourD_ave_iso shape = {FourD_ave_iso.shape}", flush=True)

    # --- Crop FourD_ave_iso ---
    start_fd = np.round((common_min - FourD_range_min) / FourD_vs).astype(int)
    end_fd = np.round((FourD_range_max - common_max) / FourD_vs).astype(int)

    print(f"[main] start_fd = {start_fd}", flush=True)
    print(f"[main] end_fd   = {end_fd}", flush=True)

    fd_crop = FourD_ave_iso[
        start_fd[0]:FourD_ave_iso.shape[0]-end_fd[0],
        start_fd[1]:FourD_ave_iso.shape[1]-end_fd[1],
        start_fd[2]:FourD_ave_iso.shape[2]-end_fd[2],
        :
    ]
    print(f"[main] fd_crop shape = {fd_crop.shape}", flush=True)

    fd_crop_ref = fd_crop.mean(axis=3)
    save_plot_3d(fd_crop_ref, "01_FourDRef_after_common_crop", vis_root)

    # --- Resample & Crop T2 ---
    print("\n[main] resampling T2 to isotropic...", flush=True)
    T2_iso = zoom(T2, T2_vs, order=1)
    print(f"[main] T2_iso shape = {T2_iso.shape}", flush=True)

    start_t2 = np.round((common_min - T2_range_min) / T2_vs).astype(int)
    end_t2 = np.round((T2_range_max - common_max) / T2_vs).astype(int)

    print(f"[main] start_t2 = {start_t2}", flush=True)
    print(f"[main] end_t2   = {end_t2}", flush=True)

    t2_crop = T2_iso[
        start_t2[0]:T2_iso.shape[0]-end_t2[0],
        start_t2[1]:T2_iso.shape[1]-end_t2[1],
        start_t2[2]:T2_iso.shape[2]-end_t2[2]
    ]
    print(f"[main] t2_crop shape = {t2_crop.shape}", flush=True)
    save_plot_3d(t2_crop, "02_T2_after_common_crop", vis_root)

    # --- Align T2 to 4D in sagittal/coronal only (X/Y) ---
    target_x, target_y = fd_crop.shape[0], fd_crop.shape[1]
    print(f"\n[main] aligning T2 in X/Y only to target_x={target_x}, target_y={target_y}", flush=True)
    t2_xy_aligned = center_crop_or_pad_2d_to_shape(t2_crop, target_x, target_y)
    print(f"[main] t2_xy_aligned shape = {t2_xy_aligned.shape}", flush=True)
    save_plot_3d(t2_xy_aligned, "03_T2_after_xy_align", vis_root)

    # --- Axial matching by LCC ---
    axial_vis_path = os.path.join(vis_root, "axial_match")
    fd_aligned, t2_aligned, best_lcc, best_fd_start, best_t2_start = axial_match_by_lcc(
        fd_crop,
        t2_xy_aligned,
        device="cuda:0",
        vis_path=axial_vis_path,
        prefix="axial_match"
    )

    print(f"[main] best axial LCC = {best_lcc:.6f}", flush=True)
    print(f"[main] best_fd_start  = {best_fd_start}", flush=True)
    print(f"[main] best_t2_start  = {best_t2_start}", flush=True)

    save_plot_3d(fd_aligned.mean(axis=3), "04_FourD_after_axial_match", vis_root)
    save_plot_3d(t2_aligned, "05_T2_after_axial_match", vis_root)
    save_abs_diff_plot(fd_aligned.mean(axis=3), t2_aligned, "06_AbsDiff_after_axial_match", vis_root)

    # --- Final Crop & Save .mat ---
    margin = 1
    print(f"\n[main] applying final margin crop, margin = {margin}", flush=True)

    if min(fd_aligned.shape[0], fd_aligned.shape[1], fd_aligned.shape[2],
           t2_aligned.shape[0], t2_aligned.shape[1], t2_aligned.shape[2]) <= 2 * margin:
        raise ValueError("margin too large after alignment/cropping.")

    T2_save = t2_aligned[
        margin:-margin,
        margin:-margin,
        margin:-margin
    ]

    FourD_ave_save = fd_aligned[
        margin:-margin,
        margin:-margin,
        margin:-margin,
        :
    ].astype(np.uint16)

    print(f"[main] T2_save shape        = {T2_save.shape}", flush=True)
    print(f"[main] FourD_ave_save shape = {FourD_ave_save.shape}", flush=True)

    save_path = os.path.join(patient_path, 'phase_T2.mat')
    io.savemat(
        save_path,
        {
            'T2_save': T2_save,
            'FourD_ave_save': FourD_ave_save,
            'best_lcc': best_lcc,
            'best_fd_start': best_fd_start,
            'best_t2_start': best_t2_start,
        },
        do_compression=True
    )
    print(f"[main] saved final mat -> {save_path}", flush=True)

    # --- Save All Open Figures ---
    print("\n[main] saving all opened figures...", flush=True)
    for idx, num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(num)
        fig_path = os.path.join(patient_path, f'T2_figure_{idx}.png')
        fig.savefig(fig_path)
        print(f"[main] saved figure {idx} -> {fig_path}", flush=True)

    # --- Create Output Directory ---
    uq_dir = os.path.join(data_base, MRN, StDate, 'UQ_4D_T2')
    os.makedirs(uq_dir, exist_ok=True)
    print(f"[main] ensured output directory exists -> {uq_dir}", flush=True)

    # --- Final Messages ---
    print("\n==================== Script Finished ====================", flush=True)
    print('Next step: Elastix Registration with Python!', flush=True)
    print('MRN:', MRN, flush=True)
    print('StDate:', StDate, flush=True)

    file_names = os.listdir(T2_path)
    if len(file_names) >= 3:
        print('Third file in T2 directory:', file_names[2], flush=True)