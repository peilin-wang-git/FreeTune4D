import os
import glob
import math
import time

import argparse
import imageio
import matplotlib
import numpy as np
import pydicom
import torch
from sklearn.cluster import KMeans
from scipy import io
from scipy.ndimage import zoom

matplotlib.use("Agg")  # 使用Agg后端
import matplotlib.pyplot as plt

import peilin
import peilin_loss

# Pipeline switches. Defaults reproduce the original pipeline.
ENABLE_SORTING = True
ENABLE_XY_ALIGNMENT = True
ENABLE_AXIAL_ALIGNMENT = True
ENABLE_QC_OUTPUTS = True
ENABLE_INTERMEDIATE_OUTPUTS = True

# Axial overlap metric: LCC, NCC, SSIM, NMI, or MIND.
AXIAL_ALIGNMENT_METRIC = "LCC"

RUNTIME_DEVICE = os.environ.get("FREETUNE4D_DEVICE", "cuda:0")
if RUNTIME_DEVICE not in {"cpu", "cuda:0"}:
    raise ValueError(f"Unsupported FREETUNE4D_DEVICE: {RUNTIME_DEVICE}")


def build_argument_parser():
    """Build the command-line interface used by the original script."""
    bases = (
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    )
    parser = argparse.ArgumentParser(
        formatter_class=type("formatter", bases, {}),
        description="FreeTune4D for UTSouthWestern",
    )
    parser.add_argument("--phase_num", type=int, default=5, help="phase number")
    parser.add_argument(
        "--base_path",
        type=str,
        default="/mnt/sda/Academics/Code/MyCode/UltraRecon-4D/DDEM.Liver/ReProductionDataset03Case0008/",
        help="base path of 3D/4D image",
    )
    parser.add_argument("--MR_number", type=str, default="raw", help="MRN")
    parser.add_argument("--st_date", type=str, default="StDate", help="StDate")
    return parser


def load_image(path, pattern="IM-*", dims=None, nTP=None):
    """
    读取 DICOM 序列，返回：
      - img: numpy 数组，shape 为 (n1, n2, n3[, nTP])
      - pos: numpy 数组，存储每帧的物理坐标，shape 为 (3, nSlices * nTP)
      - vs: 体素大小 [dx, dy, dz]
    """
    print(
        "\n[load_image] --------------------------------------------------", flush=True
    )
    print(f"[load_image] path      = {path}", flush=True)
    print(f"[load_image] pattern   = {pattern}", flush=True)
    print(f"[load_image] nTP input = {nTP}", flush=True)

    files = sorted(glob.glob(os.path.join(path, pattern + ".dcm")))
    if not files:
        raise FileNotFoundError(f"No DICOM files match {pattern} in {path}")

    print(f"[load_image] number of dicom files = {len(files)}", flush=True)
    print(f"[load_image] first file = {files[0]}", flush=True)
    print(f"[load_image] last  file = {files[-1]}", flush=True)

    ds0 = pydicom.dcmread(files[0])
    dx, dy = [float(x) for x in ds0.PixelSpacing]
    dz = float(getattr(ds0, "SliceThickness", 1.0))
    vs = np.array([dx, dy, dz], dtype=float)

    print(f"[load_image] voxel spacing = {vs}", flush=True)
    print(
        f"[load_image] rows, cols    = ({int(ds0.Rows)}, {int(ds0.Columns)})",
        flush=True,
    )

    if nTP is not None and nTP > 1:
        N = len(files)
        n1, n2 = int(ds0.Rows), int(ds0.Columns)
        n3 = N // nTP
        img = np.zeros((n1, n2, n3, nTP), dtype=ds0.pixel_array.dtype)
        pos = np.zeros((3, N), dtype=float)

        print(
            f"[load_image] detected 4D series -> shape will be ({n1}, {n2}, {n3}, {nTP})",
            flush=True,
        )

        for idx, f in enumerate(files):
            ds = pydicom.dcmread(f)
            sl = idx // nTP
            tp = idx % nTP
            img[:, :, sl, tp] = ds.pixel_array
            pos[:, idx] = ds.ImagePositionPatient

            if idx == 0 or idx == len(files) - 1:
                print(
                    f"[load_image] reading idx={idx}, slice={sl}, tp={tp}", flush=True
                )
    else:
        N = len(files)
        n1, n2 = int(ds0.Rows), int(ds0.Columns)
        n3 = N
        img = np.zeros((n1, n2, n3), dtype=ds0.pixel_array.dtype)
        pos = np.zeros((3, n3), dtype=float)

        print(
            f"[load_image] detected 3D series -> shape will be ({n1}, {n2}, {n3})",
            flush=True,
        )

        for sl, f in enumerate(files):
            ds = pydicom.dcmread(f)
            img[:, :, sl] = ds.pixel_array
            pos[:, sl] = ds.ImagePositionPatient

            if sl == 0 or sl == len(files) - 1:
                print(f"[load_image] reading slice={sl}", flush=True)

    if dims is not None and img.shape[:3] != tuple(dims):
        raise ValueError(f"Expected image shape {dims}, got {img.shape[:3]}")

    print(f"[load_image] final image shape = {img.shape}", flush=True)
    print(f"[load_image] final pos shape   = {pos.shape}", flush=True)
    print("[load_image] done.", flush=True)

    return img, pos, vs


def GIFplot2(volume, filepath, duration, intensity_range):
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
    imageio.mimsave(filepath, frames, format="GIF", duration=duration)


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
    print(
        f"[vis] save_plot_3d -> {name}, shape={img.shape}, display_range=({lo:.3f}, {hi:.3f})",
        flush=True,
    )

    peilin.plot_3DLiver(
        img_norm.copy(),
        name=name,
        titles=["Axial Plane", "Coronal Plane", "Sagittal Plane"],
        path=path,
        max=1,
        min=0,
        dpi=dpi,
    )


def save_abs_diff_plot(ref_img, mov_img, name, path, dpi=300):
    """
    保存两个 3D 图像的绝对差值图。
    """
    ref_norm, _, _ = robust_normalize(ref_img)
    mov_norm, _, _ = robust_normalize(mov_img)

    diff = np.abs(ref_norm - mov_norm)
    print(
        f"[vis] save_abs_diff_plot -> {name}, shape={diff.shape}, diff_range=({diff.min():.6f}, {diff.max():.6f})",
        flush=True,
    )

    peilin.plot_3DLiver(
        diff.copy(),
        name=name,
        titles=["Axial Diff", "Coronal Diff", "Sagittal Diff"],
        path=path,
        max=1,
        min=0,
        dpi=dpi,
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


def plot_metric_curve(scores, path, metric, name="axial_metric_curve", dpi=200):
    """Save the selected axial metric over all candidate offsets."""
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(8, 4), dpi=dpi)
    plt.plot(np.arange(len(scores)), scores, marker="o", linewidth=1)
    plt.xlabel("Axial Candidate Index")
    plt.ylabel(f"{metric} value")
    plt.title(f"Axial Sliding {metric} Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    save_path = os.path.join(path, f"{name}.jpg")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[vis] saved LCC curve -> {save_path}", flush=True)


def center_crop_or_pad_2d_to_shape(vol, target_x, target_y):
    print(
        "\n[center_crop_or_pad_2d_to_shape] ------------------------------", flush=True
    )
    print(f"[center_crop_or_pad_2d_to_shape] input shape  = {vol.shape}", flush=True)
    print(
        f"[center_crop_or_pad_2d_to_shape] target shape = ({target_x}, {target_y}, {vol.shape[2]})",
        flush=True,
    )

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

    print(
        f"[center_crop_or_pad_2d_to_shape] source x: [{xs0}:{xs1}], dest x: [{xd0}:{xd1}]",
        flush=True,
    )
    print(
        f"[center_crop_or_pad_2d_to_shape] source y: [{ys0}:{ys1}], dest y: [{yd0}:{yd1}]",
        flush=True,
    )

    out[xd0:xd1, yd0:yd1, :] = vol[xs0:xs1, ys0:ys1, :]

    print(f"[center_crop_or_pad_2d_to_shape] output shape = {out.shape}", flush=True)
    return out


def alignment_metric_direction(metric):
    """Return whether a supported axial alignment metric is maximized."""
    metric = metric.upper()
    if metric not in {"LCC", "NCC", "SSIM", "NMI", "MIND"}:
        raise ValueError(f"Unsupported axial alignment metric: {metric}")
    return metric != "MIND"


def compute_alignment_metric(
    vol1, vol2, metric=AXIAL_ALIGNMENT_METRIC, device=RUNTIME_DEVICE, eps=1e-8
):
    """Compare equal-grid overlap volumes using the selected existing metric."""
    assert vol1.shape == vol2.shape, f"Shape mismatch: {vol1.shape} vs {vol2.shape}"
    metric = metric.upper()
    v1 = torch.tensor(vol1.astype(np.float32))[None, None, ...]
    v2 = torch.tensor(vol2.astype(np.float32))[None, None, ...]

    if metric == "MIND":
        return float(peilin_loss.MINDSSC(device=device).loss(v1, v2).cpu())

    v1 = (v1 - v1.min()) / (v1.max() - v1.min() + eps)
    v2 = (v2 - v2.min()) / (v2.max() - v2.min() + eps)
    if metric == "LCC":
        value = -peilin_loss.NCC(device=device).loss(v1.to(device), v2.to(device))
    elif metric == "NCC":
        first = v1.to(device)
        second = v2.to(device)
        first = first - first.mean()
        second = second - second.mean()
        value = (first * second).mean() / (
            first.square().mean().sqrt() * second.square().mean().sqrt() + eps
        )
    elif metric == "SSIM":
        value = peilin_loss.SSIM().loss(v1.to(device), v2.to(device))
    elif metric == "NMI":
        value = peilin_loss.NMI().loss(v1, v2)
    else:
        alignment_metric_direction(metric)
    return (
        float(np.mean(value.detach().cpu().numpy()))
        if torch.is_tensor(value)
        else float(value)
    )


def score_overlap_only(
    fd_ref,
    t2_xy_aligned,
    t2_offset,
    metric=AXIAL_ALIGNMENT_METRIC,
    device=RUNTIME_DEVICE,
    min_overlap_ratio=0.5,
):
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
        an invalid sentinel, None, None, None, None, overlap_len, min_required
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
        invalid = -np.inf if alignment_metric_direction(metric) else np.inf
        return invalid, None, None, None, None, overlap_len, min_required

    # 映射回各自局部坐标
    fd_z0 = overlap_start - fd_global_start
    fd_z1 = overlap_end - fd_global_start

    t2_z0 = overlap_start - t2_global_start
    t2_z1 = overlap_end - t2_global_start

    fd_part = fd_ref[:, :, fd_z0:fd_z1]
    t2_part = t2_xy_aligned[:, :, t2_z0:t2_z1]

    if fd_part.shape != t2_part.shape:
        invalid = -np.inf if alignment_metric_direction(metric) else np.inf
        return invalid, None, None, None, None, overlap_len, min_required

    score = compute_alignment_metric(fd_part, t2_part, metric=metric, device=device)
    return score, fd_z0, fd_z1, t2_z0, t2_z1, overlap_len, min_required


def axial_match(
    fd_crop,
    t2_xy_aligned,
    metric=AXIAL_ALIGNMENT_METRIC,
    device=RUNTIME_DEVICE,
    vis_path=None,
    prefix="axial_match",
):
    print(
        "\n[axial_match_by_lcc] ==========================================", flush=True
    )
    print(f"[axial_match_by_lcc] fd_crop shape       = {fd_crop.shape}", flush=True)
    print(
        f"[axial_match_by_lcc] t2_xy_aligned shape = {t2_xy_aligned.shape}", flush=True
    )

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

    higher_is_better = alignment_metric_direction(metric)
    best_score = -np.inf if higher_is_better else np.inf
    best_offset = None
    best_fd_z0, best_fd_z1 = None, None
    best_t2_z0, best_t2_z1 = None, None

    all_scores = []

    # 合法 offset 范围：
    # 至少要保证 overlap_len >= min_required
    offset_min = -(zt - min_required)
    offset_max = zf - min_required

    total_candidates = offset_max - offset_min + 1
    print(
        f"[axial_match_by_lcc] offset range = [{offset_min}, {offset_max}]", flush=True
    )
    print(f"[axial_match_by_lcc] total candidates = {total_candidates}", flush=True)

    # baseline：让 T2 和 4D 在 axial 上尽量居中对齐
    center_offset = int(np.clip((zf - zt) // 2, offset_min, offset_max))

    baseline_score, fd_z0, fd_z1, t2_z0, t2_z1, overlap_len, _ = score_overlap_only(
        fd_ref,
        t2_xy_aligned,
        center_offset,
        metric=metric,
        device=device,
        min_overlap_ratio=0.5,
    )

    if fd_z0 is not None and vis_path is not None:
        baseline_ref = fd_ref[:, :, fd_z0:fd_z1]
        baseline_mov = t2_xy_aligned[:, :, t2_z0:t2_z1]

        print(
            f"[axial_match_by_lcc] baseline center offset = {center_offset}, "
            f"overlap fd[{fd_z0}:{fd_z1}] vs t2[{t2_z0}:{t2_z1}], "
            f"overlap_len = {overlap_len}, score = {baseline_score:.6f}",
            flush=True,
        )

        save_match_triplet(
            baseline_ref,
            baseline_mov,
            f"{prefix}_02_before_match_centerCandidate",
            vis_path,
            score=baseline_score,
        )

    for offset in range(offset_min, offset_max + 1):
        score, fd_z0, fd_z1, t2_z0, t2_z1, overlap_len, min_required = (
            score_overlap_only(
                fd_ref,
                t2_xy_aligned,
                offset,
                metric=metric,
                device=device,
                min_overlap_ratio=0.5,
            )
        )

        all_scores.append(score)

        if fd_z0 is None:
            print(
                f"[axial_match_by_lcc] offset = {offset}, "
                f"overlap_len = {overlap_len} < min_required = {min_required}, skipped.",
                flush=True,
            )
            continue

        print(
            f"[axial_match_by_lcc] offset = {offset}, "
            f"overlap fd[{fd_z0}:{fd_z1}] vs t2[{t2_z0}:{t2_z1}], "
            f"overlap_len = {overlap_len}, score = {score:.6f}",
            flush=True,
        )

        if (higher_is_better and score > best_score) or (
            not higher_is_better and score < best_score
        ):
            best_score = score
            best_offset = offset
            best_fd_z0, best_fd_z1 = fd_z0, fd_z1
            best_t2_z0, best_t2_z1 = t2_z0, t2_z1

            print(
                f"[axial_match_by_lcc] --> new best: score={best_score:.6f}, "
                f"offset={best_offset}, "
                f"fd[{best_fd_z0}:{best_fd_z1}] vs t2[{best_t2_z0}:{best_t2_z1}]",
                flush=True,
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
            score=best_score,
        )
        plot_metric_curve(
            all_scores, vis_path, metric, name=f"{prefix}_04_{metric.lower()}_curve"
        )

    print(f"[axial_match_by_lcc] final best_score    = {best_score:.6f}", flush=True)
    print(f"[axial_match_by_lcc] best_offset         = {best_offset}", flush=True)
    print(
        f"[axial_match_by_lcc] best overlap fd     = [{best_fd_z0}:{best_fd_z1}]",
        flush=True,
    )
    print(
        f"[axial_match_by_lcc] best overlap t2     = [{best_t2_z0}:{best_t2_z1}]",
        flush=True,
    )
    print(
        f"[axial_match_by_lcc] final overlap len   = {best_fd_z1 - best_fd_z0}",
        flush=True,
    )
    print(f"[axial_match_by_lcc] fd_final shape      = {fd_final.shape}", flush=True)
    print(f"[axial_match_by_lcc] t2_final shape      = {t2_final.shape}", flush=True)

    # 为了兼容你原先主程序的接收变量名，这里仍然返回 5 个值
    # best_fd_start / best_t2_start 现在表示“最终裁剪区域在各自 volume 中的起始 z”
    best_fd_start = best_fd_z0
    best_t2_start = best_t2_z0

    return fd_final, t2_final, best_score, best_fd_start, best_t2_start


def clustering(imgs_tmp, class_num=3, path="./tmp"):
    print(
        "\n[clustering] ==================================================", flush=True
    )
    print(f"[clustering] input shape = {imgs_tmp.shape}", flush=True)
    print(f"[clustering] class_num   = {class_num}", flush=True)
    print(f"[clustering] save path   = {path}", flush=True)

    imgs_tmp = torch.tensor(imgs_tmp.astype(np.float32))
    imgs = (imgs_tmp - torch.min(imgs_tmp)) / (
        torch.max(imgs_tmp) - torch.min(imgs_tmp) + 1e-8
    )

    if ENABLE_INTERMEDIATE_OUTPUTS:
        os.makedirs(path, exist_ok=True)
    batch_size = 20
    gl_device = RUNTIME_DEVICE

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

        print(
            f"[clustering] computing batch {i + 1}/{total_batches}, pair index [{st}:{ed})",
            flush=True,
        )

        imgs1 = imgs[..., index1[st:ed]].permute(3, 0, 1, 2).float()
        imgs2 = imgs[..., index2[st:ed]].permute(3, 0, 1, 2).float()

        LCC = (
            peilin_loss.NCC(device=gl_device)
            .loss(imgs1[:, None, ...].to(gl_device), imgs2[:, None, ...].to(gl_device))
            .to("cpu")
            .numpy()
        )

        metrics[st:ed] = -LCC

        print(
            f"[clustering] batch {i + 1} finished, metric range = ({metrics[st:ed].min():.6f}, {metrics[st:ed].max():.6f})",
            flush=True,
        )

    print("[clustering] filling symmetric matrix...", flush=True)
    for i in range(image_num):
        for j in range(i, image_num):
            index = index_for_index.index((i, j))
            matrix[i, j] = float(metrics[index])
            matrix[j, i] = float(metrics[index])

    if ENABLE_INTERMEDIATE_OUTPUTS:
        np.savez(os.path.join(path, "matrix.npz"), matrix=matrix, metrics=metrics)

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
        for j in range(i + 1, class_num):
            lcc = 0
            for index_i in clusters[i]:
                for index_j in clusters[j]:
                    lcc += matrix[index_i, index_j]
            lcc /= len(clusters[i]) * len(clusters[j])
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
            assert lcc_index not in sorted_clusters, (
                "{} clustering... lcc_index has been included...".format(time.ctime())
            )

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

        print(
            f"[clustering] output phase {i}: from cluster {cluster_id}, representative frame = {rep_idx}",
            flush=True,
        )

    sorted_imgs = torch.concat(sorted_imgs, dim=-1)

    if ENABLE_INTERMEDIATE_OUTPUTS:
        np.savez(
            os.path.join(path, "collected_4d.npz"),
            sorted=sorted_imgs.numpy(),
            clusters=clusters,
        )
    print(f"[clustering] final sorted shape = {sorted_imgs.shape}", flush=True)
    print(f"[clustering] representative indices = {selected_rep_indices}", flush=True)
    print("[clustering] done.", flush=True)

    return clusters, sorted_imgs.numpy()


def load_dynamic_case(patient_path, phase_num):
    """Load and validate the original dynamic DICOM series."""
    thr_dirs = glob.glob(os.path.join(patient_path, "*THRIVE*"))
    if not thr_dirs:
        raise FileNotFoundError("No THRIVE folder found")
    four_d_path = thr_dirs[0]
    dcm_files = sorted(glob.glob(os.path.join(four_d_path, "*.dcm")))
    if not dcm_files:
        raise FileNotFoundError(
            f"No .dcm files found in dynamic DICOM directory: {four_d_path}"
        )
    info = pydicom.dcmread(dcm_files[0])
    n_tp = int(info.NumberOfTemporalPositions)
    if n_tp <= 0 or len(dcm_files) % n_tp:
        raise ValueError(
            f"Invalid dynamic DICOM ordering: {len(dcm_files)} files and {n_tp} temporal positions."
        )
    four_d, positions, spacing = load_image(four_d_path, pattern="IM-*", nTP=n_tp)
    if four_d.ndim != 4 or any(size <= 0 for size in four_d.shape):
        raise ValueError(
            f"Dynamic DICOM must produce a non-empty 4D volume; got shape {four_d.shape}."
        )
    if four_d.shape[-1] != n_tp or not np.isfinite(four_d).all():
        raise ValueError("Dynamic DICOM temporal dimensions or values are invalid.")
    if ENABLE_SORTING and four_d.shape[-1] < phase_num:
        raise ValueError(
            f"Cannot create {phase_num} phases from only {four_d.shape[-1]} temporal frames."
        )
    if RUNTIME_DEVICE == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device cuda:0 was requested but is not available.")
    return four_d, positions, spacing, n_tp


def run_sorting_stage(four_d, phase_num):
    """Run the existing NCC/KMeans phase sorting, or pass every original frame."""
    if not ENABLE_SORTING:
        return four_d, None
    clusters, sorted_images = clustering(
        four_d.copy(), class_num=phase_num, path="./clustering"
    )
    return sorted_images, clusters


def load_static_case(patient_path):
    """Load and validate the static T2 DICOM series."""
    t2_dirs = glob.glob(os.path.join(patient_path, "*T2_AX_MVXD*"))
    if not t2_dirs:
        raise FileNotFoundError("No T2_AX_MVXD folder found")
    t2_path = t2_dirs[0]
    t2, positions, spacing = load_image(t2_path, pattern="IM-*")
    if t2.ndim != 3 or any(size <= 0 for size in t2.shape) or not np.isfinite(t2).all():
        raise ValueError(
            f"Static DICOM must produce a finite, non-empty 3D volume; got {t2.shape}."
        )
    return t2, positions, spacing, t2_path


def save_demo_gifs(four_d, sorted_four_d, spacing, coronal_index, patient_path):
    """Save the original pipeline's dynamic QC GIFs when QC is enabled."""
    if not ENABLE_QC_OUTPUTS:
        return
    for image, filename in (
        (four_d, "4D_AX_T2.gif"),
        (sorted_four_d, "4D_AX_ave_T2.gif"),
    ):
        gif = np.flip(np.rot90(image[coronal_index, :, :, :], k=1, axes=(0, 1)), axis=0)
        resized = zoom(gif, (spacing[2], spacing[1], 1), order=1)
        GIFplot2(
            resized,
            os.path.join(patient_path, filename),
            duration=0.3,
            intensity_range=(0, 0.8 * resized.max()),
        )


def align_preprocessed_volumes(
    four_d, four_d_pos, four_d_spacing, n_tp, t2, t2_pos, t2_spacing, vis_root
):
    """Run the original common-FOV, XY, and axial preprocessing alignment."""
    n1 = four_d.shape[0]
    coords = four_d_pos[:, np.arange(0, four_d_pos.shape[1], n_tp)]
    four_d_min = np.array([coords[0].min(), coords[1].min(), coords[2].min()])
    four_d_max = np.array(
        [
            coords[0].min() + four_d_spacing[0] * n1,
            coords[1].min() + four_d_spacing[1] * four_d.shape[1],
            coords[2].max(),
        ]
    )
    t2_min = np.array([t2_pos[0].min(), t2_pos[1].min(), t2_pos[2].min()])
    t2_max = np.array(
        [
            t2_pos[0].min() + t2_spacing[0] * t2.shape[0],
            t2_pos[1].min() + t2_spacing[1] * t2.shape[1],
            t2_pos[2].max(),
        ]
    )
    common_min = np.maximum.reduce([four_d_min, t2_min])
    common_max = np.minimum.reduce([four_d_max, t2_max])

    four_d_iso = np.stack(
        [
            zoom(four_d[..., index], four_d_spacing, order=1)
            for index in range(four_d.shape[3])
        ],
        axis=3,
    )
    start_fd = np.round((common_min - four_d_min) / four_d_spacing).astype(int)
    end_fd = np.round((four_d_max - common_max) / four_d_spacing).astype(int)
    fd_crop = four_d_iso[
        start_fd[0] : four_d_iso.shape[0] - end_fd[0],
        start_fd[1] : four_d_iso.shape[1] - end_fd[1],
        start_fd[2] : four_d_iso.shape[2] - end_fd[2],
        :,
    ]

    t2_iso = zoom(t2, t2_spacing, order=1)
    start_t2 = np.round((common_min - t2_min) / t2_spacing).astype(int)
    end_t2 = np.round((t2_max - common_max) / t2_spacing).astype(int)
    t2_crop = t2_iso[
        start_t2[0] : t2_iso.shape[0] - end_t2[0],
        start_t2[1] : t2_iso.shape[1] - end_t2[1],
        start_t2[2] : t2_iso.shape[2] - end_t2[2],
    ]
    if ENABLE_QC_OUTPUTS:
        save_plot_3d(fd_crop.mean(axis=3), "01_FourDRef_after_common_crop", vis_root)
        save_plot_3d(t2_crop, "02_T2_after_common_crop", vis_root)

    if ENABLE_XY_ALIGNMENT:
        t2_xy = center_crop_or_pad_2d_to_shape(
            t2_crop, fd_crop.shape[0], fd_crop.shape[1]
        )
    else:
        if t2_crop.shape[:2] != fd_crop.shape[:2]:
            raise ValueError(
                "XY alignment is disabled, but FourD and T2 do not have matching X/Y dimensions: "
                f"{fd_crop.shape[:2]} vs {t2_crop.shape[:2]}."
            )
        t2_xy = t2_crop
    if ENABLE_QC_OUTPUTS:
        save_plot_3d(t2_xy, "03_T2_after_xy_align", vis_root)

    if ENABLE_AXIAL_ALIGNMENT:
        fd_aligned, t2_aligned, value, fd_start, t2_start = axial_match(
            fd_crop,
            t2_xy,
            metric=AXIAL_ALIGNMENT_METRIC,
            device=RUNTIME_DEVICE,
            vis_path=os.path.join(vis_root, "axial_match")
            if ENABLE_QC_OUTPUTS
            else None,
        )
    else:
        zf, zt = fd_crop.shape[2], t2_xy.shape[2]
        overlap = min(zf, zt)
        fd_start = (zf - overlap) // 2
        t2_start = (zt - overlap) // 2
        fd_aligned = fd_crop[:, :, fd_start : fd_start + overlap, :]
        t2_aligned = t2_xy[:, :, t2_start : t2_start + overlap]
        value = np.nan

    if ENABLE_QC_OUTPUTS:
        save_plot_3d(fd_aligned.mean(axis=3), "04_FourD_after_axial_match", vis_root)
        save_plot_3d(t2_aligned, "05_T2_after_axial_match", vis_root)
        save_abs_diff_plot(
            fd_aligned.mean(axis=3),
            t2_aligned,
            "06_AbsDiff_after_axial_match",
            vis_root,
        )
    return fd_aligned, t2_aligned, value, fd_start, t2_start


def save_preprocessing_results(
    patient_path, fd_aligned, t2_aligned, metric_value, fd_start, t2_start
):
    """Apply the original final margin and save the phase MAT file."""
    margin = 1
    if min(*fd_aligned.shape[:3], *t2_aligned.shape) <= 2 * margin:
        raise ValueError("margin too large after alignment/cropping.")
    t2_save = t2_aligned[margin:-margin, margin:-margin, margin:-margin]
    four_d_save = fd_aligned[margin:-margin, margin:-margin, margin:-margin, :].astype(
        np.uint16
    )
    io.savemat(
        os.path.join(patient_path, "phase_T2.mat"),
        {
            "T2_save": t2_save,
            "FourD_ave_save": four_d_save,
            "best_lcc": metric_value,
            "best_alignment_metric": metric_value,
            "alignment_metric_name": AXIAL_ALIGNMENT_METRIC,
            "alignment_metric_higher_is_better": alignment_metric_direction(
                AXIAL_ALIGNMENT_METRIC
            ),
            "best_fd_start": fd_start,
            "best_t2_start": t2_start,
        },
        do_compression=True,
    )


def main(args=None):
    """Run the original preprocessing pipeline through explicit stages."""
    arg = build_argument_parser().parse_args(args)
    alignment_metric_direction(AXIAL_ALIGNMENT_METRIC)
    patient_path = os.path.join(arg.base_path, arg.MR_number, arg.st_date)
    os.chdir(arg.base_path)
    os.chdir(patient_path)
    vis_root = os.path.join(patient_path, "debug_vis")
    if ENABLE_QC_OUTPUTS:
        os.makedirs(vis_root, exist_ok=True)

    four_d, four_d_pos, four_d_spacing, n_tp = load_dynamic_case(
        patient_path, arg.phase_num
    )
    if ENABLE_INTERMEDIATE_OUTPUTS:
        io.savemat(
            os.path.join(patient_path, "Full_data.mat"),
            {"FourD": four_d.copy()},
            do_compression=True,
        )
    sorted_four_d, clusters = run_sorting_stage(four_d, arg.phase_num)
    if clusters is not None:
        for cluster_id, members in clusters.items():
            print(
                f"cluster {cluster_id}: size={len(members)}, members={members}",
                flush=True,
            )

    t2, t2_pos, t2_spacing, t2_path = load_static_case(patient_path)
    coronal_index = int(four_d.shape[0] / 2)
    save_demo_gifs(four_d, sorted_four_d, four_d_spacing, coronal_index, patient_path)
    aligned = align_preprocessed_volumes(
        sorted_four_d,
        four_d_pos,
        four_d_spacing,
        n_tp,
        t2,
        t2_pos,
        t2_spacing,
        vis_root,
    )
    save_preprocessing_results(patient_path, *aligned)

    if ENABLE_QC_OUTPUTS:
        for index, number in enumerate(plt.get_fignums(), start=1):
            plt.figure(number).savefig(
                os.path.join(patient_path, f"T2_figure_{index}.png")
            )
    os.makedirs(os.path.join(patient_path, "UQ_4D_T2"), exist_ok=True)
    file_names = os.listdir(t2_path)
    if len(file_names) >= 3:
        print("Third file in T2 directory:", file_names[2], flush=True)


if __name__ == "__main__":
    main()
