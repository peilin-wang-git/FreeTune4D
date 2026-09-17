"""FreeTune4D T2 motion reconstruction pipeline.

The stage functions are structural wrappers around the original registration,
DVF, inference, and DICOM export operations.
"""

import argparse
import glob
import importlib
import os
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import pydicom
import scipy.interpolate as interpolate
import scipy.io as sio
import SimpleITK as sitk
import tensorflow as tf
import torch
import torch.nn.functional as F
import voxelmorph as vxm

import peilin

# Preserve the original local PyTorch SpatialTransformer implementation.
MODEL_SOURCE_ROOT = (
    r"/mnt/sda/Academics/Code/MyCode/UltraRecon-4D/DDEM.Liver/uq4d_scripts"
)
sys.path.append(os.path.join(MODEL_SOURCE_ROOT, "voxelmorph-master", "pytorch"))
SpatialTransformer = importlib.import_module("model").SpatialTransformer

# Pipeline switches. Defaults reproduce the original full pipeline.
ENABLE_AFFINE_ALIGNMENT = True  # Existing Elastix Rigid stage.
ENABLE_BSPLINE_ALIGNMENT = True
ENABLE_COARSE_ALIGNMENT = True
ENABLE_DVF_SMOOTHING = True
ENABLE_FINE_ALIGNMENT = True
ENABLE_QC_OUTPUTS = True
ENABLE_INTERMEDIATE_OUTPUTS = True
ENABLE_UQ_DICOM_EXPORT = True
ENABLE_LQ_DICOM_EXPORT = True

COARSE_VOLUME_SIZE = [128, 128, 128]
FINE_VOLUME_SIZE = [224, 224, 224]
AMPLIFIER = 1.0


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
    parser.add_argument(
        "--base_path",
        type=str,
        default="/mnt/sda/Academics/Code/MyCode/UltraRecon-4D/26042101Foll25092901.Liver",
    )
    parser.add_argument("--MR_number", type=str, default="92441064", help="MRN")
    parser.add_argument("--st_date", type=str, default="20260410", help="StDate")
    parser.add_argument(
        "--net_path_coarse",
        type=str,
        default="/mnt/sda/Academics/Code/MyCode/UltraRecon-4D/26042101Foll25092901.Liver/coarse.h5",
    )
    parser.add_argument(
        "--net_path_fine",
        type=str,
        default="/mnt/sda/Academics/Code/MyCode/UltraRecon-4D/26042101Foll25092901.Liver/fine.h5",
    )
    parser.add_argument("--name_3d", type=str, default="T2_AX_MVXD")
    parser.add_argument("--reference_file", type=str, default="IM-301-0001.dcm")
    return parser


def dvf_interp(dvf, volume_size):
    """Interpolate the existing three-channel DVF and preserve displacement scale."""
    original_shape = dvf.shape[2:]
    components = [
        F.interpolate(
            torch.unsqueeze(dvf[:, index, :, :, :], 1), volume_size, mode="trilinear"
        )
        * (volume_size[index] / original_shape[index])
        for index in range(3)
    ]
    return torch.cat(components, 1)


def img_norm(image):
    """Apply the original image scaling operation without changing its formula."""
    return image * np.array(1 / (np.max(image) - np.min(image)))


def compute_cc(image_1, image_2):
    """Compute the original global cross-correlation reference-frame score."""
    mean_1, mean_2 = np.mean(image_1[:]), np.mean(image_2[:])
    return np.mean((image_1 - mean_1) * (image_2 - mean_2)) / (
        np.std(image_1[:]) * np.std(image_2[:])
    )


def smooth_dvf(dvf, phase_count):
    """Apply the original phase-axis B-spline smoothing to every DVF component."""
    dimensions, width, length, height = dvf.shape[1:]
    phase_axis = np.arange(phase_count)
    dvf_numpy = dvf.detach().cpu().numpy()
    smoothed = np.zeros_like(dvf_numpy)
    for i in range(width):
        for j in range(length):
            for h in range(height):
                for dimension in range(dimensions):
                    knots, coefficients, degree = interpolate.splrep(
                        phase_axis, dvf_numpy[:, dimension, i, j, h], s=0.1, k=3
                    )
                    spline = interpolate.BSpline(
                        knots, coefficients, degree, extrapolate=False
                    )
                    smoothed[:, dimension, i, j, h] = spline(phase_axis)
    return smoothed


def run_elastix(moving_image, fixed_image, path, mode="BSpline"):
    """Run the existing Elastix command and restore the moving intensity range."""
    moving_range = {"max": np.max(moving_image), "min": np.min(moving_image)}
    fixed_range = {"max": np.max(fixed_image), "min": np.min(fixed_image)}
    moving_normalized = (moving_image - moving_range["min"]) / (
        moving_range["max"] - moving_range["min"]
    )
    fixed_normalized = (fixed_image - fixed_range["min"]) / (
        fixed_range["max"] - fixed_range["min"]
    )
    os.makedirs(path, exist_ok=True)
    moving_path = os.path.join(path, "moving_image.nii.gz")
    fixed_path = os.path.join(path, "fixed_image.nii.gz")
    sitk.WriteImage(sitk.GetImageFromArray(moving_normalized * 255), moving_path)
    sitk.WriteImage(sitk.GetImageFromArray(fixed_normalized * 255), fixed_path)
    parameter_files = {
        "BSpline": "./Par0020bspline2-MI-lesswarp.txt",
        "Affine": "./parameters_Affine.txt",
        "Rigid": "./parameters_Rigid.txt",
    }
    if mode not in parameter_files:
        raise ValueError(f"Unsupported Elastix mode: {mode}")
    command = f"elastix -f {fixed_path} -m {moving_path} -p {parameter_files[mode]} -out {path}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Elastix {mode} registration failed for output path {path}."
        ) from exc
    warped = torch.tensor(
        np.squeeze(nib.load(os.path.join(path, "result.0.nii.gz")).dataobj)
        .astype(float)
        .transpose((2, 1, 0))
    )
    warped[warped < 0] = 0
    warped = (warped - torch.min(warped)) / (torch.max(warped) - torch.min(warped))
    return (
        warped * (moving_range["max"] - moving_range["min"]) + moving_range["min"]
    ).numpy()


def load_models(coarse_path, fine_path, device):
    """Build and load only the existing models required by enabled stages."""
    device_vxm = "/CPU:0" if device == "cpu" else vxm.tf.utils.setup_device("0")[0]
    coarse_model = fine_model = None
    with tf.device(device_vxm):
        if ENABLE_COARSE_ALIGNMENT:
            coarse_model = vxm.networks.HyperVxmDense(
                int_steps=5,
                reg_field="warp",
                inshape=COARSE_VOLUME_SIZE,
                int_resolution=2,
                svf_resolution=2,
                nb_unet_features=([64] * 4, [64] * 6),
            )
        if ENABLE_FINE_ALIGNMENT:
            fine_model = vxm.networks.HyperVxmDense(
                int_steps=5,
                reg_field="warp",
                inshape=FINE_VOLUME_SIZE,
                int_resolution=2,
                svf_resolution=2,
                nb_unet_features=([24] * 4, [24] * 6),
            )
        if ENABLE_COARSE_ALIGNMENT:
            coarse_model = peilin.inspect_weight_loading(coarse_model, coarse_path)
        if ENABLE_FINE_ALIGNMENT:
            fine_model = peilin.inspect_weight_loading(fine_model, fine_path)
    return coarse_model, fine_model


def load_inputs(base_path, mrn, study_date, image_name, reference_file):
    """Load the preprocessed MAT volumes and original T2 DICOM metadata."""
    patient_path = os.path.join(base_path, mrn, study_date)
    t2_path = os.path.join(patient_path, image_name)
    mat_data = sio.loadmat(os.path.join(patient_path, "phase_T2.mat"))
    four_d, t2 = mat_data["FourD_ave_save"], mat_data["T2_save"]
    header = pydicom.dcmread(os.path.join(t2_path, reference_file))
    dicom_paths = glob.glob(os.path.join(t2_path, "IM-*"))
    t2_files = [pydicom.dcmread(path) for path in dicom_paths]
    positions = np.asarray([item.ImagePositionPatient for item in t2_files])
    return (
        four_d,
        t2,
        header,
        t2_files,
        positions,
        os.path.join(patient_path, "UQ_4D_T2"),
    )


def select_reference_frame(four_d, t2):
    """Select the FourD phase with the original global CC rule."""
    scores = np.asarray(
        [compute_cc(four_d[..., frame], t2) for frame in range(four_d.shape[3])]
    )
    index = int(np.argmax(scores))
    print(f"Matched Frame is {index}")
    return four_d[..., index]


def run_initial_alignment(t2, reference_frame):
    """Run independently switchable existing Rigid and BSpline Elastix stages."""
    current = t2
    if ENABLE_AFFINE_ALIGNMENT:
        current = run_elastix(current, reference_frame, "temp_folder_1", "Rigid")
        if ENABLE_QC_OUTPUTS:
            peilin.plot_3DLiver(
                current, name="T2_affine_warped", path="./tmp_plot", min=0, max=1
            )
    if ENABLE_BSPLINE_ALIGNMENT:
        current = run_elastix(current, reference_frame, "temp_folder_2", "BSpline")
    return current


def prepare_registration(t2_aligned, reference_frame, device):
    """Prepare tensors and the original-grid spatial transformer."""
    t2_normalized = img_norm(t2_aligned)
    original_size = t2_normalized.shape
    transformer = SpatialTransformer(original_size).to(device)
    t2_tensor = (
        torch.from_numpy(
            np.asarray(t2_normalized[np.newaxis, np.newaxis, ...], dtype=np.float32)
        )
        .to(device)
        .float()
    )
    reference_normalized = img_norm(reference_frame)
    reference_tensor = (
        torch.from_numpy(
            np.asarray(
                reference_normalized[np.newaxis, np.newaxis, ...], dtype=np.float32
            )
        )
        .to(device)
        .float()
    )
    coarse_moving = F.interpolate(
        reference_tensor, size=COARSE_VOLUME_SIZE, mode="trilinear"
    )
    return t2_tensor, original_size, transformer, coarse_moving


def run_coarse_alignment(four_d, coarse_moving, model, original_size, device):
    """Estimate every phase's coarse DVF with the existing model and inference call."""
    collected = torch.zeros([four_d.shape[3], 3, *COARSE_VOLUME_SIZE]).to(device)
    for frame in range(four_d.shape[3]):
        fixed = (
            torch.from_numpy(
                np.asarray(
                    img_norm(four_d[..., frame])[np.newaxis, np.newaxis, ...],
                    dtype=np.float32,
                )
            )
            .to(device)
            .float()
        )
        input_fixed = F.interpolate(fixed, size=COARSE_VOLUME_SIZE, mode="trilinear")
        if ENABLE_QC_OUTPUTS:
            peilin.plot_3DLiver(
                coarse_moving.squeeze().cpu().numpy(),
                name=f"tmp{frame}",
                path="./tmp_plot",
                min=0,
                max=1,
            )
        with torch.no_grad():
            flow = peilin.inference_HyperMorph(
                coarse_moving.cpu(), input_fixed.cpu(), model, f"frame_{frame}"
            )
        if ENABLE_QC_OUTPUTS:
            for dimension in range(3):
                peilin.plot_3DLiver(
                    flow.squeeze()[dimension].detach().cpu().numpy(),
                    name=f"tmp_dvf_{dimension + 1}dim_{frame}",
                    path="./tmp_plot",
                    min=0,
                    max=1,
                )
        collected[frame] = flow * AMPLIFIER
    return collected


def warp_with_coarse_flow(t2_tensor, transformer, flow, original_size, device):
    """Warp T2 with one coarse flow on the original spatial grid."""
    return transformer(t2_tensor, dvf_interp(flow, original_size).to(device))


def run_fine_alignment(
    coarse_warped, fixed, model, transformer, original_size, device, frame
):
    """Apply the existing fine HyperMorph inference to the current phase."""
    input_fixed = F.interpolate(fixed, size=FINE_VOLUME_SIZE, mode="trilinear")
    input_moving = F.interpolate(coarse_warped, size=FINE_VOLUME_SIZE, mode="trilinear")
    flow = peilin.inference_HyperMorph(
        input_moving.cpu(), input_fixed.cpu(), model, f"frame_{frame}_2"
    )
    return transformer(coarse_warped, dvf_interp(flow, original_size).to(device))


def initialize_dicom_identifiers():
    """Generate the original random DICOM identifier components in the original order."""
    series_number_base = int(np.random.randint(low=1500, high=3000, size=1))
    np.random.seed(int(time.time()))
    random_seeds = np.random.choice(10000, size=10000, replace=False)
    np.random.randint(10, size=29)  # Preserve the original random-number call order.
    study_number = str(int(np.random.randint(low=1500, high=3000, size=1)))
    frame_suffix = "".join(str(int(value)) for value in np.random.randint(10, size=29))
    return series_number_base, random_seeds, study_number, frame_suffix


def export_dicom_volume(
    volume,
    frame,
    prefix,
    description,
    header,
    t2_files,
    positions,
    output_path,
    identifiers,
    start,
    slice_count,
    clamp_negative,
):
    """Write one volume with the original T2-derived DICOM metadata and scaling."""
    series_number_base, random_seeds, study_number, frame_suffix = identifiers
    np.random.seed(int(start) + slice_count)
    time.sleep(1)
    series_suffix = "".join(str(int(value)) for value in np.random.randint(10, size=29))
    series_number = pydicom.valuerep.IS(series_number_base + frame)
    for slice_index in range(min(volume.shape[2], len(t2_files))):
        dicom_header = t2_files[-slice_index]
        series_uid = pydicom.uid.UID(
            dicom_header.SeriesInstanceUID[:27] + series_suffix
        )
        np.random.seed(int(time.time() + random_seeds[slice_count]))
        instance_suffix = "".join(
            str(int(value)) for value in np.random.randint(10, size=29)
        )
        suffix_length = len(str(frame) + str(slice_index))
        sop_uid = pydicom.uid.UID(
            dicom_header.SOPInstanceUID[:27]
            + str(frame)
            + str(slice_index)
            + instance_suffix[:-suffix_length]
        )
        frame_uid = pydicom.uid.UID(
            dicom_header.FrameOfReferenceUID[:27] + frame_suffix
        )
        header.PatientName = header.PatientID
        if clamp_negative:
            volume[volume < 0] = 0
        header.PixelData = np.uint16(volume[:, :, -slice_index] * 500).tobytes()
        header.SOPInstanceUID = sop_uid
        header.FrameOfReferenceUID = frame_uid
        header.SeriesInstanceUID = series_uid
        header.SeriesNumber = series_number
        header.StudyID = study_number
        header.Rows, header.Columns = volume.shape[:2]
        header.SliceThickness = dicom_header.SliceThickness
        header.SpacingBetweenSlices = dicom_header.SliceThickness
        header.ImagePositionPatient = [
            positions[:, 0].min(),
            positions[:, 1].min(),
            positions[:, 2].max() - slice_index * dicom_header.SliceThickness,
        ]
        header.ImageOrientationPatient = dicom_header.ImageOrientationPatient
        header.PixelSpacing = dicom_header.PixelSpacing
        header.InstanceNumber = dicom_header.InstanceNumber
        header.SliceLocation = (
            positions[:, 2].max() - slice_index * dicom_header.SliceThickness
        )
        header.SeriesDescription = f"{description} frame {frame}"
        header.save_as(os.path.join(output_path, f"{prefix}{frame}_{slice_index}.dcm"))
        slice_count += 1
    return slice_count


def reconstruct_phases(
    four_d,
    t2_aligned,
    reference_frame,
    header,
    t2_files,
    positions,
    output_path,
    coarse_model,
    fine_model,
    device,
    start,
):
    """Run coarse/smoothing/fine stages and save UQ results for every phase."""
    os.makedirs(output_path, exist_ok=True)
    t2_tensor, original_size, transformer, coarse_moving = prepare_registration(
        t2_aligned, reference_frame, device
    )
    identifiers = initialize_dicom_identifiers()
    slice_count = 0
    if ENABLE_COARSE_ALIGNMENT:
        collected = run_coarse_alignment(
            four_d, coarse_moving, coarse_model, original_size, device
        )
        flows = (
            smooth_dvf(collected, four_d.shape[3])
            if ENABLE_DVF_SMOOTHING
            else collected.detach().cpu().numpy()
        )

    for frame in range(four_d.shape[3]):
        fixed = (
            torch.from_numpy(
                np.asarray(
                    img_norm(four_d[..., frame])[np.newaxis, np.newaxis, ...],
                    dtype=np.float32,
                )
            )
            .to(device)
            .float()
        )
        current = t2_tensor
        if ENABLE_COARSE_ALIGNMENT:
            flow = torch.unsqueeze(torch.from_numpy(flows[frame]), 0).to(device)
            current = warp_with_coarse_flow(
                t2_tensor, transformer, flow, original_size, device
            )
        if ENABLE_FINE_ALIGNMENT:
            current = run_fine_alignment(
                current, fixed, fine_model, transformer, original_size, device, frame
            )

        four_d_np = F.interpolate(fixed, original_size).cpu().numpy()[0, 0, :]
        uq_np = current[0, 0, :].detach().cpu().numpy()
        spacing = [
            header.PixelSpacing[0],
            header.PixelSpacing[1],
            header.SliceThickness,
        ]
        restored_size = [
            int(value) for value in np.int16(np.array(t2_aligned.shape) / spacing)
        ]
        uq_restored = (
            F.interpolate(current, restored_size)[0, 0, :].detach().cpu().numpy()
        )
        if ENABLE_INTERMEDIATE_OUTPUTS:
            sio.savemat(
                os.path.join(output_path, f"UQ_T2_{frame}.mat"),
                {f"UQ_T2_{frame}": uq_np, f"FourD_{frame}": four_d_np},
            )
        if ENABLE_QC_OUTPUTS:
            peilin.plot_3DLiver(
                t2_aligned,
                name=f"T2_def_{frame}",
                path="./tmp_plot",
                min=0,
                max=four_d_np.max(),
            )
            peilin.plot_3DLiver(
                t2_tensor.detach().cpu().numpy()[0, 0],
                name=f"T2_crop_{frame}",
                path="./tmp_plot",
                min=0,
                max=four_d_np.max(),
            )
            peilin.plot_3DLiver(
                uq_np, name=f"UQ4D_{frame}", path="./tmp_plot", min=0, max=uq_np.max()
            )
            peilin.plot_3DLiver(
                four_d_np,
                name=f"LQ4D_{frame}",
                path="./tmp_plot",
                min=0,
                max=four_d_np.max(),
            )
        if ENABLE_UQ_DICOM_EXPORT:
            slice_count = export_dicom_volume(
                uq_restored,
                frame,
                "T2w_frame",
                "UQ-T2w 4D-MRI",
                header,
                t2_files,
                positions,
                output_path,
                identifiers,
                start,
                slice_count,
                clamp_negative=True,
            )
            time.sleep(1)

    if ENABLE_LQ_DICOM_EXPORT:
        for frame in range(four_d.shape[3]):
            fixed = (
                torch.from_numpy(
                    np.asarray(
                        img_norm(four_d[..., frame])[np.newaxis, np.newaxis, ...],
                        dtype=np.float32,
                    )
                )
                .float()
                .cpu()
            )
            spacing = [
                header.PixelSpacing[0],
                header.PixelSpacing[1],
                header.SliceThickness,
            ]
            restored_size = [
                int(value) for value in np.int16(np.array(t2_aligned.shape) / spacing)
            ]
            restored = F.interpolate(fixed, restored_size)[0, 0, :].cpu().numpy()
            slice_count = export_dicom_volume(
                restored,
                frame,
                "LQ_T2w_frame",
                "LQ-T2w 4D-MRI",
                header,
                t2_files,
                positions,
                output_path,
                identifiers,
                start,
                slice_count,
                clamp_negative=False,
            )
            time.sleep(1)


def main(args=None):
    """Run the modular reconstruction pipeline with original defaults."""
    arg = build_argument_parser().parse_args(args)
    device = os.environ.get("FREETUNE4D_DEVICE", "cuda:0")
    if device not in {"cpu", "cuda:0"}:
        raise ValueError(f"Unsupported FREETUNE4D_DEVICE: {device}")
    start = time.time()
    coarse_model, fine_model = load_models(
        arg.net_path_coarse, arg.net_path_fine, device
    )
    four_d, t2, header, t2_files, positions, output_path = load_inputs(
        arg.base_path, arg.MR_number, arg.st_date, arg.name_3d, arg.reference_file
    )
    header.PatientName = arg.MR_number
    header.PatientID = arg.MR_number
    reference = select_reference_frame(four_d, t2)
    aligned_t2 = run_initial_alignment(t2, reference)
    if ENABLE_QC_OUTPUTS:
        peilin.plot_3DLiver(
            four_d[..., 0], name="FourD_raw", path="./tmp_plot", min=0, max=1
        )
        peilin.plot_3DLiver(t2, name="T2_raw", path="./tmp_plot", min=0, max=1)
        peilin.plot_3DLiver(t2, name="T2", path="./tmp_plot", min=0, max=1)
        peilin.plot_3DLiver(
            reference, name="4D_selected", path="./tmp_plot", min=0, max=1
        )
        peilin.plot_3DLiver(
            aligned_t2, name="T2_warped", path="./tmp_plot", min=0, max=1
        )
    reconstruct_phases(
        four_d,
        aligned_t2,
        reference,
        header,
        t2_files,
        positions,
        output_path,
        coarse_model,
        fine_model,
        device,
        start,
    )


if __name__ == "__main__":
    main()
