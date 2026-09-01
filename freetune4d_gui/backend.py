"""Non-invasive adapter around the existing FreeTune4D command-line scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
from typing import Callable

from .devices import DeviceInfo, cpu_device, detect_cuda_devices, select_device


LogCallback = Callable[[str], None]


class BackendError(RuntimeError):
    """Structured, actionable pipeline or subprocess failure."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "backend",
        operation: str | None = None,
        command: tuple[str, ...] = (),
        exit_code: int | None = None,
        stdout_tail: str = "",
        stderr_tail: str = "",
        root_message: str | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.operation = operation
        self.command = command
        self.exit_code = exit_code
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail
        self.root_message = root_message or message

    @property
    def details(self) -> str:
        parts = [f"Operation: {self.operation or 'unknown'}", f"Category: {self.kind}"]
        if self.exit_code is not None:
            parts.append(f"Exit code: {self.exit_code}")
        if self.command:
            parts.append("Command: " + shlex.join(self.command))
        if self.stdout_tail:
            parts.extend(("", "--- stdout tail ---", self.stdout_tail))
        if self.stderr_tail:
            parts.extend(("", "--- stderr tail ---", self.stderr_tail))
        return "\n".join(parts)


@dataclass(frozen=True)
class PipelineConfig:
    dynamic_dicom_dir: Path
    static_dicom_dir: Path
    output_root: Path
    modality: str = "T2"
    coarse_model: Path = Path("models/coarse.h5")
    fine_model: Path = Path("models/fine.h5")
    reference_dicom: Path | None = None
    phase_count: int = 5
    cuda_diagnostics: bool = False
    compute_device: str = "auto"

    def normalized(self) -> "PipelineConfig":
        reference = self.reference_dicom.expanduser().resolve() if self.reference_dicom else None
        return PipelineConfig(
            dynamic_dicom_dir=self.dynamic_dicom_dir.expanduser().resolve(),
            static_dicom_dir=self.static_dicom_dir.expanduser().resolve(),
            output_root=self.output_root.expanduser().resolve(),
            modality=self.modality.upper(),
            coarse_model=self.coarse_model.expanduser().resolve(),
            fine_model=self.fine_model.expanduser().resolve(),
            reference_dicom=reference,
            phase_count=self.phase_count,
            cuda_diagnostics=self.cuda_diagnostics,
            compute_device=self.compute_device,
        )

    def signature(self) -> str:
        values = asdict(self.normalized())
        values.pop("cuda_diagnostics", None)
        # Compute hardware does not affect device-independent medical outputs.
        values.pop("compute_device", None)
        serializable = {key: str(value) for key, value in values.items()}
        return sha256(json.dumps(serializable, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OutputSummary:
    reconstructed_dir: Path
    qc_dir: Path
    uq_dicom_count: int
    lq_dicom_count: int
    respiratory_phases: int
    qc_file_count: int

    @property
    def uq_complete(self) -> bool:
        return self.uq_dicom_count > 0

    @property
    def lq_complete(self) -> bool:
        return self.lq_dicom_count > 0

    @property
    def qc_complete(self) -> bool:
        return self.qc_file_count > 0


class FreeTune4DBackend:
    """Stages user-selected inputs and invokes the unmodified backend scripts."""

    PREPROCESS_SCRIPT = "STEP_02_UTSW_ImageTest_YP_T2_Clinic_Amp_v2.py"
    RECONSTRUCT_SCRIPT = "4DMRI Synthesis_UTSW_DVFsmooth_YP_T2_Steps.py"
    SUPPORTED_MODALITIES = {"T2": True, "T1": False}

    def __init__(self, repo_root: Path | None = None, python_executable: str | None = None, device_detector=detect_cuda_devices):
        self.repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
        self.python_executable = python_executable or sys.executable
        self._device_detector = device_detector
        self.devices: list[DeviceInfo] = []
        self.refresh_devices()

    def refresh_devices(self) -> list[DeviceInfo]:
        self.devices = [cpu_device(), *self._device_detector()]
        return self.devices

    def resolve_device(self, config: PipelineConfig) -> DeviceInfo | None:
        return select_device(config.compute_device, self.devices)

    def device_error(self, config: PipelineConfig) -> str | None:
        selected = self.resolve_device(config)
        if selected and selected.available and selected.supported:
            return None
        if selected and selected.device_type == "cpu":
            return "Current backend requires CUDA. CPU execution has not been validated and is unavailable."
        if config.compute_device == "auto":
            return "No compatible CUDA GPU detected. CPU is unavailable in the current CUDA-only backend."
        return "The selected CUDA GPU is no longer available. Refresh Compute Device and select another GPU."

    def _configure_device_environment(self, config: PipelineConfig, env: dict[str, str], log: LogCallback) -> DeviceInfo:
        device = self.resolve_device(config)
        if device is None or not device.available or not device.supported:
            raise BackendError(self.device_error(config) or "CUDA GPU unavailable.", kind="invalid_input")
        env["CUDA_VISIBLE_DEVICES"] = str(device.physical_index)
        log(f"[DEVICE] Requested: {config.compute_device}")
        cuda_devices = [candidate for candidate in self.devices if candidate.device_type == "cuda"]
        log(f"[DEVICE] Detected CUDA GPUs: {len(cuda_devices)}")
        for candidate in cuda_devices:
            log(
                f"[DEVICE] GPU {candidate.physical_index}: {candidate.name}; "
                f"Total {candidate.total_gib:.2f} GiB; Free {candidate.free_gib:.2f} GiB"
            )
        log(f"[DEVICE] Selected physical GPU: {device.physical_index} — {device.name}")
        log(f"[DEVICE] CUDA_VISIBLE_DEVICES={device.physical_index}")
        log("[DEVICE] Backend logical CUDA device: cuda:0")
        log("[DEVICE] CUDA isolation applies to both TensorFlow and PyTorch before import/initialization.")
        return device

    def validate_inputs(self, config: PipelineConfig) -> list[str]:
        config = config.normalized()
        errors: list[str] = []
        if config.modality not in self.SUPPORTED_MODALITIES:
            errors.append("MRI modality must be T1 or T2.")
        elif not self.SUPPORTED_MODALITIES[config.modality]:
            errors.append("T1-weighted reconstruction is not implemented by the current backend.")
        self._validate_dicom_dir(config.dynamic_dicom_dir, "Dynamic LQ 4D-MRI", errors)
        self._validate_dicom_dir(config.static_dicom_dir, "Static UQ 3D-MRI", errors)
        if config.output_root.exists() and not config.output_root.is_dir():
            errors.append("Output path exists but is not a directory.")
        else:
            try:
                config.output_root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                errors.append(f"Output directory cannot be created: {exc}")
        for label, filename, model in (
            ("Coarse", "coarse.h5", config.coarse_model),
            ("Fine", "fine.h5", config.fine_model),
        ):
            if not model.is_file():
                errors.append(f"{label} model file not found. Please select a valid {filename} file under Model Settings.")
            elif model.suffix.lower() not in {".h5", ".hdf5"}:
                errors.append(f"{label} model must be an HDF5 file (.h5 or .hdf5). Please select it under Model Settings.")
        if config.reference_dicom and not config.reference_dicom.is_file():
            errors.append(f"Reference DICOM file does not exist: {config.reference_dicom}")
        elif config.reference_dicom and config.reference_dicom.parent != config.static_dicom_dir:
            errors.append("Reference DICOM must be located inside the selected Static UQ 3D-MRI directory.")
        if not 1 <= config.phase_count <= 65:
            errors.append("Respiratory phase count must be between 1 and 65.")
        for filename in (self.PREPROCESS_SCRIPT, self.RECONSTRUCT_SCRIPT, "parameters_Rigid.txt", "Par0020bspline2-MI-lesswarp.txt"):
            if not (self.repo_root / filename).exists():
                errors.append(f"Required backend resource is missing: {filename}")
        return errors

    @staticmethod
    def _dicom_candidates(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        candidates = sorted(path for path in directory.iterdir() if path.is_file() and (path.suffix.lower() == ".dcm" or path.name.startswith("IM-")))
        return candidates

    def _validate_dicom_dir(self, directory: Path, label: str, errors: list[str]) -> None:
        if not directory.is_dir():
            errors.append(f"{label} DICOM directory does not exist: {directory}")
        elif not self._dicom_candidates(directory):
            errors.append(f"No DICOM files were found in the {label} directory.")
        elif not any(path.name.startswith("IM-") and path.suffix.lower() == ".dcm" for path in directory.iterdir() if path.is_file()):
            errors.append(f"The current backend requires {label} files named IM-*.dcm.")

    def output_paths(self, config: PipelineConfig) -> tuple[Path, Path, Path]:
        root = config.normalized().output_root
        return root / "preprocessing", root / "reconstructed", root / "QC"

    def has_existing_outputs(self, config: PipelineConfig) -> bool:
        return any(path.exists() and any(path.iterdir()) for path in self.output_paths(config))

    def clear_managed_outputs(self, config: PipelineConfig) -> None:
        for path in self.output_paths(config):
            if path.exists():
                shutil.rmtree(path)

    def run_preprocessing(self, config: PipelineConfig, log: LogCallback = print) -> Path:
        config = config.normalized()
        errors = self.validate_inputs(config)
        if errors:
            raise BackendError("\n".join(errors), kind="invalid_input", operation="preprocessing")
        preprocessing, _, qc = self.output_paths(config)
        preprocessing.mkdir(parents=True, exist_ok=True)
        qc.mkdir(parents=True, exist_ok=True)
        patient_dir = self._prepare_staging_case(config, preprocessing)
        backend_base = preprocessing / "_backend_case"
        command = [
            self.python_executable,
            str(self.repo_root / self.PREPROCESS_SCRIPT),
            "--phase_num", str(config.phase_count),
            "--base_path", str(backend_base),
            "--MR_number", "case",
            "--st_date", "session",
        ]
        env = os.environ.copy()
        self._configure_device_environment(config, env, log)
        if config.cuda_diagnostics:
            env["CUDA_LAUNCH_BLOCKING"] = "1"
            log("CUDA diagnostic mode enabled for this run: CUDA_LAUNCH_BLOCKING=1")
        self._run("preprocessing", command, self.repo_root, log, env)
        phase_file = patient_dir / "phase_T2.mat"
        if not phase_file.is_file() or phase_file.stat().st_size == 0:
            raise BackendError(
                "Preprocessing process exited successfully, but phase_T2.mat was not generated.",
                kind="output_validation",
                operation="preprocessing",
            )
        shutil.copy2(phase_file, preprocessing / phase_file.name)
        full_data = patient_dir / "Full_data.mat"
        if full_data.is_file():
            shutil.copy2(full_data, preprocessing / full_data.name)
        self._copy_qc(patient_dir / "debug_vis", qc / "preprocessing")
        self._write_manifest(config, preprocessing / "case_manifest.json", "preprocessing_completed")
        return phase_file

    def validate_preprocessing_output(self, config: PipelineConfig) -> bool:
        preprocessing, _, _ = self.output_paths(config)
        manifest = preprocessing / "case_manifest.json"
        phase_file = preprocessing / "_backend_case" / "case" / "session" / "phase_T2.mat"
        if not manifest.is_file() or not phase_file.is_file() or phase_file.stat().st_size == 0:
            return False
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return data.get("signature") == config.signature() and data.get("status") == "preprocessing_completed"

    def run_motion_reconstruction(self, config: PipelineConfig, log: LogCallback = print) -> OutputSummary:
        config = config.normalized()
        if not self.validate_preprocessing_output(config):
            raise BackendError("Validated preprocessing output for the current case was not found.", kind="invalid_input", operation="motion reconstruction")
        preprocessing, reconstructed, qc = self.output_paths(config)
        patient_dir = preprocessing / "_backend_case" / "case" / "session"
        backend_base = preprocessing / "_backend_case"
        static_link = patient_dir / "T2_AX_MVXD"
        reference = config.reference_dicom or self._select_reference_dicom(config.static_dicom_dir)
        if not reference:
            raise BackendError("A reference DICOM could not be selected from the static MRI directory.", kind="invalid_input", operation="motion reconstruction")
        runtime_dir = preprocessing / "_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_runtime(runtime_dir)
        command = [
            self.python_executable,
            str(self.repo_root / self.RECONSTRUCT_SCRIPT),
            "--base_path", str(backend_base),
            "--MR_number", "case",
            "--st_date", "session",
            "--name_3d", static_link.name,
            "--reference_file", reference.name,
            "--net_path_coarse", str(config.coarse_model),
            "--net_path_fine", str(config.fine_model),
        ]
        env = os.environ.copy()
        self._configure_device_environment(config, env, log)
        pytorch_path = self.repo_root / "voxelmorph-master" / "pytorch"
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(pytorch_path), str(self.repo_root), env.get("PYTHONPATH", "")]))
        if config.cuda_diagnostics:
            env["CUDA_LAUNCH_BLOCKING"] = "1"
            log("CUDA diagnostic mode enabled for this run: CUDA_LAUNCH_BLOCKING=1")
        self._run("motion reconstruction", command, runtime_dir, log, env)
        backend_output = patient_dir / "UQ_4D_T2"
        if not backend_output.is_dir():
            raise BackendError(
                "Reconstruction process exited successfully, but UQ_4D_T2 was not generated.",
                kind="output_validation",
                operation="motion reconstruction",
            )
        reconstructed.mkdir(parents=True, exist_ok=True)
        self._copy_tree_contents(backend_output, reconstructed)
        self._copy_qc(runtime_dir / "tmp_plot", qc / "reconstruction")
        self._write_manifest(config, reconstructed / "case_manifest.json", "reconstruction_completed")
        return self.validate_reconstruction_output(config)

    def validate_reconstruction_output(self, config: PipelineConfig) -> OutputSummary:
        _, reconstructed, qc = self.output_paths(config)
        uq_files = list(reconstructed.glob("T2w_frame*_*.dcm"))
        lq_files = list(reconstructed.glob("LQ_T2w_frame*_*.dcm"))
        phases = set()
        for path in uq_files:
            match = re.search(r"T2w_frame(\d+)_", path.name)
            if match:
                phases.add(int(match.group(1)))
        reconstruction_qc = qc / "reconstruction"
        qc_files = [path for path in reconstruction_qc.rglob("*") if path.is_file()] if reconstruction_qc.is_dir() else []
        return OutputSummary(reconstructed, qc, len(uq_files), len(lq_files), len(phases), len(qc_files))

    def _prepare_staging_case(self, config: PipelineConfig, preprocessing: Path) -> Path:
        patient_dir = preprocessing / "_backend_case" / "case" / "session"
        patient_dir.mkdir(parents=True, exist_ok=True)
        self._replace_link(patient_dir / "DYNAMIC_THRIVE", config.dynamic_dicom_dir)
        self._replace_link(patient_dir / "T2_AX_MVXD", config.static_dicom_dir)
        return patient_dir

    def _prepare_runtime(self, runtime_dir: Path) -> None:
        for filename in ("parameters_Rigid.txt", "parameters_Affine.txt", "parameters_BSpline.txt", "Par0020bspline2-MI-lesswarp.txt"):
            source = self.repo_root / filename
            if source.exists():
                shutil.copy2(source, runtime_dir / filename)
        links = {
            "elastix-5.0.1-win64": self.repo_root / "elastix-5.0.1-win64",
            "matlab_elastix-master": self.repo_root / "matlab_elastix-master",
            "octave-tablicious-master": self.repo_root / "octave_tablicious_master",
        }
        for name, target in links.items():
            if target.exists():
                self._replace_link(runtime_dir / name, target)
        (runtime_dir / "yamlmatlab-master").mkdir(exist_ok=True)
        (runtime_dir / "addMfile").mkdir(exist_ok=True)

    @staticmethod
    def _replace_link(link: Path, target: Path) -> None:
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            shutil.copytree(target, link)

    @staticmethod
    def _copy_tree_contents(source: Path, destination: Path) -> None:
        for item in source.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    def _copy_qc(self, source: Path, destination: Path) -> None:
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            self._copy_tree_contents(source, destination)

    @staticmethod
    def _select_reference_dicom(directory: Path) -> Path | None:
        files = FreeTune4DBackend._dicom_candidates(directory)
        return files[0] if files else None

    @staticmethod
    def _write_manifest(config: PipelineConfig, path: Path, status: str) -> None:
        data = {"signature": config.signature(), "status": status, "modality": config.modality}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _classify_failure(text: str) -> tuple[str, str]:
        lowered = text.lower()
        meaningful = [line.strip() for line in text.splitlines() if line.strip()]
        python_exceptions = [
            line for line in meaningful
            if re.match(r"^(?:[\w.]+(?:Error|Exception)|AssertionError):", line)
        ]
        exception_lines = [line for line in meaningful if re.search(r"(?:error|exception|assert triggered|failed)", line, re.IGNORECASE)]
        root = python_exceptions[-1] if python_exceptions else (exception_lines[-1] if exception_lines else (meaningful[-1] if meaningful else "Backend process failed without diagnostic output."))
        if "outofmemoryerror" in lowered or "cuda out of memory" in lowered:
            return "cuda_oom", root
        if "cuda" in lowered or "cudnn" in lowered or "device-side assert" in lowered:
            return "cuda", root
        if any(token in lowered for token in ("load_weights", "model_weights", ".h5", "weight file")):
            return "model_loading", root
        if any(token in lowered for token in ("modulenotfounderror", "no module named", "command not found", "filenotfounderror")):
            return "missing_dependency", root
        if any(token in lowered for token in ("dicom", "invalid input", "shape mismatch", "no valid axial overlap")):
            return "invalid_input", root
        return "backend", root

    @staticmethod
    def _run(operation: str, command: list[str], cwd: Path, log: LogCallback, env: dict[str, str] | None = None) -> None:
        rendered_command = shlex.join(command)
        log(f"Backend operation: {operation}")
        log(f"Working directory: {cwd}")
        log(f"Python executable: {command[0]}")
        log(f"Exact command: {rendered_command}")
        log(f"CUDA_VISIBLE_DEVICES: {(env or os.environ).get('CUDA_VISIBLE_DEVICES', '<not set>')}")
        log(f"CUDA_LAUNCH_BLOCKING: {(env or os.environ).get('CUDA_LAUNCH_BLOCKING', '<not set>')}")
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise BackendError(
                f"Unable to start {operation}: {exc}",
                kind="missing_dependency",
                operation=operation,
                command=tuple(command),
                root_message=str(exc),
            ) from exc

        stdout_tail: deque[str] = deque(maxlen=120)
        stderr_tail: deque[str] = deque(maxlen=120)

        def read_stream(stream, prefix: str, target: deque[str]) -> None:
            for line in iter(stream.readline, ""):
                clean = line.rstrip("\r\n")
                target.append(clean)
                log(f"{prefix}{clean}")
            stream.close()

        assert process.stdout is not None and process.stderr is not None
        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, "[stdout] ", stdout_tail), daemon=True)
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, "[stderr] ", stderr_tail), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        if return_code:
            stdout_text = "\n".join(stdout_tail)
            stderr_text = "\n".join(stderr_tail)
            kind, root_message = FreeTune4DBackend._classify_failure("\n".join((stdout_text, stderr_text)))
            raise BackendError(
                f"{operation.capitalize()} failed: {root_message}",
                kind=kind,
                operation=operation,
                command=tuple(command),
                exit_code=return_code,
                stdout_tail=stdout_text,
                stderr_tail=stderr_text,
                root_message=root_message,
            )
