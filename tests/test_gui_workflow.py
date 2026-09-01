from pathlib import Path
import tempfile
import unittest

from freetune4d_gui.backend import BackendError, FreeTune4DBackend, PipelineConfig
from freetune4d_gui.app import FreeTune4DApp
from freetune4d_gui.controller import WorkflowController, WorkflowState
from freetune4d_gui.devices import DeviceInfo, cpu_device, select_device
from freetune4d_gui.typography import TYPOGRAPHY


class SimulatedBackend(FreeTune4DBackend):
    """Executes adapter behavior while simulating only the unavailable subprocess."""

    def __init__(self, repo_root):
        device = DeviceInfo(2, "Test GPU", 24 * 1024**3, 20 * 1024**3)
        super().__init__(repo_root, device_detector=lambda: [device])

    def _run(self, operation, command, cwd, log, env=None):
        self.last_operation = operation
        self.last_command = command
        self.last_env = env or {}
        log("simulated subprocess: " + " ".join(command))
        if self.PREPROCESS_SCRIPT in command[1]:
            base = Path(command[command.index("--base_path") + 1])
            patient = base / "case" / "session"
            (patient / "phase_T2.mat").write_bytes(b"MAT")
            (patient / "Full_data.mat").write_bytes(b"MAT")
            debug = patient / "debug_vis"
            debug.mkdir()
            (debug / "alignment.jpg").write_bytes(b"JPG")
        else:
            base = Path(command[command.index("--base_path") + 1])
            output = base / "case" / "session" / "UQ_4D_T2"
            output.mkdir()
            for frame in range(2):
                (output / f"T2w_frame{frame}_0.dcm").write_bytes(b"DICOM")
                (output / f"LQ_T2w_frame{frame}_0.dcm").write_bytes(b"DICOM")
            qc = Path(cwd) / "tmp_plot"
            qc.mkdir()
            (qc / "UQ4D_0.jpg").write_bytes(b"JPG")


class BackendAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.dynamic = root / "dynamic"
        self.static = root / "static"
        self.output = root / "output"
        self.models = root / "models"
        for directory in (self.dynamic, self.static, self.models):
            directory.mkdir()
        (self.dynamic / "IM-0001.dcm").write_bytes(b"DICOM")
        (self.static / "IM-0001.dcm").write_bytes(b"DICOM")
        (self.models / "coarse.h5").write_bytes(b"H5")
        (self.models / "fine.h5").write_bytes(b"H5")
        self.config = PipelineConfig(
            self.dynamic,
            self.static,
            self.output,
            "T2",
            self.models / "coarse.h5",
            self.models / "fine.h5",
        )
        self.backend = SimulatedBackend(Path(__file__).resolve().parents[1])

    def tearDown(self):
        self.temp.cleanup()

    def test_validation_requires_real_resources_and_rejects_t1(self):
        self.assertEqual([], self.backend.validate_inputs(self.config))
        t1 = PipelineConfig(**{**self.config.__dict__, "modality": "T1"})
        self.assertIn("not implemented", " ".join(self.backend.validate_inputs(t1)))
        missing = PipelineConfig(**{**self.config.__dict__, "coarse_model": self.models / "missing.h5"})
        self.assertIn("select a valid coarse.h5", " ".join(self.backend.validate_inputs(missing)))
        wrong_extension = self.models / "coarse.bin"
        wrong_extension.write_bytes(b"model")
        invalid_type = PipelineConfig(**{**self.config.__dict__, "coarse_model": wrong_extension})
        self.assertIn("HDF5", " ".join(self.backend.validate_inputs(invalid_type)))

    def test_real_adapter_contract_creates_required_output_structure(self):
        logs = []
        self.backend.run_preprocessing(self.config, logs.append)
        preprocessing, reconstructed, qc = self.backend.output_paths(self.config)
        self.assertTrue((preprocessing / "phase_T2.mat").is_file())
        self.assertTrue(self.backend.validate_preprocessing_output(self.config))
        summary = self.backend.run_motion_reconstruction(self.config, logs.append)
        self.assertTrue(reconstructed.is_dir())
        self.assertTrue(qc.is_dir())
        self.assertEqual(2, summary.respiratory_phases)
        self.assertEqual(2, summary.uq_dicom_count)
        self.assertEqual(2, summary.lq_dicom_count)
        self.assertGreater(summary.qc_file_count, 0)
        self.assertTrue(any("STEP_02" in line for line in logs))
        self.assertEqual(str(self.config.coarse_model), self.backend.last_command[self.backend.last_command.index("--net_path_coarse") + 1])
        self.assertEqual(str(self.config.fine_model), self.backend.last_command[self.backend.last_command.index("--net_path_fine") + 1])

    def test_manifest_prevents_cross_case_reconstruction(self):
        self.backend.run_preprocessing(self.config, lambda _line: None)
        changed = PipelineConfig(**{**self.config.__dict__, "phase_count": 6})
        self.assertFalse(self.backend.validate_preprocessing_output(changed))
        with self.assertRaises(BackendError):
            self.backend.run_motion_reconstruction(changed)

    def test_structured_process_error_retains_both_streams_and_root_cause(self):
        logs = []
        command = [
            self.backend.python_executable,
            "-c",
            "import sys; print('normal context'); print('RuntimeError: CUDA device-side assert triggered', file=sys.stderr); raise SystemExit(7)",
        ]
        with self.assertRaises(BackendError) as caught:
            FreeTune4DBackend._run("preprocessing", command, Path(self.temp.name), logs.append)
        error = caught.exception
        self.assertEqual("cuda", error.kind)
        self.assertEqual(7, error.exit_code)
        self.assertIn("normal context", error.stdout_tail)
        self.assertIn("device-side assert", error.stderr_tail)
        self.assertEqual("RuntimeError: CUDA device-side assert triggered", error.root_message)
        self.assertIn("Exact command:", "\n".join(logs))
        self.assertIn("[stderr]", "\n".join(logs))

    def test_cuda_out_of_memory_has_specific_category(self):
        kind, message = FreeTune4DBackend._classify_failure(
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 932.00 MiB"
        )
        self.assertEqual("cuda_oom", kind)
        self.assertIn("OutOfMemoryError", message)

    def test_cuda_diagnostic_mode_is_scoped_to_backend_child(self):
        diagnostic = PipelineConfig(**{**self.config.__dict__, "cuda_diagnostics": True})
        self.backend.run_preprocessing(diagnostic, lambda _line: None)
        self.assertEqual("1", self.backend.last_env.get("CUDA_LAUNCH_BLOCKING"))
        self.assertNotEqual("1", __import__("os").environ.get("CUDA_LAUNCH_BLOCKING"))


class WorkflowStateTests(unittest.TestCase):
    def setUp(self):
        root = Path("/tmp/gui-state")
        self.config = PipelineConfig(root / "dynamic", root / "static", root / "output", coarse_model=root / "coarse.h5", fine_model=root / "fine.h5")
        self.controller = WorkflowController()

    def test_strict_success_transition(self):
        self.controller.inputs_validated(self.config, [])
        self.assertTrue(self.controller.can_preprocess)
        self.assertFalse(self.controller.can_reconstruct)
        self.controller.start_preprocessing()
        self.assertFalse(self.controller.can_reconstruct)
        self.controller.preprocessing_succeeded()
        self.assertTrue(self.controller.can_reconstruct)
        self.controller.start_reconstruction()
        self.controller.reconstruction_succeeded()
        self.assertEqual(WorkflowState.COMPLETED, self.controller.state)

    def test_input_change_invalidates_preprocessing(self):
        self.controller.inputs_validated(self.config, [])
        self.controller.start_preprocessing()
        self.controller.preprocessing_succeeded()
        changed = PipelineConfig(**{**self.config.__dict__, "output_root": Path("/tmp/other-case")})
        self.controller.inputs_validated(changed, [])
        self.assertFalse(self.controller.preprocessing_valid)
        self.assertFalse(self.controller.can_reconstruct)
        self.assertEqual(WorkflowState.READY, self.controller.state)

    def test_failure_recovery(self):
        self.controller.inputs_validated(self.config, [])
        self.controller.start_preprocessing()
        self.controller.operation_failed("preprocessing")
        self.controller.recover_after_failure()
        self.assertEqual(WorkflowState.READY, self.controller.state)
        self.assertFalse(self.controller.can_reconstruct)

    def test_failed_preprocessing_is_visible_but_retryable(self):
        self.controller.inputs_validated(self.config, [])
        self.controller.start_preprocessing()
        self.controller.operation_failed("preprocessing")
        self.assertEqual(WorkflowState.FAILED, self.controller.state)
        self.assertTrue(self.controller.can_preprocess)
        self.assertFalse(self.controller.can_reconstruct)
        self.controller.start_preprocessing()
        self.assertEqual(WorkflowState.PREPROCESSING, self.controller.state)

    def test_cuda_diagnostic_toggle_does_not_invalidate_preprocessing(self):
        self.controller.inputs_validated(self.config, [])
        self.controller.start_preprocessing()
        self.controller.preprocessing_succeeded()
        diagnostic = PipelineConfig(**{**self.config.__dict__, "cuda_diagnostics": True})
        self.controller.inputs_validated(diagnostic, [])
        self.assertEqual(WorkflowState.PREPROCESSED, self.controller.state)
        self.assertTrue(self.controller.can_reconstruct)

    def test_compute_device_change_preserves_device_independent_preprocessing(self):
        self.controller.inputs_validated(self.config, [])
        self.controller.start_preprocessing()
        self.controller.preprocessing_succeeded()
        changed = PipelineConfig(**{**self.config.__dict__, "compute_device": "cuda:2"})
        self.controller.inputs_validated(changed, [])
        self.assertEqual(WorkflowState.PREPROCESSED, self.controller.state)
        self.assertTrue(self.controller.preprocessing_valid)


class GuiLayoutContractTests(unittest.TestCase):
    def test_readable_fonts_and_responsive_layout_contract(self):
        self.assertEqual((65, 35), FreeTune4DApp.COLUMN_WEIGHTS)
        self.assertGreaterEqual(FreeTune4DApp.MINIMUM_SIZE[0], 1100)
        self.assertGreaterEqual(FreeTune4DApp.MINIMUM_SIZE[1], 720)
        self.assertEqual(24, FreeTune4DApp.FONT_SIZES["title"])
        self.assertEqual(20, FreeTune4DApp.FONT_SIZES["section"])
        self.assertEqual(17, FreeTune4DApp.FONT_SIZES["normal"])
        self.assertEqual(15, FreeTune4DApp.FONT_SIZES["log"])
        self.assertGreaterEqual(TYPOGRAPHY.CONTROL_HEIGHT_PX, 36)
        self.assertGreaterEqual(TYPOGRAPHY.PRIMARY_HEIGHT_PX, 44)
        self.assertEqual(700, TYPOGRAPHY.TITLE_WEIGHT)
        self.assertEqual(600, TYPOGRAPHY.SECTION_WEIGHT)
        self.assertEqual(500, TYPOGRAPHY.LABEL_WEIGHT)
        self.assertEqual(400, TYPOGRAPHY.BODY_WEIGHT)


class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_is_first_class_and_supported(self):
        cpu = cpu_device()
        self.assertEqual("cpu", cpu.key)
        self.assertTrue(cpu.available)
        self.assertTrue(cpu.supported)
        self.assertEqual("CPU — slower", cpu.display_name)
        self.assertIs(cpu, select_device("cpu", [cpu]))

    def test_auto_selects_gpu_with_most_free_memory(self):
        devices = [
            DeviceInfo(0, "Busy GPU", 24 * 1024**3, 200 * 1024**2),
            DeviceInfo(1, "Available GPU", 24 * 1024**3, 20 * 1024**3),
        ]
        self.assertEqual(1, select_device("auto", devices).physical_index)
        self.assertTrue(devices[0].low_memory)

    def test_manual_physical_gpu_is_isolated_for_child(self):
        device = DeviceInfo(2, "Selected GPU", 24 * 1024**3, 18 * 1024**3)
        backend = FreeTune4DBackend(device_detector=lambda: [device])
        config = PipelineConfig(Path("."), Path("."), Path("."), compute_device="cuda:2")
        env = {}
        backend._configure_device_environment(config, env, lambda _message: None)
        self.assertEqual("2", env["CUDA_VISIBLE_DEVICES"])
        self.assertEqual("cuda:0", env["FREETUNE4D_DEVICE"])

    def test_backend_maps_cpu_to_hidden_cuda_and_cpu_runtime(self):
        backend = FreeTune4DBackend(device_detector=lambda: [])
        config = PipelineConfig(Path("."), Path("."), Path("."), compute_device="cpu")
        self.assertIsNone(backend.device_error(config))
        env = {}
        backend._configure_device_environment(config, env, lambda _message: None)
        self.assertEqual("", env["CUDA_VISIBLE_DEVICES"])
        self.assertEqual("cpu", env["FREETUNE4D_DEVICE"])

    def test_auto_falls_back_to_cpu_without_gpu(self):
        backend = FreeTune4DBackend(device_detector=lambda: [])
        config = PipelineConfig(Path("."), Path("."), Path("."), compute_device="auto")
        self.assertEqual("cpu", backend.resolve_device(config).key)

    def test_active_pipeline_scripts_consume_runtime_device(self):
        root = Path(__file__).resolve().parents[1]
        preprocessing = (root / FreeTune4DBackend.PREPROCESS_SCRIPT).read_text(encoding="utf-8")
        reconstruction = (root / FreeTune4DBackend.RECONSTRUCT_SCRIPT).read_text(encoding="utf-8")
        self.assertIn('RUNTIME_DEVICE = os.environ.get("FREETUNE4D_DEVICE"', preprocessing)
        self.assertIn('device = os.environ.get("FREETUNE4D_DEVICE"', reconstruction)
        self.assertIn('device_vxm = "/CPU:0"', reconstruction)


if __name__ == "__main__":
    unittest.main()
