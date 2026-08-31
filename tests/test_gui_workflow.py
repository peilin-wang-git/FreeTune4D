from pathlib import Path
import tempfile
import unittest

from freetune4d_gui.backend import BackendError, FreeTune4DBackend, PipelineConfig
from freetune4d_gui.app import FreeTune4DApp
from freetune4d_gui.controller import WorkflowController, WorkflowState


class SimulatedBackend(FreeTune4DBackend):
    """Executes adapter behavior while simulating only the unavailable subprocess."""

    def _run(self, operation, command, cwd, log, env=None):
        self.last_operation = operation
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
        self.assertIn("Coarse model", " ".join(self.backend.validate_inputs(missing)))

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


class GuiLayoutContractTests(unittest.TestCase):
    def test_readable_fonts_and_responsive_layout_contract(self):
        self.assertEqual((65, 35), FreeTune4DApp.COLUMN_WEIGHTS)
        self.assertGreaterEqual(FreeTune4DApp.MINIMUM_SIZE[0], 1100)
        self.assertGreaterEqual(FreeTune4DApp.MINIMUM_SIZE[1], 720)
        self.assertEqual(20, FreeTune4DApp.FONT_SIZES["title"])
        self.assertEqual(14, FreeTune4DApp.FONT_SIZES["section"])
        self.assertEqual(12, FreeTune4DApp.FONT_SIZES["normal"])
        self.assertEqual(11, FreeTune4DApp.FONT_SIZES["log"])


if __name__ == "__main__":
    unittest.main()
