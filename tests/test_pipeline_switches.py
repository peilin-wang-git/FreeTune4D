"""Static contracts for the two directly executable pipeline switch sets."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / "STEP_02_UTSW_ImageTest_YP_T2_Clinic_Amp_v2.py"
RECONSTRUCT = ROOT / "4DMRI Synthesis_UTSW_DVFsmooth_YP_T2_Steps.py"


def assigned_constants(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def load_function(path, function_name, namespace):
    """Compile one dependency-light function from a script for switch tests."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    ast.fix_missing_locations(function)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[function_name]


class PipelineSwitchTests(unittest.TestCase):
    def test_preprocessing_defaults_preserve_original_pipeline(self):
        values = assigned_constants(PREPROCESS)
        self.assertEqual("LCC", values["AXIAL_ALIGNMENT_METRIC"])
        for name in (
            "ENABLE_SORTING",
            "ENABLE_XY_ALIGNMENT",
            "ENABLE_AXIAL_ALIGNMENT",
            "ENABLE_QC_OUTPUTS",
            "ENABLE_INTERMEDIATE_OUTPUTS",
        ):
            self.assertIs(values[name], True)

    def test_reconstruction_defaults_preserve_original_pipeline(self):
        values = assigned_constants(RECONSTRUCT)
        for name in (
            "ENABLE_AFFINE_ALIGNMENT",
            "ENABLE_BSPLINE_ALIGNMENT",
            "ENABLE_COARSE_ALIGNMENT",
            "ENABLE_DVF_SMOOTHING",
            "ENABLE_FINE_ALIGNMENT",
            "ENABLE_QC_OUTPUTS",
            "ENABLE_INTERMEDIATE_OUTPUTS",
            "ENABLE_UQ_DICOM_EXPORT",
            "ENABLE_LQ_DICOM_EXPORT",
        ):
            self.assertIs(values[name], True)

    def test_both_scripts_retain_direct_execution(self):
        for path in (PREPROCESS, RECONSTRUCT):
            source = path.read_text(encoding="utf-8")
            self.assertIn('if __name__ == "__main__":', source)
            self.assertIn("main()", source)

    def test_rigid_and_bspline_switches_support_all_combinations(self):
        for rigid_enabled in (False, True):
            for bspline_enabled in (False, True):
                calls = []

                def elastix(image, _reference, _path, mode):
                    calls.append(mode)
                    return image + mode

                namespace = {
                    "ENABLE_AFFINE_ALIGNMENT": rigid_enabled,
                    "ENABLE_BSPLINE_ALIGNMENT": bspline_enabled,
                    "ENABLE_QC_OUTPUTS": False,
                    "run_elastix": elastix,
                }
                run_initial_alignment = load_function(
                    RECONSTRUCT, "run_initial_alignment", namespace
                )
                result = run_initial_alignment("T2", "fixed")
                expected_calls = []
                if rigid_enabled:
                    expected_calls.append("Rigid")
                if bspline_enabled:
                    expected_calls.append("BSpline")
                self.assertEqual(expected_calls, calls)
                self.assertEqual("T2" + "".join(expected_calls), result)

    def test_metric_directions_include_lower_is_better_mind(self):
        namespace = {}
        direction = load_function(PREPROCESS, "alignment_metric_direction", namespace)
        for metric in ("LCC", "NCC", "SSIM", "NMI"):
            self.assertTrue(direction(metric))
        self.assertFalse(direction("MIND"))
        with self.assertRaisesRegex(ValueError, "Unsupported axial alignment metric"):
            direction("unknown")


if __name__ == "__main__":
    unittest.main()
