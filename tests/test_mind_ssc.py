"""Sanity and contract tests for the 3D MIND-SSC registration metric."""

import unittest

import numpy as np
import torch

from peilin_loss import MINDSSC, compute_mind_ssc_distance, mind_ssc_descriptor


class MindSscTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        self.image = torch.rand((1, 1, 12, 13, 14), generator=generator)

    def test_descriptor_shape_and_finite_values(self):
        descriptor = mind_ssc_descriptor(self.image)
        self.assertEqual((1, 12, 12, 13, 14), tuple(descriptor.shape))
        self.assertTrue(torch.isfinite(descriptor).all())

    def test_identical_inputs_have_zero_distance(self):
        self.assertEqual(0.0, compute_mind_ssc_distance(self.image, self.image).item())
        self.assertEqual(0.0, MINDSSC().loss(self.image, self.image.clone()).item())

    def test_positive_intensity_scaling_is_more_stable_than_raw_mse(self):
        scaled = self.image * 3.0
        mind_distance = compute_mind_ssc_distance(self.image, scaled).item()
        raw_mse = torch.mean((self.image - scaled) ** 2).item()
        self.assertLess(mind_distance, raw_mse)

    def test_spatial_shift_increases_distance(self):
        identical = compute_mind_ssc_distance(self.image, self.image).item()
        shifted = torch.roll(self.image, shifts=3, dims=4)
        shifted_distance = compute_mind_ssc_distance(self.image, shifted).item()
        self.assertGreater(shifted_distance, identical)

    def test_constant_image_is_finite(self):
        descriptor = mind_ssc_descriptor(torch.ones((9, 10, 11)))
        self.assertTrue(torch.isfinite(descriptor).all())
        self.assertEqual(
            0.0,
            compute_mind_ssc_distance(
                np.ones((9, 10, 11)), np.ones((9, 10, 11))
            ).item(),
        )

    def test_different_spatial_shapes_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "MIND-SSC requires fixed and moving images on the same spatial grid",
        ):
            compute_mind_ssc_distance(torch.ones((8, 9, 10)), torch.ones((8, 9, 11)))

    def test_mask_is_applied_only_to_final_aggregation(self):
        mask = torch.zeros_like(self.image)
        mask[..., 2:8, 2:8, 2:8] = 1
        distance = compute_mind_ssc_distance(
            self.image, torch.roll(self.image, 1, 2), mask=mask
        )
        self.assertTrue(torch.isfinite(distance))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_configured_gpu_device(self):
        image = self.image.cuda()
        descriptor = mind_ssc_descriptor(image)
        self.assertEqual(image.device, descriptor.device)
        self.assertEqual(
            image.device, MINDSSC(device=image.device).loss(image, image).device
        )


if __name__ == "__main__":
    unittest.main()
