import importlib.util
import os
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from ttabc.model_selection.moon import solve_moon
from ttabc.model_selection.stata import solve_stata
from ttabc.utils.sampler import LabelShiftOnlineBatchSampler, extract_dataset_labels


class SolverTests(unittest.TestCase):
    def features(self, device="cpu"):
        torch.manual_seed(2022)
        common = F.normalize(torch.randn(512, device=device), dim=-1)
        text = F.normalize(0.5 * common +
                           F.normalize(torch.randn(20, 512, device=device), dim=-1), dim=-1)
        image = F.normalize(0.5 * common + 0.1 * text[torch.arange(64, device=device) % 6] +
                            F.normalize(torch.randn(64, 512, device=device), dim=-1), dim=-1)
        return image, 100.0 * image @ text.t(), text

    def test_probabilities_and_no_mutation(self):
        image, logits, text = self.features()
        originals = [x.clone() for x in (image, logits, text)]
        for soft in (False, True):
            output = solve_moon(image, logits, text, soft_beta=soft)
            self.assertEqual(output.shape, (64, 20))
            self.assertFalse(output.requires_grad)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output.sum(dim=1), torch.ones(64))
        for original, value in zip(originals, (image, logits, text)):
            torch.testing.assert_close(original, value, rtol=0, atol=0)

    def test_small_batches_and_extreme_logits(self):
        image, logits, text = self.features()
        for solver in (solve_stata, solve_moon):
            for n in (1, 2, 3):
                output = solver(image[:n].half(), logits[:n].half(), text.half())
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output.sum(dim=1), torch.ones(n))
            torch.testing.assert_close(solver(image[:1], logits[:1], text),
                                       logits[:1].softmax(dim=1))
        output = solve_moon(image, logits * 1000, text, n_neighbors=0)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.sum(dim=1), torch.ones(64))
        torch.testing.assert_close(solve_moon(image, logits[:, :1], text[:1]), torch.ones(64, 1))

    def test_invalid_shapes(self):
        image, logits, text = self.features()
        with self.assertRaises(ValueError):
            solve_moon(image, logits[:, :-1], text)
        with self.assertRaises(ValueError):
            solve_moon(image, logits, text, n_neighbors=-1)

    @unittest.skipUnless(os.environ.get("MOON_UPSTREAM") and torch.cuda.is_available(),
                         "set MOON_UPSTREAM to official solvers/MOON.py and enable CUDA")
    def test_official_solver_parity(self):
        spec = importlib.util.spec_from_file_location("official_moon", os.environ["MOON_UPSTREAM"])
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        image, logits, text = self.features("cuda")
        for soft in (False, True):
            for neighbors, iterations in ((3, 10), (0, 10), (3, 0)):
                kwargs = dict(soft_beta=soft, n_neighbors=neighbors, max_iter=iterations)
                with torch.no_grad():
                    _, expected = upstream.MOON_solver(
                        image, torch.zeros(64, device="cuda"), text.t().unsqueeze(0), **kwargs)
                    actual = solve_moon(image, logits, text, **kwargs).cpu()
                print(f"official parity {kwargs}: max error={(actual - expected).abs().max().item():.3g}")
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
                torch.testing.assert_close(actual.argmax(1), expected.argmax(1), rtol=0, atol=0)


class SamplerTests(unittest.TestCase):
    def test_seed_and_coverage(self):
        labels = np.repeat(np.arange(20), 10)
        for gamma in (0.1, 0.01, 0.001, -1):
            streams = []
            for _ in range(2):
                np.random.seed(2022)
                sampler = LabelShiftOnlineBatchSampler(labels, gamma, 8, 64, seed=2022)
                streams.append(sampler.indices)
                self.assertEqual(sorted(sampler.indices), list(range(200)))
                self.assertEqual(len(list(sampler)), 3)
                self.assertEqual(sum(map(len, sampler)), 192)
            self.assertEqual(streams[0], streams[1])

    def test_subset_labels(self):
        dataset = torch.utils.data.TensorDataset(torch.zeros(6))
        dataset.targets = [0, 0, 1, 1, 2, 2]
        subset = torch.utils.data.Subset(dataset, [4, 1, 3])
        np.testing.assert_array_equal(extract_dataset_labels(subset), [2, 0, 1])


if __name__ == "__main__":
    unittest.main()
