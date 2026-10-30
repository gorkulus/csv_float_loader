"""CPU contract tests using the real ComfyUI API, without starting its server.

Set COMFYUI_PATH to a ComfyUI checkout and use its Python environment with -B.
Only this repository's nodes are imported; optional custom-node packs are not.
"""
import asyncio
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# Never create bytecode in the ComfyUI checkout, even without the suggested -B.
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if os.environ.get("COMFYUI_PATH"):
    sys.path.insert(0, str(Path(os.environ["COMFYUI_PATH"]).resolve()))

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("csv_nodes_under_test", ROOT / "nodes.py")
assert SPEC is not None and SPEC.loader is not None
nodes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nodes)
FIXTURE = ROOT / "tests" / "fixtures" / "example.csv"


def column(path: str | Path = FIXTURE, **overrides):
    args = dict(csv_path=str(path), column=2, has_header=True,
                skip_first_frames=0, frame_load_cap=0, select_every_nth=1,
                default_value=0.0)
    args.update(overrides)
    return nodes.CSVColumnFloats.execute(**args).args


def remap(values, **overrides):
    args = dict(mode="range_remap", in_min=0, in_max=10, out_min=0,
                out_max=1, gain=0.1, offset=0, clamp=False,
                clamp_min=0, clamp_max=1)
    args.update(overrides)
    return nodes.FloatsRemap.execute(values, **args).args[0]


def mask_result(mask, **overrides):
    args = dict(base_grow=0, grow_gain=1, base_blur=0, blur_gain=1, use_gpu=False)
    args.update(overrides)
    return nodes.GrowBlurMaskByFloats.execute(mask, **args).args[0]


class NodeContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def csv(self, text):
        path = Path(self.temp.name) / "input.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_registration_sockets_and_defaults(self):
        extension = asyncio.run(nodes.comfy_entrypoint())
        registered = asyncio.run(extension.get_node_list())
        self.assertEqual(len(registered), 5)
        self.assertEqual(len({n.define_schema().node_id for n in registered}), 5)
        for node in registered:
            schema = node.define_schema()
            self.assertFalse(schema.is_input_list)
            self.assertTrue(all(not out.is_output_list for out in schema.outputs))
        schema = nodes.CSVColumnFloats.define_schema()
        defaults = {item.id: getattr(item, "default", None) for item in schema.inputs}
        self.assertEqual(defaults, dict(csv_path="", column=5, has_header=True,
                                       skip_first_frames=0, frame_load_cap=0,
                                       select_every_nth=1, default_value=0.0))
        self.assertEqual(schema.outputs[0].io_type, "FLOATS")
        self.assertEqual(nodes.FloatsToStrengthList.define_schema().outputs[0].io_type, "FLOAT")

    def test_documented_example_and_selection_order(self):
        self.assertEqual(column(), ([0, 2, 4, 6, 8, 10], 6))
        self.assertEqual(column(skip_first_frames=1, select_every_nth=2,
                                frame_load_cap=2), ([2, 6], 2))
        self.assertEqual(column(column=99, default_value=-1), ([-1] * 6, 6))

    def test_bom_quotes_and_default_cells(self):
        path = self.csv('\ufeffframe,cv\n0,"2.5"\n1,\n2,no\n3\n\n')
        self.assertEqual(column(path, column=1, default_value=-1), ([2.5, -1, -1, -1, -1], 5))

    def test_header_heuristic_is_first_nonblank_within_sixteen(self):
        for row, expected in [(["0", "cv"], False), (["text", "1"], True),
                              (["", "text", "1"], True), (["", "2", "text"], False),
                              ([""] * 16 + ["text"], False), ([], False)]:
            with self.subTest(row=row):
                self.assertEqual(nodes._looks_like_header(row), expected)
        path = self.csv("0,cv\n1,2\n")
        self.assertEqual(column(path, column=1), ([0, 2], 2))
        path = self.csv("text,7\n1,2\n")
        self.assertEqual(column(path, column=1), ([2], 1))
        self.assertEqual(column(path, column=1, has_header=False), ([7, 2], 2))

    def test_nonfinite_values_parse_without_fallback(self):
        values, count = column(self.csv("cv\nNaN\nInf\n"), column=0)
        self.assertEqual(count, 2)
        self.assertTrue(math.isnan(values[0]))
        self.assertTrue(math.isinf(values[1]))

    def test_scalar_row_and_bounds(self):
        result = nodes.CSVFloatLoader.execute(str(FIXTURE), 1, True, -1).args
        self.assertEqual(result, (1.0, 0.04, 2.0) + (-1.0,) * 13)
        with self.assertRaises(IndexError):
            nodes.CSVFloatLoader.execute(str(FIXTURE), 6, True, 0)

    def test_empty_missing_path_and_selection_errors(self):
        with self.assertRaises(ValueError):
            column("")
        with self.assertRaises(FileNotFoundError):
            column(Path(self.temp.name) / "missing.csv")
        with self.assertRaises(ValueError):
            column(self.csv(""))
        with self.assertRaises(ValueError):
            column(skip_first_frames=6)

    def test_path_precedence_and_mtime_fingerprint(self):
        path = self.csv("0,1,2\n")
        cwd = Path(self.temp.name) / "cwd"
        cwd.mkdir()
        local = cwd / path.name
        local.write_text("0,1,9\n", encoding="utf-8")
        old_cwd = Path.cwd()
        try:
            os.chdir(cwd)
            with patch.object(nodes.folder_paths, "get_input_directory", return_value=self.temp.name):
                self.assertEqual(column(path.name), ([9], 1))
                local.unlink()
                self.assertEqual(column(path.name), ([2], 1))
        finally:
            os.chdir(old_cwd)
        before = nodes.CSVColumnFloats.fingerprint_inputs(str(path), 2, True, 0, 0, 1, 0)
        stamp = path.stat().st_mtime + 10
        os.utime(path, (stamp, stamp))
        after = nodes.CSVColumnFloats.fingerprint_inputs(str(path), 2, True, 0, 0, 1, 0)
        self.assertNotEqual(before, after)

    def test_remap_modes_and_edges(self):
        self.assertEqual(remap([0, 5, 10, 20]), [0, 0.5, 1, 2])
        self.assertEqual(remap([1, 2], mode="gain_offset", gain=2, offset=1), [3, 5])
        self.assertEqual(remap([5, 10], in_max=0, out_min=3, offset=2), [5, 5])
        self.assertEqual(remap([-10, 20], clamp=True, clamp_min=1, clamp_max=0), [0, 1])
        self.assertEqual(remap([]), [])

    def test_bridge_keeps_one_list_payload(self):
        self.assertEqual(nodes.FloatsToStrengthList.execute(("1", 2)).args, ([1.0, 2.0],))
        self.assertEqual(nodes.FloatsToStrengthList.execute(None).args, ([],))

    def test_mask_clamp_2d_dtype_and_device(self):
        mask = torch.tensor([[-1, 0.5], [1, 2]], dtype=torch.float64, device="cpu")
        out = mask_result(mask)
        self.assertTrue(torch.equal(out, mask.clamp(0, 1).unsqueeze(0)))
        self.assertEqual(out.dtype, mask.dtype)
        self.assertEqual(out.device, mask.device)
        self.assertTrue(torch.equal(mask_result(mask, grow_floats=[], blur_floats=[]), out))

    def test_mask_grow_short_lists_extra_values_and_rounding(self):
        mask = torch.zeros(3, 7, 7, device="cpu")
        mask[:, 3, 3] = 1
        out = mask_result(mask, grow_floats=[0, 1])
        self.assertEqual(out.sum(dim=(1, 2)).tolist(), [1, 9, 9])
        self.assertTrue(torch.equal(out, mask_result(mask, grow_floats=[0, 1, 1, 99])))
        rounded = mask_result(mask, grow_floats=[0.5, 1.5, -1])
        self.assertEqual(rounded.sum(dim=(1, 2)).tolist(), [1, 25, 0])

    def test_mask_blur_negative_sigma_and_chunk_boundary(self):
        mask = torch.zeros(9, 7, 7, dtype=torch.float64, device="cpu")
        mask[:, 3, 3] = 1
        out = mask_result(mask, blur_floats=[-1, 1])
        self.assertTrue(torch.equal(out[0], mask[0]))
        self.assertGreater(out[1, 3, 3].item(), 0)
        self.assertLess(out[1, 3, 3].item(), 1)
        self.assertTrue(torch.allclose(out[1], out[-1]))
        self.assertEqual(out.dtype, mask.dtype)
        self.assertEqual(out.device, mask.device)
        self.assertTrue(torch.all((out >= 0) & (out <= 1)).item())

    def test_mask_nan_grow_is_not_silently_replaced(self):
        with self.assertRaises(ValueError):
            mask_result(torch.zeros(1, 3, 3, device="cpu"), grow_floats=[float("nan")])


if __name__ == "__main__":
    unittest.main()
