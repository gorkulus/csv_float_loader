import csv
import os

import folder_paths

from comfy_api.latest import ComfyExtension, io


# ---------------------------------------------------------------------------
# Shared CSV helpers
# ---------------------------------------------------------------------------

def _resolve_csv_path(csv_path: str) -> str:
    """Resolve a user-supplied CSV path.

    Tries, in order: the path as given (absolute or relative to cwd), then
    relative to ComfyUI's input directory.
    """
    csv_path = (csv_path or "").strip().strip('"')
    if not csv_path:
        raise ValueError("No CSV path provided.")

    candidates = [csv_path]
    try:
        input_dir = folder_paths.get_input_directory()
        candidates.append(os.path.join(input_dir, csv_path))
    except Exception:
        pass

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(
        "CSV file not found. Tried: " + ", ".join(repr(c) for c in candidates)
    )


def _load_rows(csv_path: str) -> list[list[str]]:
    resolved = _resolve_csv_path(csv_path)
    with open(resolved, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        rows = [row for row in reader]
    return rows


def _looks_like_header(row: list[str]) -> bool:
    """A row is a header if none of its first cells parse as a float."""
    for cell in row[:16]:
        cell = (cell or "").strip()
        if cell == "":
            continue
        try:
            float(cell)
            return False
        except ValueError:
            return True
    return False


def _mtime_or_none(csv_path: str):
    try:
        return os.path.getmtime(_resolve_csv_path(csv_path))
    except Exception:
        return None


def _data_rows(csv_path: str, has_header: bool) -> list[list[str]]:
    rows = _load_rows(csv_path)
    if has_header and rows and _looks_like_header(rows[0]):
        rows = rows[1:]
    return rows


def _slice_like_vhs(rows, skip_first_frames, frame_load_cap, select_every_nth):
    """Mirror VideoHelperSuite's Load Video frame selection ORDER:
    skip first N, then take every Nth, then cap the count.
    This keeps a per-column CV list frame-aligned with a video loaded by VHS.
    """
    if skip_first_frames and skip_first_frames > 0:
        rows = rows[skip_first_frames:]
    step = max(1, int(select_every_nth))
    rows = rows[::step]
    if frame_load_cap and frame_load_cap > 0:
        rows = rows[:frame_load_cap]
    return rows


def _parse_cell(row, idx, default_value):
    if idx < 0 or idx >= len(row):
        return float(default_value)
    cell = (row[idx] or "").strip()
    if cell == "":
        return float(default_value)
    try:
        return float(cell)
    except ValueError:
        return float(default_value)


# A per-frame list of floats, interoperable with the Yvann Nodes "FLOATS" type.
Floats = io.Custom("FLOATS")


NUM_OUTPUTS = 16


# ---------------------------------------------------------------------------
# Original node: one row -> 16 scalar float outputs (per-frame iteration style)
# ---------------------------------------------------------------------------

class CSVFloatLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSVFloatLoader_LoadRow",
            display_name="CSV Float Loader (16 columns)",
            category="csv/float",
            description=(
                "Loads a CSV and outputs the first 16 columns of a chosen row "
                "as 16 float outputs. The 'row' input selects which data row."
            ),
            inputs=[
                io.String.Input(
                    "csv_path",
                    default="data.csv",
                    tooltip="CSV file: absolute path, or a name relative to ComfyUI's input/ folder.",
                ),
                io.Int.Input(
                    "row",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    step=1,
                    tooltip="Which data row to output (0-based). Header row is skipped automatically if present.",
                ),
                io.Boolean.Input(
                    "has_header",
                    default=True,
                    tooltip="If true, auto-detect and skip a non-numeric header row.",
                ),
                io.Float.Input(
                    "default_value",
                    default=0.0,
                    tooltip="Value used for missing/blank/non-numeric cells and missing columns.",
                ),
            ],
            outputs=[
                io.Float.Output(display_name=f"col_{i}") for i in range(NUM_OUTPUTS)
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, csv_path, row, has_header, default_value):
        return (csv_path, row, has_header, default_value, _mtime_or_none(csv_path))

    @classmethod
    def execute(cls, csv_path, row, has_header, default_value) -> io.NodeOutput:
        rows = _data_rows(csv_path, has_header)
        if not rows:
            raise ValueError("CSV contains no data rows.")
        if row < 0 or row >= len(rows):
            raise IndexError(
                f"Row index {row} out of range; CSV has {len(rows)} data row(s) (0..{len(rows) - 1})."
            )
        selected = rows[row]
        values = [_parse_cell(selected, i, default_value) for i in range(NUM_OUTPUTS)]
        return io.NodeOutput(*values)


# ---------------------------------------------------------------------------
# New node: one column -> a per-frame FLOATS list (schedule style)
# ---------------------------------------------------------------------------

class CSVColumnFloats(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSVFloatLoader_Column",
            display_name="CSV CV Column \u2192 FLOATS (frames)",
            category="csv/float",
            description=(
                "Reads ONE column of a CSV as a per-frame FLOATS list (one value "
                "per data row). The skip/cap/every-nth controls mirror VHS Load "
                "Video's frame selection so the CV stays aligned with the video."
            ),
            inputs=[
                io.String.Input(
                    "csv_path",
                    default="take-0002.csv",
                    tooltip="CSV file: absolute path, or a name relative to ComfyUI's input/ folder.",
                ),
                io.Int.Input(
                    "column",
                    default=5,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip="Which column (0-based) to read as the CV signal.",
                ),
                io.Boolean.Input(
                    "has_header",
                    default=True,
                    tooltip="If true, auto-detect and skip a non-numeric header row.",
                ),
                io.Int.Input(
                    "skip_first_frames",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    step=1,
                    tooltip="Mirror of VHS Load Video: skip this many leading rows/frames.",
                ),
                io.Int.Input(
                    "frame_load_cap",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    step=1,
                    tooltip="Mirror of VHS Load Video: max rows/frames to output (0 = all).",
                ),
                io.Int.Input(
                    "select_every_nth",
                    default=1,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Mirror of VHS Load Video: take every Nth row/frame.",
                ),
                io.Float.Input(
                    "default_value",
                    default=0.0,
                    tooltip="Value used for missing/blank/non-numeric cells.",
                ),
            ],
            outputs=[
                Floats.Output(display_name="cv_floats"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, csv_path, column, has_header, skip_first_frames,
                           frame_load_cap, select_every_nth, default_value):
        return (
            csv_path, column, has_header, skip_first_frames, frame_load_cap,
            select_every_nth, default_value, _mtime_or_none(csv_path),
        )

    @classmethod
    def execute(cls, csv_path, column, has_header, skip_first_frames,
                frame_load_cap, select_every_nth, default_value) -> io.NodeOutput:
        rows = _data_rows(csv_path, has_header)
        rows = _slice_like_vhs(rows, skip_first_frames, frame_load_cap, select_every_nth)
        if not rows:
            raise ValueError("No data rows after applying skip/cap/every-nth.")
        values = [_parse_cell(r, column, default_value) for r in rows]
        return io.NodeOutput(values, len(values))


# ---------------------------------------------------------------------------
# New node: scale / remap every value in a FLOATS list
# ---------------------------------------------------------------------------

class FloatsRemap(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSVFloatLoader_FloatsRemap",
            display_name="CV Floats Remap / Scale",
            category="csv/float",
            description=(
                "Applies the SAME scaling operation to every value in a FLOATS "
                "list. 'range_remap' maps in_min..in_max onto out_min..out_max "
                "(e.g. -10..10 -> 0..1, or -10..10 -> -1..1), THEN adds offset. "
                "'gain_offset' does value * gain + offset (handy for additive "
                "modulation). offset applies in both modes."
            ),
            inputs=[
                Floats.Input("floats", tooltip="Per-frame FLOATS list to transform."),
                io.Combo.Input(
                    "mode",
                    options=["range_remap", "gain_offset"],
                    tooltip="range_remap: linear in->out range. gain_offset: v*gain+offset.",
                ),
                io.Float.Input("in_min", default=-10.0, min=-1.0e6, max=1.0e6, step=0.01, tooltip="range_remap: input low."),
                io.Float.Input("in_max", default=10.0, min=-1.0e6, max=1.0e6, step=0.01, tooltip="range_remap: input high."),
                io.Float.Input("out_min", default=0.0, min=-1.0e6, max=1.0e6, step=0.01, tooltip="range_remap: output low."),
                io.Float.Input("out_max", default=1.0, min=-1.0e6, max=1.0e6, step=0.01, tooltip="range_remap: output high."),
                io.Float.Input("gain", default=0.1, min=-1.0e6, max=1.0e6, step=0.001, tooltip="gain_offset: multiplier."),
                io.Float.Input("offset", default=0.0, min=-1.0e6, max=1.0e6, step=0.01, tooltip="Added at the end in BOTH modes: after the range remap, or after value*gain."),
                io.Boolean.Input("clamp", default=False, tooltip="Clamp result to [clamp_min, clamp_max]."),
                io.Float.Input("clamp_min", default=0.0, min=-1.0e6, max=1.0e6, step=0.01),
                io.Float.Input("clamp_max", default=1.0, min=-1.0e6, max=1.0e6, step=0.01),
            ],
            outputs=[Floats.Output(display_name="floats")],
        )

    @classmethod
    def execute(cls, floats, mode, in_min, in_max, out_min, out_max, gain,
                offset, clamp, clamp_min, clamp_max) -> io.NodeOutput:
        data = list(floats) if floats is not None else []
        out = []
        span = (in_max - in_min)
        for v in data:
            v = float(v)
            if mode == "range_remap":
                t = 0.0 if span == 0 else (v - in_min) / span
                r = t * (out_max - out_min) + out_min + offset
            else:
                r = v * gain + offset
            if clamp:
                lo, hi = (clamp_min, clamp_max) if clamp_min <= clamp_max else (clamp_max, clamp_min)
                r = max(lo, min(hi, r))
            out.append(r)
        return io.NodeOutput(out)


# ---------------------------------------------------------------------------
# New node: per-frame grow + blur of a MASK batch, driven by FLOATS
# ---------------------------------------------------------------------------

class GrowBlurMaskByFloats(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSVFloatLoader_GrowBlurMaskByFloats",
            display_name="CV Grow/Blur Mask (per-frame)",
            category="csv/float",
            description=(
                "List-capable replacement for scalar Grow Mask + Mask Blur. For "
                "each frame i the grow pixels = base_grow + grow_gain*grow_floats[i] "
                "and the blur sigma = base_blur + blur_gain*blur_floats[i]. Leave a "
                "floats input unconnected to use only the base value for that effect."
            ),
            inputs=[
                io.Mask.Input("mask", tooltip="Mask batch [frames, H, W]."),
                Floats.Input("grow_floats", optional=True, tooltip="Per-frame grow modulation."),
                Floats.Input("blur_floats", optional=True, tooltip="Per-frame blur modulation."),
                io.Int.Input("base_grow", default=0, min=-512, max=512, step=1,
                             tooltip="Constant grow (px) added to grow modulation. Negative shrinks."),
                io.Float.Input("grow_gain", default=1.0, min=-1.0e6, max=1.0e6, step=0.01,
                               tooltip="Multiplier on grow_floats before adding to base_grow."),
                io.Float.Input("base_blur", default=0.0, min=0.0, step=0.01,
                               tooltip="Constant blur sigma added to blur modulation."),
                io.Float.Input("blur_gain", default=1.0, min=-1.0e6, max=1.0e6, step=0.01,
                               tooltip="Multiplier on blur_floats before adding to base_blur."),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @staticmethod
    def _at(flist, i):
        if not flist:
            return 0.0
        return float(flist[i if i < len(flist) else len(flist) - 1])

    @classmethod
    def execute(cls, mask, base_grow, grow_gain, base_blur, blur_gain,
                grow_floats=None, blur_floats=None) -> io.NodeOutput:
        import torch
        import torch.nn.functional as F

        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0)
        b, h, w = m.shape[0], m.shape[-2], m.shape[-1]

        def grow(x, px):  # x [1,1,H,W]
            if px == 0:
                return x
            k = 2 * abs(px) + 1
            pad = abs(px)
            if px > 0:
                return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
            return 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=pad)

        def blur(x, sigma):  # x [1,1,H,W]
            if sigma <= 0:
                return x
            radius = max(1, int(round(sigma * 3)))
            radius = min(radius, max(1, min(h, w) - 1))
            xs = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
            k1d = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
            k1d = k1d / k1d.sum()
            kx = k1d.view(1, 1, 1, -1)
            ky = k1d.view(1, 1, -1, 1)
            x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
            x = F.conv2d(x, kx)
            x = F.conv2d(x, ky)
            return x

        out = torch.empty_like(m)
        for i in range(b):
            frame = m[i].unsqueeze(0).unsqueeze(0).float()
            g = int(round(base_grow + grow_gain * cls._at(grow_floats, i)))
            s = max(0.0, base_blur + blur_gain * cls._at(blur_floats, i))
            frame = grow(frame, g)
            frame = blur(frame, s)
            out[i] = frame[0, 0].clamp(0.0, 1.0).to(out.dtype)

        return io.NodeOutput(out)


# ---------------------------------------------------------------------------
# New node: bridge a FLOATS list onto ACN's "FLOAT" float_strengths input
#
# Advanced-ControlNet's "Latent Keyframe From List" / "Timestep Keyframe From
# List" expose float_strengths as a plain "FLOAT" force_input. In this (V3)
# version of ACN those nodes are NOT declared INPUT_IS_LIST; their execute
# instead inspects the single incoming value and iterates it when it is an
# Iterable (isinstance(float_strengths, Iterable)). So the whole per-frame
# list must arrive as ONE object on that FLOAT socket.
#
# Therefore this bridge outputs is_output_list=False and passes the entire
# list through as a single value. Emitting a native per-item output list
# (is_output_list=True) instead triggers ComfyUI per-item expansion: the
# keyframe node -- and the ControlNet loader/apply downstream -- would run
# once PER FRAME, loading the ControlNet N times and exhausting VRAM. The
# only thing this node changes is the FLOATS->FLOAT type label; values pass
# through unmodified.
# ---------------------------------------------------------------------------


class FloatsToStrengthList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSVFloatLoader_FloatsToStrengthList",
            display_name="CV FLOATS \u2192 Strength List (ACN)",
            category="csv/float",
            description=(
                "Re-labels a FLOATS list onto a plain FLOAT socket so it can "
                "feed Advanced-ControlNet nodes whose float_strengths input is "
                "a FLOAT force_input, e.g. 'Latent Keyframe From List'. The "
                "whole list is delivered as ONE object (NOT a per-item output "
                "list), which is what those nodes iterate; emitting a per-item "
                "list would make ComfyUI run the ControlNet loader once per "
                "frame. Values pass through unmodified."
            ),
            inputs=[
                Floats.Input("floats", tooltip="Per-frame FLOATS list to pass through."),
            ],
            outputs=[
                io.Float.Output(display_name="float_strengths", is_output_list=False),
            ],
        )

    @classmethod
    def execute(cls, floats) -> io.NodeOutput:
        out = [float(v) for v in (floats if floats is not None else [])]
        return io.NodeOutput(out)


class CSVFloatLoaderExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CSVFloatLoader,
            CSVColumnFloats,
            FloatsRemap,
            GrowBlurMaskByFloats,
            FloatsToStrengthList,
        ]


async def comfy_entrypoint() -> CSVFloatLoaderExtension:
    return CSVFloatLoaderExtension()
