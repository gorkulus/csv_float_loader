# CSV CV FLOATS — ComfyUI custom nodes

Early experimental nodes for CSV control-voltage (CV) schedules and per-frame
mask modulation. Originally built to bring VCV Rack signals into AnimateDiff.
The author reports use on Windows and Linux with recent ComfyUI versions; this
is not a version-by-version compatibility guarantee. Models and workflow inputs
are not bundled, and not every graph is guaranteed to reproduce.

Built with Comfy Agent.

## Install

Requires an existing ComfyUI installation with the V3 API (`comfy_api.latest`)
and PyTorch. This pack adds no separate pip dependencies. From your ComfyUI root:

```sh
git clone https://github.com/gorkulus/csv_float_loader.git custom_nodes/csv_float_loader
```

Restart ComfyUI. Do not overwrite an existing installation. Yvann Nodes,
VideoHelperSuite (VHS), and Advanced-ControlNet (ACN) are optional integrations,
not import requirements.

CSV paths are checked first as supplied (absolute or relative to the ComfyUI
process working directory), then relative to ComfyUI's `input/` directory.
That directory is not a sandbox: other server-local readable paths also work.

## Nodes

All five nodes are in **`csv/float`**. `FLOATS` carries **one Python list per
execution**, not ComfyUI's per-item execution list. It interoperates with the
`FLOATS` type in [ComfyUI_Yvann-Nodes](https://github.com/yvann-ba/ComfyUI_Yvann-Nodes),
including its Floats Visualizer.

### CSV CV Column → FLOATS (frames) — `CSVFloatLoader_Column`

Reads one column; returns `cv_floats` and `count`. Columns are **0-based**.
Defaults: empty `csv_path` (enter a file), `column=5` (sixth column; choose yours),
`has_header=True`, `default_value=0`, skip `0`, every-nth `1`, cap `0` (unlimited).
The empty filename default affects new nodes, not paths stored in saved workflows.

Selection order is **skip → every nth → cap** after header detection. Prepare one
data row per source video frame in matching order. Frame numbers/timestamps are
ordinary columns: no sorting, FPS conversion, interpolation, or missing-frame
recovery occurs. With VHS, use matching selection settings and avoid changing
FPS unless the CSV was prepared for that exact frame sequence. Compare `count`
with the actual loaded image/mask batch length; matching settings alone do not
validate alignment. No rows after selection is an error.

### CV Floats Remap / Scale — `CSVFloatLoader_FloatsRemap`

Applies the same operation to every value, returning `FLOATS`:

- `range_remap`: linear `in_min..in_max` → `out_min..out_max`, then add `offset`.
- `gain_offset`: `value * gain + offset`.

Defaults: first mode `range_remap`, input `-10..10`, output `0..1`, gain `0.1`,
offset `0`, clamp disabled (bounds `0..1`). Without clamping, out-of-range inputs
extrapolate. Equal input bounds yield `out_min + offset`; reversed clamp bounds
are normalized. An empty sequence returns an empty sequence.

### CV Grow/Blur Mask (per-frame) — `CSVFloatLoader_GrowBlurMaskByFloats`

For each frame: grow pixels = `base_grow + grow_gain * grow_floats[i]`;
blur sigma = `base_blur + blur_gain * blur_floats[i]`. Defaults: both bases `0`,
both gains `1`, `use_gpu=False`, optional control inputs unconnected.

Use floating-point masks shaped `[frames,H,W]`; a 2D mask becomes one frame.
The mask batch determines output length. Short controls hold their last value,
extra values are ignored, and absent/empty controls add zero modulation. Match
lengths explicitly for strict alignment.

Grow uses Python's ties-to-even integer rounding and square dilation (positive)
or erosion (negative); gain can push it beyond the base widget's bounds.
Negative blur becomes zero. Blur radius is rounded from three sigma and capped
at `max(1, min(H,W)-1)`. Final masks are clamped to `[0,1]` and returned in the
input dtype/device. Large gains/radii and long batches can be expensive;
eight-frame processing chunks do not eliminate full-batch memory costs.

`use_gpu=True` permits CPU→CUDA computation when CUDA is available; `False`
keeps computation on the input device, **not necessarily CPU**. CPU/CUDA
numerical equivalence is not guaranteed.

### CSV Float Loader (16 columns) — `CSVFloatLoader_LoadRow`

Selects one 0-based data `row` and emits sixteen independent scalar `FLOAT`
outputs, `col_0` through `col_15`. It does not iterate over rows automatically.
Defaults: `csv_path=data.csv`, `row=0`, `has_header=True`, `default_value=0`.
Empty data or an out-of-range row is an error.

### CV FLOATS → Strength List (ACN) — `CSVFloatLoader_FloatsToStrengthList`

Converts each element with `float()` and carries the whole list through a
`FLOAT` socket as **one object** (`is_output_list=False`). Use only with
iterable-aware ACN `float_strengths` inputs, such as compatible **Latent Keyframe
From List** / **Timestep Keyframe From List** nodes. This is not a scalar
converter and is not suitable for arbitrary scalar-only `FLOAT` consumers.
Per-item output expansion would instead execute downstream nodes per element.

## CSV format and example

Use comma-delimited UTF-8 (optional BOM); quoted CSV fields are supported.
With `has_header=True`, only the first row is considered: its **first nonblank
cell among the first 16 cells** decides. Nonnumeric means remove the row;
numeric means keep it; all blank means keep it. Thus numeric-first mixed headers
remain, and text-first data rows can be lost. Disable detection for data-only
files if needed. Blank rows remain data rows. Missing columns and blank or
nonnumeric cells use `default_value`. NaN/infinity parse as floats but are unsafe
modulation inputs: supply finite values.

Copy [`tests/fixtures/example.csv`](tests/fixtures/example.csv) to
`ComfyUI/input/example.csv` (synthetic data only):

```csv
frame,time,cv
0,0.00,0
1,0.04,2
2,0.08,4
3,0.12,6
4,0.16,8
5,0.20,10
```

On the column node, use `csv_path=example.csv`, `column=2`, `has_header=True`,
`skip_first_frames=1`, `select_every_nth=2`, `frame_load_cap=2`:
`cv_floats=[2.0,6.0]`, `count=2`. To normalize the unselected sequence, remap
`0..10` to `0..1` with offset `0`. Connect to a matching mask batch's grow/blur
controls, or use the bridge for compatible ACN list inputs.

The entire CSV is read into memory on execution. File change detection uses
modification time, not content hashes; replacing content while preserving its
timestamp can leave cached results stale.

## CPU tests

Use your existing ComfyUI Python environment; no models, server, GPU, or optional
node packs are needed. From this repository (POSIX shell):

```sh
COMFYUI_PATH=/path/to/ComfyUI /path/to/ComfyUI/.venv/bin/python -B -m unittest discover -s tests -v
```

PowerShell (adjust the interpreter path for your installation):

```powershell
$env:COMFYUI_PATH = 'C:\path\to\ComfyUI'
& 'C:\path\to\ComfyUI\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
```

These direct CPU unit tests do not establish full-graph reproduction, Windows
execution, GPU equivalence, or compatibility with every ComfyUI release.

## License

Licensed under **GNU GPL version 3 or (at your option) any later version**
(`GPL-3.0-or-later`). See [LICENSE](LICENSE) for the full terms.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.
