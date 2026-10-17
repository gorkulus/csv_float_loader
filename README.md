# CSV CV FLOATS — ComfyUI custom nodes

A small ComfyUI node pack for driving a workflow from recorded control-voltage (CV)
data stored in a CSV — one row per video frame — and for applying that per-frame
signal to masks. Built originally to bring VCV Rack modulation (filter cutoff,
envelope/VCA, etc.) into an AnimateDiff pipeline, frame-aligned with a paired video.

The per-frame signal type is `FLOATS`, interoperable with the
[ComfyUI-Yvann-Nodes](https://github.com/yvann-ba/ComfyUI-Yvann-Nodes) `FLOATS`
type, so these nodes can feed Yvann's Floats Visualizer and friends.

## Requirements

- ComfyUI with the V3 node API (`comfy_api.latest`).
- No external Python packages. Uses only the standard library plus ComfyUI's
  own `folder_paths` and `torch`.

## Install

Clone (or copy) this folder into your ComfyUI `custom_nodes/` directory and
restart ComfyUI:

```
ComfyUI/custom_nodes/csv_float_loader/
```

## Nodes

All nodes appear under the **`csv/float`** category.

### CSV CV Column → FLOATS (frames) — `CSVFloatLoader_Column`
Reads ONE column of a CSV as a per-frame `FLOATS` list (one value per data row).
The `skip_first_frames` / `frame_load_cap` / `select_every_nth` controls mirror
VideoHelperSuite's Load Video frame selection ORDER, so the CV list stays aligned
with a video loaded by VHS. Outputs `cv_floats` and a `count`.
**Columns are 0-based** (the first column is `0`).

### CV Floats Remap / Scale — `CSVFloatLoader_FloatsRemap`
Applies the same scaling to every value in a `FLOATS` list.
- `range_remap`: linear map of `in_min..in_max` onto `out_min..out_max`, then `+ offset`.
- `gain_offset`: `value * gain + offset`.
`offset` is applied at the end in both modes. Optional clamp to `[clamp_min, clamp_max]`.

### CV Grow/Blur Mask (per-frame) — `CSVFloatLoader_GrowBlurMaskByFloats`
List-capable replacement for scalar Grow Mask + Mask Blur. For each frame `i`:
- grow px = `base_grow + grow_gain * grow_floats[i]` (negative grow shrinks)
- blur sigma = `base_blur + blur_gain * blur_floats[i]`
Leave either floats input unconnected to use only the base value for that effect.
Grow is quantized to integer pixels; blur sigma is continuous.

### CSV Float Loader (16 columns) — `CSVFloatLoader_LoadRow`
Loads a CSV and outputs the first 16 columns of a chosen `row` as 16 scalar float
outputs. Useful for per-row (rather than per-column) access.

## CSV format

One row per frame. Column 0 is typically a frame number, optionally followed by
timing columns, then one or more CV columns. A non-numeric header row is
auto-detected and skipped when `has_header` is true. Blank/non-numeric cells fall
back to `default_value`.

## License

TBD.
