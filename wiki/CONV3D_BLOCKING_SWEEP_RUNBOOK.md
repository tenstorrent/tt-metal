# Conv3d Blocking Sweep Runbook

How to sweep, record, and update conv3d blockings for a new mesh/resolution/t_chunk
configuration on BH Loud Box (2x4 or 1x1 for per-device 4x8/4x32 shapes).

---

## Overview

The brute-force sweep benchmarks every valid `(C_in_block, C_out_block, T_out_block,
H_out_block, W_out_block)` combination for each VAE decoder conv3d layer, finds the
fastest, and writes results to `sweep_results_<tag>/` JSON files.  After the sweep,
update the blocking table in `models/tt_dit/utils/conv3d.py` and run the perf test
to confirm the improvement.

---

## Key files

| File | Purpose |
|------|---------|
| `models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py` | Sweep script — run via pytest |
| `models/tt_dit/utils/conv3d.py` | Blocking table — update after sweep |
| `models/tt_dit/tests/models/wan2_2/CONV3D_PRODUCTION_SHAPES.md` | Per-device H/W dims for every mesh config |
| `models/tt_dit/tests/models/wan2_2/test_performance_wan.py` | End-to-end perf test |
| `sweep_results_<tag>/` | JSON output directory (one file per layer) |
| `sweep_log_<tag>.txt` | Full pytest tee'd output |

---

## Step 0: Prerequisites

```bash
cd /localdev/kevinmi/tt-metal
source python_env/bin/activate
export PYTHONPATH=$(pwd)
export TT_DIT_CACHE_DIR=/localdev/kevinmi/.cache
```

Confirm the device mesh is healthy:

```bash
tt-smi -ls
```

If you see ARC/NOC errors, reset all devices before starting:

```bash
tt-smi -r 0,1,2,3,4,5,6,7
```

---

## Step 1: Understand the shapes

Read `CONV3D_PRODUCTION_SHAPES.md` to find per-device `H` and `W` for your target
mesh config. The sweep uses **padded** H/W as input:

- For `(3,3,3)` and `(1,3,3)` kernels: `H_sweep = H_unpadded + 2`, `W_sweep = W_unpadded + 2`
- For `(3,1,1)` kernels: no spatial padding, `H_sweep = H_unpadded`, `W_sweep = W_unpadded`

The blocking table key uses `H_out = H_sweep - (kH - 1)` and `W_out = W_sweep - (kW - 1)`,
which equals the unpadded `H_unpadded` for `(0,1,1)` internal padding.

For cached t_chunk configs, compute T values from `compute_decoder_dims()` in `conv3d.py`.
The pattern for cached t_chunk=N:
- stage 0: `T_res = N + 2`, `T_tconv = N + 2`, `T_spatial = 2*N`
- stage 1: `T_res = 2N + 2`, etc.

---

## Step 2: Add the layer list and fixture

In `bruteforce_conv3d_sweep.py`, add a new layer list and test function at the bottom.
Ordering matters — **put the most compute-intensive layers first** so you get the biggest
wins early before any hangs occur.

Compute intensity ≈ `T × H_sweep × W_sweep` (larger = run first).

```python
_SWEEP_LAYERS_H4W8_720P_T16 = [
    # (name, C_in, C_out, kernel, stride, padding, T, H_sweep, W_sweep, h_factor, w_factor)
    ("up3_res", 96, 96, (3,3,3), (1,1,1), (0,0,0), 66, 186, 162, 4, 8),
    ...
]

@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {}]],   # 1x1 for per-device dims; (2,4) for actual 2x4
    ids=["bh_4x8_1x1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS_H4W8_720P_T16,
    ids=[l[0] for l in _SWEEP_LAYERS_H4W8_720P_T16],
)
def test_bruteforce_sweep_h4w8_720p_t16(
    mesh_device, mesh_shape, layer_name, ...
):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    output = f"sweep_results_h4w8_720p_t16/{layer_name}_{C_in}x{C_out}.json"
    run_sweep(device, C_in, C_out, kernel, T, H, W, output,
              stride=stride, padding=padding,
              h_factor=h_factor, w_factor=w_factor,
              max_combos=500,
              max_t_block=8,    # see hang mitigations below
              hw_product=32)
```

### Mesh choice

| Target config | Fixture mesh | Why |
|---------------|-------------|-----|
| BH 2x4 (480p) | `(2,4)` + `line_params` | Actual 8-chip mesh |
| BH 4x8 (720p) | `(1,1)` + `{}` | Run per-device dims on 1 chip |
| BH 4x32 (720p) | `(1,1)` + `{}` | Run per-device dims on 1 chip |

---

## Step 3: Add placeholder entries to conv3d.py

Add placeholder blocking entries BEFORE running the sweep so the sweep can seed
`best_us` from the table.  Use `(C_in_block, C_out_block, 1, 8, 4)` as a safe default.

Keys: `(h_factor, w_factor, C_in, C_out, kernel_size, T, H_out, W_out)`

---

## Step 4: Run the sweep

```bash
mkdir -p sweep_results_h4w8_720p_t16

pytest models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
    -k "bh_4x8_1x1" -s --timeout=0 \
    2>&1 | tee sweep_log_4x8_t16.txt
```

Monitor with a second terminal or the monitor script below.

### Hang mitigations (BH hardware — universal)

BH conv3d kernels hang on certain `(T_block, H_block, W_block)` combos that push
L1 past the device limit.  Two parameters in `run_sweep` control this:

- **`hw_product=32`** — only test combos where `H_block * W_block == 32`.
  This prevents hangs on all shapes tested so far and is the single most
  important mitigation.  Without it, h*w ∈ {4, 8, 16, 64, 128} all cause hangs.
- **`max_t_block`** — cap the maximum T_block tested.
  - BH 2x4 480p t=7: `max_t_block=7` (T=8+ hang for large spatial layers)
  - BH 4x8 720p t=16: `max_t_block=8` (T=9+ hang for C_in≥192)
  - Guideline: start with `max_t_block = t_chunk` (e.g. 16 for t_chunk=16),
    lower if you see hangs, confirm the threshold empirically.

If a layer hangs (log goes stale >90s):
1. `pkill -f "bruteforce_conv3d_sweep"`
2. `tt-smi -r 0,1,2,3,4,5,6,7`  ← always reset after a kill
3. Save the partial JSON manually (see Step 5)
4. Restart from the next layer with `-k "... and layer_name"`

---

## Step 5: Monitor and save partials

Use a stale-log watcher to detect hangs automatically:

```bash
prev_mtime=0; while true; do
  mtime=$(stat -c %Y sweep_log_4x8_t16.txt 2>/dev/null || echo 0)
  now=$(date +%s); age=$(( now - mtime ))
  if [ "$mtime" != "$prev_mtime" ]; then
    line=$(tail -1 sweep_log_4x8_t16.txt 2>/dev/null)
    case "$line" in *BEST*|*PASSED*|*FAILED*|*Done*|*Saved*) echo "$line" ;; esac
    prev_mtime=$mtime
  elif [ $age -gt 90 ]; then
    echo "STALE ${age}s: $(tail -1 sweep_log_4x8_t16.txt)"
  fi
  sleep 5
done
```

When a layer completes, the sweep writes `sweep_results_<tag>/<layer>_<Cin>x<Cout>.json`
with `"best_blocking": [cin, cout, t, h, w]` and `"best_us"`.

For killed/partial runs, save a JSON manually:

```python
import json
data = {
    "C_in": 192, "C_out": 192, "kernel": [3,3,3],
    "T": 66, "H": 94, "W": 82, "h_factor": 4, "w_factor": 8,
    "best_blocking": [96, 96, 8, 4, 8], "best_us": 12401,
    "note": "PARTIAL — hung at combo 46. T=8 confirmed best.",
    "top_20": [{"blocking": [96,96,8,4,8], "us": 12401, "status": "ok"}],
    "all_results": []
}
with open("sweep_results_h4w8_720p_t16/up2_res_192x192.json", "w") as f:
    json.dump(data, f, indent=2)
```

---

## Step 6: Read results and update conv3d.py

```python
import json, pathlib
for f in sorted(pathlib.Path("sweep_results_h4w8_720p_t16").glob("*.json")):
    d = json.loads(f.read_text())
    print(f"{f.stem}: {d['best_blocking']} = {round(d['best_us'])}us")
```

Update `models/tt_dit/utils/conv3d.py` — values only, never change keys:

```python
# Before
(4, 8, 96, 96, (3, 3, 3), 66, 184, 160): (96, 96, 1, 8, 4),  # up3_res — TODO
# After
(4, 8, 96, 96, (3, 3, 3), 66, 184, 160): (96, 96, 8, 4, 8),  # up3_res — 11980us
```

---

## Step 7: Run the perf test

The perf test hardcodes `num_inference_steps = 40`. Edit it to 1 for a quick check:

```bash
# Quick check (1 step — measures VAE decode directly)
# Edit test: num_inference_steps = 1
pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py \
    -k "bh_2x4sp1tp0 and resolution_480p and t2v" \
    -xvs --timeout=0

# Full run (40 steps — as CI runs it)
# Edit test: num_inference_steps = 40
pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py \
    -k "bh_2x4sp1tp0 and resolution_480p and t2v" \
    -xvs
```

**Report**: look for `VAE Decoding | Mean: X.XXs` in the output.

---

## T_block heuristics (empirically confirmed on BH)

| Config | t_chunk | T wins | Notes |
|--------|---------|--------|-------|
| BH 2x4 480p | 7 | T=7 (temporal), T=1-4 (spatial, tconv) | T=9+ hangs |
| BH 4x8 720p | 16 | T=8 (= t_chunk/2, large stages), T=2-4 (mid) | T=9+ hangs |
| BH 4x32 720p | 16 | T=3-4 | Much smaller W per device |

General rule: **T=t_chunk/2** tends to be the sweet spot for temporal conv3d
(kT=3) layers on large stages.  T=1 wins for tconv and spatial (kT=1) layers.

---

## Troubleshooting

**ARC core timeout / `Status: 0xffffffff`**
Device is in a bad state from a hard kill.  Always run `tt-smi -r 0,...,7` after
killing a sweep process before starting the next one.

**Hang at the same combo every time**
The L1 estimator is slightly underestimating for that shape.  Lower `max_t_block`
by 1-2 or accept the partial result — the sweep has already found the best safe combo.

**`best_blocking: null` in JSON**
The table seeded a better time than all sweep combos.  Keep the existing table value.

**Warmup >200s on first run after rebuild**
Normal — JIT compiles all new kernels.  Run twice; second run warmup should be ~40s.
