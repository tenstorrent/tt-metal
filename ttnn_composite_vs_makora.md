# TTNN composite ops vs. Makora fusion_store kernels

Mapping of the 19 AI-generated kernels in `/localdev/dnijemcevic/kernels/Tenstorrent/fusion_store/` to their corresponding TTNN implementations in this repo.

## Background: TTNN composite ops

A TTNN **composite op** is a high-level op implemented by orchestrating other (already-bound) TTNN ops, rather than dispatching to a dedicated device kernel. It has no program factory of its own — its `invoke()` method just chains `ttnn::add`, `ttnn::multiply`, `ttnn::where`, etc. Bindings go through helpers like `bind_unary_composite` / `bind_binary_composite`.

Source layout:
- Unary: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_composite.hpp` + `device/unary_composite_op.cpp`
- Binary: `ttnn/cpp/ttnn/operations/eltwise/binary/binary_composite.hpp` + `device/binary_composite_op.cpp`

Caveat: file-name "_composite" is a misnomer — both files contain a mix of true composites (chains of other ttnn:: ops) and primitives (single SFPU / fused kernel dispatches). Classification below was done by reading the actual `invoke` body.

## Binary kernels (5)

| Makora kernel | TTNN classification | TTNN impl path | Decomposes into | Test path |
|---|---|---|---|---|
| `atan2` | **Primitive** (`BinaryOpType::ATAN2`) | `binary_composite_op.cpp:153` | dedicated kernel `atan2_binary_tile` | `tests/ttnn/unit_tests/operations/eltwise/test_binary_composite.py` |
| `isclose` | **Composite** | `binary_composite_op.cpp:49` | `subtract`, `abs`, `multiply`, `add`, `le`, `where` (+ `isnan` for `equal_nan`) | `tests/ttnn/unit_tests/operations/eltwise/test_binary_composite.py` |
| `nextafter` | **Composite** | `binary_composite_op.cpp:29` | `where`, `gt`, `lt`, `add`, `subtract` | `tests/ttnn/unit_tests/operations/eltwise/test_nextafter.py` |
| `outer` | **Composite** | `binary_composite_op.cpp:546` | `reshape`, `to_layout`, `matmul` | `tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_outer.py` |
| `remainder` | **Primitive** (`BinaryOpType::REMAINDER`) | `binary_composite_op.cpp:449` | dedicated kernel | `tests/ttnn/unit_tests/operations/eltwise/test_remainder.py` |

## Unary kernels (8)

| Makora kernel | TTNN classification | TTNN impl path | Decomposes into | Test path |
|---|---|---|---|---|
| `digamma` | **Primitive** (SFPU `DIGAMMA`) | `unary.cpp:150` (`DEFINE_UNARY_OP`) | dedicated SFPU kernel | `tests/ttnn/unit_tests/operations/eltwise/test_math.py`, `test_composite.py` |
| `glu` | **Composite** | `unary_composite_op.cpp:338` | split, `sigmoid`, `multiply` | `test_activation.py`, `test_composite.py` |
| `lgamma` | **Primitive** (SFPU `LGAMMA`) | `unary.cpp:149` | dedicated SFPU kernel. `_lgamma` decomposition still exists at `unary_composite_op.cpp:32` but is now internal-only (called from `multigammaln`) | `test_composite.py` |
| `multigammaln` | **Composite** | `unary_composite_op.cpp:196` | calls `_lgamma` 4× with shifts + `add` | `test_math.py`, `test_composite.py` |
| `polygamma` | **Primitive-style** (single SFPU dispatch `POLYGAMMA`) | `unary_composite_op.cpp:421` | one SFPU op with Euler–Maclaurin correction; comment explicitly calls it "single kernel dispatch instead of 11+ composite ops" | `test_math.py`, `test_composite.py` |
| `reglu` | **Composite** | `unary_composite_op.cpp:351` | split, `relu`, `multiply` | `test_activation.py`, `test_composite.py` |
| `swiglu` | **Composite** | `unary_composite_op.cpp:378` | split, `swish`, `multiply` | `test_activation.py`, `test_composite.py` |
| `triu` | **Composite** | `unary_composite_op.cpp:405` | `index_triu`, `multiply` | `test_math.py`, `test_composite.py` |

## Eltwise + reduction kernels (6)

None of these patterns exist as a composite (or primitive) op in TTNN. The TTNN-equivalent is a Python chain of two separate ops.

| Makora kernel | Fused pattern | TTNN chain (no fused op exists) |
|---|---|---|
| `atan_mean_high_channel` | `mean(atan(x), dim=-1)` (high-channel layout) | `ttnn.atan` → `ttnn.mean` |
| `atan_mean_tall` | `mean(atan(x), dim=-1)` (tall) | `ttnn.atan` → `ttnn.mean` |
| `log_max_square` | `max(log(x²), dim=-1)` | `ttnn.multiply` → `ttnn.log` → `ttnn.max` |
| `sigmoid_min_high_channel` | `min(sigmoid(x), dim=-1)` via `-max(-sigmoid(x))` | `ttnn.sigmoid` → `ttnn.min` |
| `sigmoid_min_tall` | same | same |
| `sigmoid_min_wide` | same | same |

Reduction tests live under `tests/ttnn/unit_tests/operations/reduce/`.

## Categorisation of the comparison

Not all 19 entries in the Makora benchmark are an apples-to-apples comparison against a TTNN composite. Three buckets:

### A. Makora-fused-kernel vs. TTNN composite (true fusion comparison)
The fairest comparison: Makora's single fused kernel competes against an N-step `ttnn::*` chain.

`isclose`, `nextafter`, `outer`, `glu`, `reglu`, `swiglu`, `triu`, `multigammaln`

Total speedups (per Makora README): 0.55× (`outer`) up to 7.21× (`multigammaln`).

### B. Makora-fused-kernel vs. TTNN primitive (single-kernel comparison)
Makora's fusion is competing against an existing single-kernel implementation. Speedups here reflect a better single-kernel implementation, not fusion gain.

`atan2`, `remainder`, `digamma`, `lgamma`, `polygamma`

Total speedups: 1.47–3.03×.

### C. Makora-fused-kernel vs. two-kernel Python chain (no TTNN op exists)
TTNN has no composite for this pattern at all; the only baseline is a user-written `ttnn.X(); ttnn.Y()` chain.

All six `eltwise_reduction` kernels.

Total speedups (per Makora README): ≈0.74–0.95×. **To investigate:** none of these have been independently measured by this harness — the eltwise+reduction kernels are out of scope per the original instruction.

## Practical implications (refined later)

> *These were early observations based on Makora's published claims alone. The per-op attributions and bucket-B uplift timeline below supersede them — see "Per-op speedup attribution" for what actually drives each speedup.*

- For bucket A, the headline claim was: replacing the C++ decomposition with a fused kernel would be a win. Per-op attribution shows this is op-specific (multigammaln is algorithmic debt, isclose is genuine fusion, glu is slice elimination).
- For bucket B, the headline claim was: TTNN's existing primitives have headroom. Bucket-B uplift timeline shows the opposite — primitives were uplifted in March–April 2026 and now beat Makora.
- **To investigate:** whether bucket C (eltwise+reduction) patterns appear in real models often enough to justify fused TTNN ops. Out of scope for this investigation.

## Verification harness

Three Python scripts at the repo root (`/localdev/dnijemcevic/tt-metal/`):

| Script | Purpose |
|---|---|
| `verify_makora.py` | Headline harness. Runs any of 12 wired ops vs. matching TTNN op, reports per-shape + gmean speedup, PCC, max_abs_diff. Patches API drift in Makora kernels at module load (see below). |
| `verify_lgamma_uplift.py` | Per-op deep dive for `multigammaln`. Times `ttnn.lgamma` (SFPU prim), current `ttnn.multigammaln`, and a hand-rolled "uplifted" rewrite using `ttnn.lgamma`×4 + adds. Used to attribute the speedup to algorithm-vs-fusion. |
| `verify_isclose_decomposition.py` | Per-op deep dive for `isclose`. Times `ttnn.add` (single-prim baseline), `ttnn.isclose` composite, a hand-rolled Python chain mirroring the C++ composite, and Makora. |
| `verify_glu_decomposition.py` | Per-op deep dive for the `glu` family. Takes a CLI arg `glu|reglu|swiglu|all`. Times each of the 4 composite ops separately (slice / slice / activation / multiply), a hand-rolled chain, the `ttnn.*` composite, and Makora. |

`verify_makora.py` is the entry point. It runs a Makora kernel and the matching TTNN op on the same inputs, measures `DEVICE KERNEL DURATION [ns]` for each via the in-process profiler API, and reports four signals per shape:

- `ttnn` and `makora` — median device-kernel-duration in ns over `--iters` iterations (default 10) after `--warmup` warmups (default 2).
- `speedup` — `ttnn / makora` median ratio. Directly comparable to Makora's `shape→device_speedup` per-shape numbers.
- `pcc` — Pearson correlation coefficient between the two outputs (cast to fp32). Sanity check; mostly rounds to 1.0000 unless one implementation is producing wholly different values.
- `max_abs_diff` — the **per-element maximum absolute difference** between the two outputs (fp32). For elementwise ops this is the most informative numerical-correctness signal and the right column to read first. `0.00e+00` means bit-identical bf16 outputs; non-zero values quantify how far the two implementations disagree on at least one element. `nan` / `inf` indicate that random inputs landed outside the function's domain (e.g. negative inputs to `lgamma`/`multigammaln`, integer poles for `digamma`/`polygamma`); both implementations may agree on the singularity but produce different sign/magnitude infinities, which the diff exposes.

After all shapes finish, a `GMEAN over N shapes` row prints the geometric mean of the per-shape speedups. This matches Makora's `Device Speedup (GMean)` column. (Single-shape runs skip this row.)

### Scope: device-kernel duration only

**This investigation measures `DEVICE KERNEL DURATION [ns]` only — not wall-clock.** Compares against Makora's `device_speedups_by_shape` and `device_speedup_gmean` (which match the README's "Per-Shape" and "Device Speedup (GMean)" columns).

Each Makora op folder also has a `.json` with `total_speedups_by_shape` / `total_speedup` (wall-clock, host+device). Those numbers diverge sharply from the device-kernel ratios in some cases — e.g. `triu` at (4,1,384,4096) is 0.28× on device kernel (per Makora's own .json and confirmed by our measurement) but **38.92× on wall-clock** (per Makora's .json — not independently measured here). **To investigate:** the wall-clock gap likely reflects host-side overhead in TTNN's `index_triu`, but this hasn't been independently confirmed. Out of scope for the current harness; would need `time.perf_counter()` instrumentation.

### Profiler workflow it relies on

Three env vars must be set, otherwise `ttnn.get_latest_programs_perf_data()` returns `{}`:

```bash
export TT_METAL_DEVICE_PROFILER=1            # master switch
export TT_METAL_PROFILER_MID_RUN_DUMP=1      # flush before device close
export TT_METAL_PROFILER_CPP_POST_PROCESS=1  # populate program_analyses_results
```

In the loop the harness does:

```python
ttnn.synchronize_device(device)
ttnn.ReadDeviceProfiler(device)
perf = ttnn.get_latest_programs_perf_data()
duration_ns = perf[chip_id][0].program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration
```

Reference test that uses the same pattern: `tests/ttnn/profiling/test_get_perf_data.py`.

### Usage

```bash
python verify_makora.py --list                          # enumerate supported ops
python verify_makora.py multigammaln --readme-shapes    # all README shapes
python verify_makora.py isclose --shape 4 384 4096      # one shape
```

Wrap with `flock /tmp/tt-device.lock` to share the lock with `run_safe_pytest.sh`.

Sample output (truncated):

```
Op: isclose  iters=10  warmup=2
  isclose  shape=(32, 32)          ttnn=   28493 ns  makora=    5023 ns  speedup= 5.67x  pcc=1.0000  max_abs_diff=0.00e+00
  isclose  shape=(4, 384)          ttnn=   29853 ns  makora=    5197 ns  speedup= 5.74x  pcc=1.0000  max_abs_diff=0.00e+00
  isclose  shape=(4, 384, 4096)    ttnn= 1922425 ns  makora=  335741 ns  speedup= 5.73x  pcc=1.0009  max_abs_diff=0.00e+00
  isclose  shape=(4, 1, 384, 4096) ttnn= 1919475 ns  makora=  336488 ns  speedup= 5.70x  pcc=1.0009  max_abs_diff=0.00e+00
  isclose  GMEAN over 4 shapes:    ttnn=  236697 ns  makora=   41441 ns  speedup= 5.71x
```

### API-drift patches applied at module load

The kernels in `fusion_store/` were authored against an older tt-metal. To make them compile on current `main`, the harness rewrites three things in each kernel source string after `importlib` loads the module (kernel files on disk are untouched):

| Old API | Current API |
|---|---|
| `#include "compute_kernel_api/X"` | `#include "api/compute/X"` |
| `#include "dataflow_api.h"` | `#include "api/dataflow/dataflow_api.h"` |
| `where_fp32_tile(...)` | `where_tile<DataFormat::Float32>(...)` |
| `where_tile(...)` (untyped) | `where_tile<DataFormat::Float16_b>(...)` (bf16 dst) |
| `namespace NAMESPACE { void MAIN { ... } }` | `void kernel_main() { ... }` |

Implemented in `_patch_includes()` / `_load_makora_module()` in the harness.

## Verified results (measured 2026-05-08, current `main`)

`max_abs_diff` is the per-shape elementwise absolute error between Makora's output and TTNN's output, computed in fp32. PCC is included as a sanity check; for elementwise ops `max_abs_diff` is the more informative correctness signal.

### Composite-baseline ops (Makora claims hold)

For these ops the TTNN baseline is a multi-op decomposition (`bucket A` from above). Makora's published per-shape speedups match or are slightly exceeded by our measurements.

| Op | Shape | Makora claim | Measured | PCC | max_abs_diff |
|---|---|---|---|---|---|
| **multigammaln** | (32, 32)          | 5.79× | **6.23×** | 1.0000 | nan† |
|                  | (32, 128)         | 5.93× | **6.24×** | 1.0000 | nan† |
|                  | (5, 2240, 32)     | 3.78× | **3.95×** | 1.0000 | nan† |
| **isclose**      | (32, 32)          | 5.46× | **5.64×** | 1.0000 | 0.00e+00 |
|                  | (4, 384)          | 5.81× | **5.78×** | 1.0000 | 0.00e+00 |
|                  | (4, 384, 4096)    | 4.53× | **5.73×** | 1.0009 | 0.00e+00 |
|                  | (4, 1, 384, 4096) | 4.53× | **5.72×** | 1.0009 | 0.00e+00 |
| **glu**          | (32, 32, 32, 64)  | 3.02× | **3.18×** | 1.0000 | 0.00e+00 |
|                  | (3, 2, 32, 4096)  | 2.58× | **3.42×** | 1.0000 | 0.00e+00 |
| **reglu**        | (1, 1, 32, 64)    | 2.69× | **2.26×** | 1.0000 | 0.00e+00 |
|                  | (1, 1, 128, 512)  | 3.00× | **2.68×** | 1.0000 | 0.00e+00 |
|                  | (3, 2, 1024, 4096)| 3.10× | **3.12×** | 1.0007 | 0.00e+00 |
| **swiglu**       | (1, 1, 32, 64)    | 1.98× | **2.03×** | 1.0000 | 0.00e+00 |
|                  | (1, 1, 128, 512)  | 2.26× | **2.36×** | 1.0000 | 0.00e+00 |
|                  | (1, 1, 1024, 4096)| 2.76× | **2.76×** | 1.0002 | 0.00e+00 |
| **triu**         | (32, 32)          | 1.14× | **1.46×** | 1.0000 | 0.00e+00 |
|                  | (32, 64)          | 1.14× | **1.46×** | 1.0000 | 0.00e+00 |
|                  | (4, 384, 4096)    | 0.28× | **0.34×** | 1.0001 | 0.00e+00 |
|                  | (4, 1, 384, 4096) | 0.28× | **0.34×** | 1.0001 | 0.00e+00 |

†`multigammaln`: max_abs_diff is nan because random `randn` inputs include values outside the domain (x ≤ 0); both implementations correctly produce NaN on the same elements, so PCC remains 1.0.

### Primitive-baseline ops (Makora claims **inverted**)

For these ops the TTNN baseline is a single SFPU primitive (`bucket B` from above). Makora's published 2.8–6.3× speedups are **inverted in current main** — Makora kernels are 2–4× slower than the primitives. Numerical agreement is also degraded for several of them.

| Op | Shape | Makora claim | Measured | PCC | max_abs_diff |
|---|---|---|---|---|---|
| **atan2**     | (32, 32)          | 6.22× | **0.34×** | 1.0000 | 1.56e-02 |
|               | (4, 384)          | 6.60× | **0.35×** | 1.0000 | 1.56e-02 |
|               | (4, 384, 4096)    | 4.09× | **0.25×** | 0.9998 | 1.56e-02 |
|               | (4, 1, 384, 4096) | 4.09× | **0.25×** | 0.9998 | 1.56e-02 |
| **remainder** | (32, 32)          | 2.99× | **0.40×** | 1.0000 | 7.81e-03 |
|               | (4, 384)          | 2.99× | **0.33×** | 1.0000 | 7.81e-03 |
|               | (4, 384, 4096)    | 3.69× | **0.26×** | 1.0002 | **9.84e-01** ‡ |
|               | (4, 1, 384, 4096) | 3.69× | **0.27×** | 1.0002 | **9.84e-01** ‡ |
| **digamma**   | (32, 128)         | 7.35× | **0.48×** | 1.0000 | inf§ |
|               | (5, 2240, 32)     | 6.90× | **0.45×** | 1.0000 | inf§ |
|               | (3, 2, 32, 5600)  | 4.83× | **0.43×** | 1.0000 | inf§ |
| **lgamma**    | (32, 32)          | 4.30× | **0.45×** | 1.0000 | nan† |
|               | (32, 128)         | 4.40× | **0.45×** | 1.0000 | nan† |
|               | (5, 2240, 32)     | 2.89× | **0.43×** | 1.0000 | nan† |
| **polygamma** | (32, 32)          | 3.41× | **0.52×** | 1.0000 | inf§ |
|               | (32, 128)         | 3.60× | **0.52×** | 1.0000 | inf§ |
|               | (5, 2240, 32)     | 2.65× | **0.51×** | 1.0000 | inf§ |
|               | (3, 2, 32, 5600)  | 1.89× | **0.51×** | 1.0000 | inf§ |

‡ `remainder`: max_abs_diff ≈ 1 reflects a real algorithmic difference between TTNN's bit-shifted mantissa truncation and Makora's `div + floor + sub` chain on rounding-edge inputs. See bucket-B `remainder` row for details.
§ `digamma` / `polygamma`: random `randn` inputs include points where the function diverges (digamma is −∞ at x=0,−1,−2,…). For polygamma specifically, Makora's truncation-without-tail algorithm and TTNN's Euler-Maclaurin algorithm produce different sign/magnitude infinities at the poles — confirmed in bucket-B section.

### Geometric-mean comparison (matches Makora's `Device Speedup (GMean)`)

| Op | Bucket | Makora gmean | Measured gmean |
|---|---|---|---|
| multigammaln | A (composite) | 5.06× | **5.39×** |
| isclose      | A (composite) | 5.05× | **5.71×** |
| glu          | A (composite) | 2.79× | **3.30×** |
| reglu        | A (composite) | 2.93× | **2.67×** |
| swiglu       | A (composite) | 2.31× | **2.36×** |
| triu         | A (composite) | 0.57× | **0.71×** |
| **atan2**    | B (primitive) | 5.12× | **0.29×** |
| **remainder**| B (primitive) | 3.32× | **0.31×** |
| **digamma**  | B (primitive) | 6.26× | **0.45×** |
| **lgamma**   | B (primitive) | 3.80× | **0.44×** |
| **polygamma**| B (primitive) | 2.80× | **0.51×** |

## Notes & open questions

Confirmed:

- For composite-baseline ops (bucket A), measured per-shape speedups match Makora's claims to within ~10%, sometimes with our number slightly higher.
- For primitive-baseline ops (bucket B), Makora's claimed speedups are inverted in current main — this is explained by the March–April 2026 SFPU uplift wave (see bucket-B uplift timeline below).
- For `triu`, our device-kernel measurements agree with Makora's published 0.28×/0.28× at large shapes — Makora is genuinely slower on device kernel duration there. Wall-clock comparison is out of scope (see scope note).
- For `remainder`, the 9.84e-01 max_abs_diff reflects genuine algorithmic disagreement between TTNN's bit-shift truncation and Makora's `div + floor` (different rounding-edge behavior).

To investigate:

- Whether Makora's "TTNN baseline" at benchmark time was the legacy composite path (which the bucket-B uplift wave replaced) or a different chain. The git history confirms each TTNN primitive was uplifted between March–April 2026; we don't know which exact baseline Makora measured against. Confirming would require Makora's original benchmark harness or their benchmark date.
- Domain-restricted input distribution for `digamma` / `polygamma` to get clean max_abs_diff numbers (currently inf at the singularities).
- For `polygamma`, which implementation is correct at integer poles. Makora's algorithm has no tail correction; TTNN's has Euler-Maclaurin tail correction with documented "trigamma max ULP drops from ~108 to ~1." Likely TTNN is correct, but not directly verified against a reference.

## Layout / format support of Makora kernels

Background (from `tech_reports/tensor_layouts/tensor_layouts.md`): TTNN tensors have two orthogonal layout axes — **tensor layout** (row-major or tiled), **memory layout** (interleaved or sharded) — plus storage (DRAM/L1) and dtype.

Audit of all 19 Makora kernels' `host()` code shows a narrow operating envelope:

| Property | What Makora kernels support |
|---|---|
| Tensor layout | **TILE_LAYOUT only** (asserted in 18/19; `lgamma` auto-converts from row-major) |
| Memory layout (input) | Interleaved (assumed; works via `TensorAccessorArgs` but per-core work assignment ignores shard locality) |
| Memory layout (output) | Always **DRAM interleaved** — outputs go through `ttnn.allocate_tensor_on_device(...)` with default config; the input's memory config is never propagated |
| dtype | bf16 only: `atan2`, `nextafter`, `digamma`, `glu`, `reglu` (5). bf16/fp32: `isclose`, `outer`, `lgamma`, `multigammaln`, `polygamma`, `swiglu`, `triu` and all 6 eltwise+reduction (13). Forced fp32: `remainder` (1). |
| Shape constraints | Op-specific — `glu`/`reglu` need rank=4 dim=-1; `swiglu`/`sigmoid_min_*`/`log_max_square` need rank≥2; `outer` needs rank≤4. All assume tile-aligned shapes. |

Two operational consequences:

1. **The speedup measurements are scoped to "interleaved DRAM TILE_LAYOUT" inputs.** Outside that envelope (sharded activations, L1-resident tensors, row-major intermediates), the Makora kernel can't accept the input — you'd fall back to the TTNN composite.
2. **No kernel propagates the input's memory config to the output.** A sharded input would silently produce a DRAM-interleaved output, breaking pipelines that expect locality.

The TTNN composites being replaced *do* handle the full layout matrix because they're built on hardened `ttnn::*` primitives. None of the Makora kernels currently expose layout-aware dispatch. **To investigate:** whether a productionised version would need a dispatcher (input → fused kernel vs. composite fallback) or another mechanism — TTNN ops with multiple program factories use the dispatcher pattern, but the right approach for these ops hasn't been evaluated.

## Per-op speedup attribution

The composite-A speedups have different root causes. Five ops attributed so far (one finding per op family):

### `multigammaln` — speedup is tech debt, not fusion

- TTNN's `multigammaln` (`unary_composite_op.cpp:196`) calls `_lgamma` (`unary_composite_op.cpp:32`) 4× — a **Lanczos 6-term polynomial** decomposed into ~35 `ttnn::*` calls per lgamma → ~150 programs total.
- TTNN's user-facing `ttnn.lgamma` is a **single SFPU Stirling primitive** (`unary.cpp:149`, `DEFINE_UNARY_OP(lgamma, LGAMMA)`; kernels at `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/lgamma_kernel.cpp` and `lgamma_fast_kernel.cpp`).
- Makora (`/localdev/dnijemcevic/kernels/Tenstorrent/fusion_store/unary/multigammaln/multigammaln.py`) inlines the **same Lanczos** algorithm fused into one kernel — they did *not* use the SFPU Stirling primitive. Identical coefficients to `_lgamma` (`76.18009172947146`, `-86.50532032941677`, …, `0.918938531357171`).
- Hand-rolled "uplifted" rewrite using `ttnn.lgamma` × 4 + adds (9 separate programs, no fusion) measured at **5.6× faster than current composite** — within noise of Makora's **5.4×**.
- Source comment confirms tech debt: `// TODO: Remove this once the multigammaln is uplifted` (`unary_composite_op.cpp:30`).
- **Conclusion: ~all of the speedup is recoverable by routing `multigammaln` to `ttnn.lgamma`. Fusion contributes ≈0× on top.** Script: `verify_lgamma_uplift.py`.

### `isclose` — speedup is real fusion, no tech debt

- TTNN's `isclose` (`binary_composite_op.cpp:49`) is **11 `ttnn::*` ops** (`isnan`, `where`, `isnan`, `where`, `subtract`, `abs`, `abs`, `multiply`, `add`, `le`, `where`).
- Hand-rolled Python chain (in `verify_isclose_decomposition.py`, `_manual_isclose`) calling the same 11 ops measures **identical** to the C++ composite (ratio 1.00–1.02× across all shapes) → no hidden cost in the C++ implementation.
- No SFPU `isclose_tile` primitive exists; no algorithmic alternative.
- Makora (`/localdev/dnijemcevic/kernels/Tenstorrent/fusion_store/binary/isclose/isclose.py`) uses the **same algorithm** as the composite, just fused in one kernel via SFPU tile primitives (`isnan_tile`, `where_tile`, `sub_binary_tile`, `abs_tile`, `mul_unary_tile`, `add_unary_tile`, `unary_le_tile`).
- Cost ratios vs. a single `ttnn.add` on same shape: composite is **8.5–12.8×** of `add` (matches the 11-program chain, with bandwidth saturation at large shapes); Makora is **1.5–2.2×** of `add` (≈ bandwidth floor for a kernel doing ~11 SFPU ops/tile).
- comp/Makora ratio is rock-stable at **5.71×** across 4 orders of magnitude of tensor size.
- **Conclusion: ~all of the speedup is from fusion. The only TTNN-side fix is to write a fused kernel.** Script: `verify_isclose_decomposition.py`.

### `glu` / `reglu` / `swiglu` — speedup is from slice elimination, not activation fusion

- All three composites (`unary_composite_op.cpp:338` / `:351` / `:378`) share the same shape: 2× `ttnn::slice` (via `split_tensor_for_glu` at `:147`) + activation + `ttnn::multiply`. Activations: glu→`sigmoid(ACCURATE)`, reglu→`relu`, swiglu→`swish`.
- Hand-rolled Python chains in `verify_glu_decomposition.py` match the C++ composites to <1% across all shapes for all three ops — no hidden cost.
- **Slices are not free metadata** — each `ttnn::slice` on the last dim is a real DRAM copy of half the input. The 2 slices contribute **24–49% of composite cost** across all measured shapes (highest fraction for ops with the cheapest activations).
- Speedup ordering glu (3.2–3.4×) > reglu (2.3–3.1×) > swiglu (2.0–2.7×) is fully explained by activation cost: swish ≈ 2× sigmoid/relu, so the slice-as-fraction is smaller and the win is correspondingly smaller.
- Last-dim tile-aligned slices are still copies in TTNN: tiles in DRAM are stored row-major-by-tile-row, so left-half tiles are interleaved with right-half tiles → no contiguous view possible.
- Makora (`/localdev/dnijemcevic/kernels/Tenstorrent/fusion_store/unary/{glu,reglu,swiglu}/`) avoids slicing entirely: its **reader kernel directly reads tile pairs `(t, t + Wt/2)` from the input into separate circular buffers**. The "split" is implicit in tile-coordinate arithmetic, costing nothing.
- TTNN has no general-purpose "fuse a slice into a reader" abstraction. Specialised ops do stride-aware reads (SDPA, conv2d, sharded matmul), but eltwise ops universally read whole tensors.
- **Conclusion: most of the speedup is slice elimination via stride-aware reading, not activation+multiply fusion. The only TTNN-side fix is a custom fused kernel — there's no expressive escape hatch in the current op model.** Script: `verify_glu_decomposition.py [glu|reglu|swiglu|all]`.

### `atan2` — primitive vs. primitive (concrete bucket-B example)

This op is bucket B (primitive-baseline). We measured Makora 0.29× device gmean vs. their claimed 5.12× — a clean inversion. Investigation explains why:

- TTNN's `ttnn.atan2` (`binary_composite_op.cpp:153` → `BinaryOpType::ATAN2`) is a **tuned single-pass SFPU primitive** at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atan2.h` (copyright 2026 — recent). One SFPU pass per tile: range-reduce to [0,1] via `vec_min_max + reciprocal`, then a minimax polynomial (3-term bf16 / 7-term fp32) with branchless special-case handling.
- Makora's `atan2.py` is also one fused program but **a different algorithm**: `recip(x) → mul(y, recip) → atan() + 5 mask ops + 6 fixup mul/add/sub + 3 fill/where for x==0`. ~17 SFPU `_tile` calls per tile.
- Both end up as one device program (same dispatch model). The difference is purely algorithmic — TTNN computes atan2 directly with hardware-aware branchless logic; Makora computes `atan(y/x)` and patches up quadrants from masks.
- `max_abs_diff = 1.56e-2` is consistent with two different polynomial fits at bf16 (≈ 0.5% of atan2's [-π, π] range) — not a numerical bug.
- TTNN's atan2 SFPU did not exist before April 2026 (see bucket-B uplift timeline). The pre-April baseline was a "composite/fallback" per the uplift commit. Against current `ttnn.atan2`, our measurement flips.
- **To investigate:** what exact baseline Makora benchmarked against. Hypothesis is the pre-April-2026 composite, but their benchmark harness/date hasn't been confirmed.
- The same shape of explanation **is confirmed** for the other 4 inverted bucket-B ops:

| Op | TTNN SFPU algorithm | Makora's algorithm | Note |
|---|---|---|---|
| `digamma` | Piecewise rational `P(x)/Q(x)`, 53-coeff LUT, 2 segments. (2026.) | Manual `recip + log + Bernoulli series` chain. | |
| `polygamma` | Exact sum k=0..10 + Euler-Maclaurin tail correction (B₂, B₄, B₆). (2026.) | Same exact sum **without tail correction** — hard truncation at k=10. | Makora is less accurate, not just slower. Source of the `inf` `max_abs_diff` near integer poles. |
| `lgamma` | Stirling approximation via `lgamma_stirling_tile` / `lgamma_adjusted_tile`. (2026.) | Lanczos polynomial chain (same coefficients as the legacy `_lgamma`). | Same Stirling-vs-Lanczos pattern as `multigammaln`. |
| `remainder` | Bit-shifted mantissa truncation (`exexp` + `shft`) in one SFPU pass. (2023.) | `div_binary_tile + floor_tile + sub_binary_tile` chain. | Source of the `9.84e-01` `max_abs_diff` — `div+floor` accumulates error the bit-shift approach avoids. Two valid algorithms that disagree on rounding edges. |

All five bucket-B SFPU headers are recent (mostly 2026 copyright), strong evidence the primitives post-date Makora's benchmark. In every case both Makora and TTNN run as one device program — the difference is purely algorithmic, with TTNN choosing a tighter hardware-aware algorithm. **No TTNN-side fix needed for any bucket-B op.**

#### Bucket-B uplift timeline (from `git log --follow`)

The five primitives were created or uplifted in a tight **March–April 2026** window. Makora's benchmarks pre-date this wave.

| Op | Pre-uplift state | Date | Commit / PR |
|---|---|---|---|
| `lgamma` (bf16) | composite Lanczos chain | **2026-03-03..06** | `41c953b1482` / PR #38748 |
| `lgamma` (fp32) | same | **2026-03-24** | `22ad2b3975a` / PR #39646 |
| `digamma` (Stirling) | composite `recip + log + Bernoulli` (≈Makora's algorithm) | **2026-03-31** | `3772313280a` |
| `digamma` (LUT P/Q) | Stirling (8 days old) | **2026-04-08** | `a9198941d96` / PR #39399 |
| `polygamma` | **composite k=0..10 truncation, no tail correction** (≈Makora's algorithm exactly) | **2026-04-03** | `716493289f5` |
| `atan2` | composite/fallback | **2026-04-07** | `aed2421dfe6` / PR #41251 |
| `remainder` | composite binary (since Aug 2024) | **2026-04-09** | `ed8a0f787f9` |

Notable specifics:

- `polygamma` is the strongest case: 6 months ago TTNN's polygamma was the **exact same algorithm Makora ships now** (k=0..10 truncation). The April 3 uplift added the Euler-Maclaurin Bernoulli tail correction — the comment "trigamma max ULP drops from ~108 to ~1" comes from this commit. So Makora wins against pre-uplift TTNN, loses against post-uplift TTNN, and is also numerically inferior at integer poles.
- `digamma` was rewritten twice in 8 days (composite → Stirling SFPU → LUT-based rational). The current LUT design is only ~5 weeks old.
- `lgamma`'s uplift in early March is what created the dormant `_lgamma` Lanczos in `unary_composite_op.cpp:32`. It still serves `multigammaln` only — the `// TODO: Remove once the multigammaln is uplifted` comment dates from this transition.
- `atan2` did not exist as an SFPU primitive before April 2026.

The `multigammaln` story (composite still using the Lanczos `_lgamma`, recoverable with a 10-line rewrite to call `ttnn.lgamma`) sits in the same shape as these — a remaining pre-uplift composite. The bucket-B uplift wave just hasn't reached it yet.

### Generalised takeaway

Four ops, four different attribution patterns:

- **multigammaln** (bucket A) — *algorithm* in the composite is wrong (Lanczos chain when an SFPU Stirling kernel exists). Fix at the algorithmic level; no kernel work. Fusion ≈ 0×.
- **isclose** (bucket A) — algorithm + primitives are right; the *dispatch model* is wrong (11 programs for ops that fit in one fused kernel). Fix requires a fused kernel. Fusion ≈ entire speedup.
- **glu** (bucket A) — algorithm + dispatch are reasonable; the *helper primitive* (`ttnn::slice`) has a high constant cost and TTNN has no zero-copy view. Fix requires either a slice-fusing kernel (Makora's approach) or a strided-read primitive in TTNN's op model.
- **atan2** (bucket B) — TTNN already has a tuned single-pass SFPU primitive that beats Makora's manual `atan(y/x) + fixup` chain. Makora's claimed speedup was against an older/composite TTNN baseline. No TTNN-side fix needed.

**Observation:** the "Makora is faster because they fuse" framing is a partial story across the four attributed ops. For multigammaln the cause is algorithmic (composite uses Lanczos when an SFPU Stirling primitive exists). For glu it's slice cost. For atan2 it's a recent SFPU uplift. Only isclose fits the pure-fusion narrative.

**To investigate:** `triu` — the remaining unattributed composite-A win. Device-kernel curve is non-monotonic (Makora 1.46× small / 0.34× large). The .json shows 38.92× wall-clock at the largest shape, which suggests TTNN's `index_triu` may have an O(volume) host-side mask construction, but this hasn't been independently verified. The bucket-A multigammaln/isclose/glu attribution methodology applied to `triu` would resolve this.

## Status / next steps

Verified end-to-end (11 of 12 wired ops):
- ✅ Composite-baseline (claims hold): `multigammaln`, `isclose`, `glu`, `triu`, `reglu`, `swiglu`.
- ⚠️ Primitive-baseline (claims inverted): `atan2`, `remainder`, `digamma`, `lgamma`, `polygamma`.

Outstanding:
- `nextafter` — kernel host code uses `runtime_args=[[[] for _ in range(grid.y)] for _ in range(grid.x)]` (older 2D-list-of-empties shape); current API expects flat `list[tuple[CoreCoord, VectorUInt32]]`. Needs Python-side monkey-patch of `ttnn.KernelDescriptor`, not a C++ source rewrite.
- `outer` — different-shape inputs, needs `_make_inputs` extension.
- 6 eltwise+reduction kernels — out of scope for this checkpoint.
