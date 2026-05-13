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
| Memory layout (input) | **DRAM interleaved assumed.** No kernel asserts or branches on the input's memory config. Inputs are accessed via `TensorAccessor` (constructed from `TensorAccessorArgs(input).get_compile_time_args()`), which transparently resolves both interleaved and sharded layouts — so a sharded input would still run *correctly*. But the per-core work split assigns tiles by global linear tile-id (`core_idx * base_per_core + …`), assuming uniform NoC access cost from every core to every tile — the property of DRAM-interleaved layouts. With an L1-sharded input, each core would read most of its assigned tiles across the NoC from other cores' L1, **defeating sharding's locality benefit**. |
| Memory layout (output) | **Always DRAM interleaved.** All kernels call `ttnn.allocate_tensor_on_device(shape, dtype, layout, device)` with no `memory_config` argument. The C++ binding (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:298`) defaults to `MemoryConfig{}`, which the constructor at `tt_metal/api/tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp:38` documents as `// Interleaved DRAM`. The input's memory config is never propagated. |
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

## Apples-to-apples vs agent-generated kernels

Two bucket-A ops were regenerated end-to-end by the incremental pipeline
(`run_op.py` → planner → implementer → verifier) and then benchmarked against
Makora's hand-written kernel via `verify_makora.py`. Both compared at
device-kernel duration (same harness as the rest of this doc).

### `multigammaln_lanczos` (agent) vs `multigammaln` (Makora)

Setup: both kernels run at fp32. Makora's `multigammaln` host wrapper accepts
fp32 transparently, so no patches required on either side. Agent's `glu`-style
file naming (`multigammaln_lanczos`) reflects that the algorithm is the
Lanczos 6-term polynomial — same as the (nuked) TTNN composite and the same
as Makora's kernel.

Measured 2026-05-12 (Phase 0 baseline, then Refinement 1 — DST reuse):

| Shape | Phase 0 speedup | After R1 | PCC | max_abs_diff |
|---|---|---|---|---|
| (1, 1, 32, 32) | 1.04× | 1.02× | 1.0000 | 6.7e-2 |
| (1, 1, 32, 128) | 1.04× | 1.06× | 1.0000 | 6.7e-2 |
| (1, 5, 2240, 32) | 1.04× | 1.06× | 1.0000 | 6.8e-2 |
| **GMEAN** | **1.04×** | **1.04×** | | |

Phase 0 baseline: ttnn 163 965 ns, makora 157 856 ns gmean.
After R1: ttnn 162 486 ns, makora 157 842 ns gmean.

Speedup ratio is identical across all three shapes, indicating the gap is per-tile compute structural overhead, not dispatch or DRAM. `max_abs_diff` is the bf16-noise floor of two different algebraic regroupings of the same Lanczos formula.

**Agent's design choices** (independent of Makora):
- Same Lanczos coefficients and pole-zeroing strategy.
- Algebraic simplification: fuses `+0.918938` and `−5.5` into one compile-time constant `4.581...`. Makora doesn't do this — re-derives `x+4.5` from the input CB.
- Phase 0 used a single `cb_accumulator` CB with round-trip between iterations (vs Makora's pure-DST accumulator). 6 `tile_regs_acquire` cycles per output tile vs Makora's 1.
- Refinement 1 collapsed those 6 cycles to 1 while keeping the accumulator in D0 — matches Makora's structure. A register-pressure trick the agent had to invent (spill D1, run the multiply, reload D1 from CB) is documented in the kernel but is not in Makora's code.
- Multi-core distribution via `split_work_to_cores` over the full Tensix grid — same as Makora.

### `glu_fused` (agent) vs `glu` (Makora)

Setup: both kernels run at bf16. Makora's `glu` is bf16-locked at host/CB/kernel level. Agent's Phase 0 was fp32-only; relaxed the dtype validator to accept bf16 and added a dtype-aware compute config (bf16 → LoFi + `fp32_dest_acc_en=False` + default unpack; fp32 retains HiFi4 + `fp32_dest_acc_en=True` + UnpackToDestFp32). Tracked as the op's Refinement 2.

Measured 2026-05-12:

| Shape | Speedup | PCC | max_abs_diff |
|---|---|---|---|
| (32, 32, 32, 64) | 1.02× | 1.0000 | 0.00e+00 |
| (3, 2, 32, 4096) | 1.06× | 1.0000 | 0.00e+00 |
| **GMEAN** | **1.04×** | | |

GMEAN: ttnn 30 782 ns, makora 29 655 ns. **Outputs are bit-identical to Makora's at bf16.**

**Agent's design choices** (independent of Makora):
- Reader computes tile-pair indices `(t, t + Wt_half)` from a single input tensor's tile-id arithmetic — the reader-level split trick the audit doc identifies as Makora's win against the TTNN composite. The agent invented this independently from the constraint "no `ttnn::slice` allowed in the implementation".
- Two reads issued concurrently on NoC0, single `noc_async_read_barrier` per output tile — same as Makora.
- Compute body is a 4-step `sfpu_chain`: `Load A`, `Load B`, `Sigmoid<Approx::Exact>`, `SfpuMul`. Approx::Exact matches the ACCURATE sigmoid mode of the TTNN composite.
- 3 CBs (input_a, input_b, output), 2 pages each, double-buffered — same as Makora.
- Multi-core distribution via `split_work_to_cores` over the full Tensix grid — same as Makora. Tile-to-core mapping is mathematically equivalent (output tiles split with the +1-on-early-cores remainder).

### `atan_mean` (agent) vs `atan_mean_tall` + `atan_mean_high_channel` (Makora) — bucket C, first

Setup: both kernels run at bf16. Makora ships **two** shape-specialised kernels (`atan_mean_tall` for `(1,1,H,W)` with H≥1024 and W≤64, `atan_mean_high_channel` for `(N,C,H,W)` with N or C in the hundreds and H,W small). The agent's prompt required **one** kernel covering both shape regimes — see `eval/prompts/atan_mean.txt`. Phase 0 was fp32-only; Refinement 2 added bf16/bf8 dtype support (agent kept `fp32_dest_acc_en=True` even for bf16 — a deliberate precision-favoring design choice, documented in `capabilities.md`).

Measured 2026-05-13 (post-R2):

| Variant | Shape | Speedup (ttnn/makora) | PCC | max_abs_diff |
|---|---|---|---|---|
| tall | (1, 1, 2048, 64) | 1.07× | 1.0000 | 1.95e-3 |
| tall | (1, 1, 1024, 64) | 1.04× | 1.0000 | 1.95e-3 |
| tall | (1, 1, 2048, 32) | 1.07× | 1.0000 | 1.95e-3 |
| tall | (1, 1, 1024, 32) | 1.02× | 1.0000 | 1.95e-3 |
| **tall GMEAN** | | **1.05×** | | |
| high_channel | (1, 256, 64, 64) | 1.04× | 1.0000 | 2.93e-3 |
| high_channel | (256, 1, 64, 64) | 1.06× | 1.0000 | 2.93e-3 |
| high_channel | (1, 128, 128, 128) | 1.05× | 1.0000 | 2.93e-3 |
| high_channel | (128, 1, 128, 128) | 1.09× | 1.0000 | 2.93e-3 |
| **high_channel GMEAN** | | **1.06×** | | |

The 5–6% per-shape gap is entirely the agent's `fp32_dest_acc_en=True` retention at bf16 vs Makora's `fp32_dest_acc_en=False`. Pre-R2, when the same `atan_mean.py` was patched manually to set `fp32_dest_acc_en=False` for bf16 inputs, the same shape set measured at **1.00× tall / 0.99× high_channel** — parity. So the perf gap is a documented precision-vs-perf trade-off the agent chose to make, not a structural inefficiency.

**Agent's design choices** (independent of Makora):
- **One kernel, both regimes.** The agent's compute kernel uses `compute_kernel_lib::sfpu_atan` + `compute_kernel_lib::reduce<AVG, REDUCE_ROW>` — both helpers parametrised on `Wt` at compile-time, no shape-specialised branches. Distribution via `split_work_to_cores` adapts naturally to either regime (few output tiles in tall, hundreds in high_channel).
- **Helper-based reduce path.** Agent routes through `kernel_lib::reduce` which dispatches AVG+REDUCE_ROW to the matmul-based path (col-0 scaler). Makora hand-rolls the legacy `reduce_init/reduce_tile<SUM, REDUCE_ROW>` path with a row-0 scaler. Both produce correct output at bf16; the agent's path is the one the LLK team is migrating new code onto.
- **Buffer-the-row reduction.** `cb_atan_tiles` is sized to `Wt` pages — `sfpu_atan` pushes `Wt` post-atan tiles, then `reduce<>` consumes them. Matches Makora's `atan_mean_tall` structure (single `tile_regs_acquire` per row-tile). The chunked structure Makora used in `atan_mean_high_channel` (with explicit accumulator CB and per-chunk DST acquire/release) is *not* present in the agent's kernel — the agent picked the simpler structure and it scales fine on high_channel shapes because Wt stays small there too.
- **Precision-favoring compute config.** `fp32_dest_acc_en=True` for **all** input dtypes, including bf16/bf8. Documented in `capabilities.md`. Trades ~5% perf at bf16 for keeping the reduce-stage accumulation in fp32.

### Net assessment

Three agent-generated kernels measured against Makora's hand-written kernels:

| Op | Bucket | Agent gmean / Makora gmean | Note |
|---|---|---|---|
| multigammaln_lanczos | A | 1.04× | After R1 (DST reuse). Phase 0 was ~1.25×. |
| glu_fused | A | 1.04× | After R2 (dtype-aware compute config to match Makora's perf-favoring bf16 settings). |
| atan_mean (both Makora shape sets) | C | 1.05–1.06× | After R2 (bf16 enablement, precision-favoring config). Same kernel covers both Makora variants. |

Across both buckets and all three ops, the agent lands within 4–6% of Makora's hand-written kernels at the kernel's chosen design point. Structural design choices — multi-core distribution, NoC pattern, CB topology, compute-body algorithm — match Makora's. Per-op specifics:

- **multigammaln_lanczos**: Phase 0 implementer used a per-iteration accumulator CB (6 DST cycles per tile), about 25% off Makora's pure-DST design. A single-line refinement prompt ("don't round-trip through cb_accumulator") was enough for the implementer to refactor to Makora-equivalent structure (1 DST cycle per tile). Algebraic regrouping and one register-pressure trick are agent-original.
- **glu_fused**: Phase 0 implementer invented the tile-pair reader directly from the no-slice constraint. The 4% residual gap is in the noise band; functional outputs are bit-identical at bf16. No structural refinement needed; only a dtype-aware compute config to actually realize the perf at bf16.
- **atan_mean**: First bucket-C measurement. Phase 0 implementer chose the matmul-based reduce path via the helper library (correct for current LLK), wrote one kernel for both Makora shape regimes, and produced 30/30 tests passing on first compile. R2 (bf16 support) added a parametrized dtype test suite (20/20). Agent elected to keep fp32 accumulation at bf16 — a documented precision-first design point that costs ~5% vs Makora's perf-first config. Pre-R2 manual override to match Makora's compute config closed the gap to parity, confirming the kernel structure itself is on par. **Bucket-C insight:** Makora's two shape-specialised kernels (tall / high_channel) had no structural reason to be split — they perform near-identically on each other's shapes, and the agent's single parameterised kernel matches them both within ±2%.

## Methodology notes (working scratchpad)

Tacit findings from running multigammaln_lanczos and glu_fused end-to-end —
not yet integrated into the formal methodology, kept here as a checkpoint for
the next exploration round (bucket C / eltwise+reduction).

### Prompt design

- **Bucket A prompts have a clear "paste the TTNN composite" template** —
  multigammaln_lanczos and glu_fused both used it. The agent is told to
  translate a named algorithm with explicit prohibitions on which ttnn::* /
  SFPU primitives it can call from the entry point.
- **Bucket C has no TTNN composite to translate from.** Eltwise+reduction
  patterns (atan_mean_*, log_max_square, sigmoid_min_*) only have a
  user-written ttnn.X() ; ttnn.Y() chain as their baseline. The prompt
  structure must shift: describe the math against a PyTorch reference
  directly, list the building-block primitives the agent may use
  (atan, sigmoid, mean, max, …), and let the agent design the fusion from
  scratch. There is no "this composite, but fused" instruction to give.
- **Minimal prompts beat heavy prompts on refinements.** A 70-line
  refinement spec that pre-solved the DST-budget allocation problem
  produced an inferior result to a one-line "Try not to round-trip through
  cb_accumulator" + non-regression rule. Heavy spec defeats the eval — it
  measures transcription, not engineering judgement. Default to minimal;
  add detail only if the agent visibly diverges.
- **The `# golden: <op_name>` header drives the pipeline's output
  directory.** If the prompt's import path uses a different name (e.g.
  `# golden: glu` but `from ttnn.operations.glu_fused import glu_fused`),
  the pipeline writes to the wrong dir and the acceptance test breaks.
  Always keep `# golden:` + import path + signature name consistent.

### Compute config

- **Phase 0 "max precision" settings (HiFi4 + fp32_dest_acc + UnpackToDestFp32)
  are correct for fp32 inputs but pure overhead at bf16.** The fp32 settings
  on a bf16 input ran ~1.3× slower than the bf16-tuned settings (LoFi +
  fp32_dest_acc_en=False + default unpack) with **bit-identical output**.
  The 33% gap was config, not kernel structure.
- **Refinement 2 (bf16 support) is more than a validator change.** It must
  also dispatch a dtype-aware compute config in the program descriptor.
  Without that, bf16 "works" but at the wrong precision regime.
- **Math fidelity has no effect on SFPU-only kernels.** Multiply via
  `SfpuMul` doesn't use the FPU, so HiFi4 vs LoFi changes nothing for those
  ops. The setting still affects FPU ops (matmul, reduce, conv).

### Kernel structure findings

- **The "tile-pair reader" trick (`a_tile_idx = ...; b_tile_idx = a + W_half`)
  was invented independently by the agent** from the constraint "no
  `ttnn::slice` allowed". The TTNN op model has no general abstraction for
  reader-level slicing; the agent worked it out from first principles. This
  trick is reusable for any "operate on halves of a single input" pattern.
- **DST register reuse across iterations is the high-leverage refinement
  for chained-accumulation kernels.** The Phase 0 multigammaln_lanczos
  kernel issued 6 `tile_regs_acquire/release` cycles per output element
  (one per lgamma sub-evaluation + finalize). Collapsing to 1 cycle
  (accumulator stays in D0 across iterations) closed most of the gap to
  Makora. The agent invented a "spill D1 to compute (a-0.5)*log(a+4.5),
  reload from CB" register-pressure trick during this refinement —
  documented in the kernel comments.

### Pipeline + harness gotchas

- **`run_safe_pytest.sh` acquires `/tmp/tt-device.lock` on its own fd 9.**
  Wrapping it with an outer `flock /tmp/tt-device.lock` causes
  self-deadlock — the inner flock waits forever for a lock the outer
  process already holds on a different fd. `verify_makora.py` does NOT
  flock internally, so wrap *it* with flock when sharing the device.
- **Two nuke patterns coexist.** Standalone ops (e.g.
  `ttnn/cpp/ttnn/operations/multigammaln_lanczos/`) → handled by the
  `/nuke-op` skill's directory discovery. Inline composite functions
  inside shared `*_composite_op.cpp` files (glu, isclose, multigammaln
  forward) → require manual surgery: remove the function definition,
  the header declaration, the nanobind binding, and any Python golden
  function registration. The skill won't find these. Plan separately.
- **JIT cache must be cleared with `rm -rf built/tt-metal-cache*` after
  source edits.** Per the `TT_METAL_CACHE` env var in `.bashrc`. The cache
  is content-hashed so modified sources get a new dir, but stale cached
  entries waste disk and can confuse subsequent runs.
- **The `# golden:` header → pipeline output dir mapping** (already noted
  above under prompt design). Repeating because it cost us a full pipeline
  launch once.

### Eltwise+reduction prep (when picking this up next)

Makora's bucket C kernels live at
`/localdev/dnijemcevic/kernels/Tenstorrent/fusion_store/eltwise_reduction/`.
Six kernels:
- `atan_mean_high_channel`, `atan_mean_tall`
- `log_max_square`
- `sigmoid_min_high_channel`, `sigmoid_min_tall`, `sigmoid_min_wide`

Per-op published speedups from Makora's README are **0.74–0.95×** — Makora
is *slower* than the naive `ttnn.atan(); ttnn.mean()` Python chain at large
shapes. Two open questions for this category:

1. Is there a kernel-level reason Makora is slower (specific implementation
   choice), or is the bare ttnn-op chain just well-optimised for these
   patterns?
2. Would a from-scratch agent-generated fused kernel land closer to the
   ttnn chain (matching its perf) or closer to Makora (worse)?

Prompt template for the first eltwise+reduction op would be different from
multigammaln_lanczos.txt and glu_fused.txt — no composite to paste, just a
PyTorch reference and the list of allowed primitives (SFPU activations +
the reduce helper family at `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_*.hpp`).

## Status / next steps

Verified end-to-end (11 of 12 wired ops):
- ✅ Composite-baseline (claims hold): `multigammaln`, `isclose`, `glu`, `triu`, `reglu`, `swiglu`.
- ⚠️ Primitive-baseline (claims inverted): `atan2`, `remainder`, `digamma`, `lgamma`, `polygamma`.

Outstanding:
- `nextafter` — kernel host code uses `runtime_args=[[[] for _ in range(grid.y)] for _ in range(grid.x)]` (older 2D-list-of-empties shape); current API expects flat `list[tuple[CoreCoord, VectorUInt32]]`. Needs Python-side monkey-patch of `ttnn.KernelDescriptor`, not a C++ source rewrite.
- `outer` — different-shape inputs, needs `_make_inputs` extension.
- 6 eltwise+reduction kernels — out of scope for this checkpoint.
