# Verification Report: conv2d_nhwc

Board: Blackhole p150b (110-core grid). Golden results: `/tmp/conv2d_nhwc_golden_r3/`. A compact copy of the final `verifier_report.json` (summary, category counts, loud categories) is next to this file; the full report is in the results dir.

## Code Review

### Fixed

1. **9 SUPPORTED cells failed: "input window exceeds L1" (supported_fail, `validation`).**
   - Cells: translated tt-forge fp32 shapes with wide rows, e.g. 128×128×512→512, 512×512×256→256, 256×256×480→64.
   - Cause: R0 always fetched whole input rows. One tile-row's window was `(d(k−1)+1) · W · C_in · 4 B`, e.g. 786 KiB for 128×512.
   - Fix: built the design's deferred R10 **column clip** in the reader and in the host mirror `_window_bounds`:
     - A block inside one output row gets a 2-D window, `rows × [col_lo, col_hi]`.
     - A multi-row block gets a contiguous flattened range. Its first and last input rows are trimmed to the columns read, but only when that row is not padding-clamped. A clamped row can be shared with the next output row.
   - The gather index generalizes to `(row − row_lo)·win_w + (col − col_lo)`.
   - Side benefit: less halo traffic (ρ) on every wide-row shape.
2. **Two more cells failed after `strided` was claimed.**
   - Cells: SDXL 32×32×1280→1280 s2, fp32. A one-tile-row window is 5 rows × 32 cols × 5 KiB, about 800 KiB.
   - Fix: `select_blocking` now walks the 768 KiB `CB_BUDGET_BYTES` first, treating it as a footprint *preference* (perf lamp L5). It falls back to a hard cap, `l1_cap_bytes() = get_max_worker_l1_unreserved_size() − 128 KiB`, before refusing.
   - The walk now caches `window_max(bm)` instead of recomputing it per bn.
3. **DRY: compute kernel hard-coded the CB indices** (`constexpr cb_act_rm = 1`, …), while reader and writer took them as CT args. They are now CT args 0–6 of compute, from the one table in the descriptor module.
4. **DRY: the DEST config was stated twice.** `dst_tiles = 4` in `select_blocking` and `fp32_dest_acc_en=True` / `HiFi4` in the descriptor were independent literals, as was `T_acc = 4096`. They now all derive from `MATH_FIDELITY` / `FP32_DEST_ACC_EN`: `DST_TILES`, `ACC_DTYPE`, the partials page format and the compute config.
5. **Weight tile bytes were derived from the activation's `elem_bytes`.** That is wrong as soon as `weight_dtype != dtype`: a bf8b tile is 1088 B. `select_blocking` now takes `weight_tile_bytes = tile_bytes(weight_tensor.dtype)`. It feeds the footprint and the weight/bias CB page sizes.
6. **Writer: one `noc_async_write_barrier` per output tile-row.** It now uses `noc_async_writes_flushed()` before the `cb_pop_front` (the source slots are safe to recycle) and one `noc_async_write_barrier()` at kernel end.
7. **`validate()` did not check the prepared weight against the call.** A weight prepared for a different C_in / kernel_size / groups would silently produce garbage. It now raises `ValueError` when `weight.shape[-2] != K_t·32`.

### SUPPORTED grown (verifier probe)

- **`stride_mode += "strided"`.** The gather (`h_base = ho·s − p`) and the window bounds were already stride-parametric. A probe over 8 strided fp32 shapes passed with zero kernel changes (PCC ≥ 0.999995): k1/k3/k5, s2/s3, odd spatial, N=2. All golden strided cells inside SUPPORTED then passed.
- The acceptance test's "rejects stride" check now targets dilation, which is still unsupported. Added `test_conv2d_nhwc_strided` (4 shapes) and `test_conv2d_nhwc_wide_rows_column_clip` (2 shapes, including W_out=200, which is not a tile multiple).

### Reviewed, no change needed

- Kernel style is current: `void kernel_main()`, `api/dataflow/dataflow_api.h`, `TensorAccessor` everywhere, CB push and wait counts balance on every CB, and the alias pair `cb_partials` / `cb_biased` closes per block.
- Helpers:
  - `tilize`, `matmul_block` (packer L1 acc, `caller_owns_pack_target`, `TilizePreKBlock` as `PreKBlockFn`), `add_bias_bcast_rows` (RowBroadcast, in place via the alias) and `untilize`.
  - The raw `llk_pack_reconfig_l1_acc(0)` is the design's documented gap in the matmul helper.
  - Bias broadcast: row 0 of the prepared bias tile, RowBroadcast. Correct, and no redundant fill.
- Design conformance: algorithm, phase/RISC ownership, the 2-D m×n split over the whole grid (`g_n > 1` only when the m split leaves cores idle) and the CB set all match `op_design.md`.
- Blocking-model fidelity:
  - bm, bk, bn and the three depths are single-source module/CT knobs. Every CB size and trip count derives from them.
  - bm = `core_m_tiles` on all but large-window shapes. bk is the largest divisor that fits.
  - No CB scales unconditionally with a whole-op dimension.
- Expression:
  - The reader gather and the tilize hand off one tile-row × bk at a time through a depth-2 `cb_act_rm`. This is the design's planned streaming window over m (ledger: `cb_act_rm` streams m), not a collapsed block.
  - The tilize output (`cb_act_tiles`), the matmul and the bias run on whole bm×bk / bm×bn blocks.
  - Coarsening the gather handshake is a perf lever (Refinements 3 and 6), not a defect.
- Prompt `## Rules`: the prompt has no `## Rules` section. The one-dispatch rule holds: one `ttnn.generic_op` per call.

### Deferred (architectural)

- DRAM output-row slicing (rest of R10), for windows that exceed even the hard cap. No golden shape reaches it after the column clip.
- Interleaved → in-op HEIGHT/BLOCK placement (R7, technique 7). This is a perf regime with no flagged target. See Recommendations.

## Registry Conformance

- `INPUT_TAGGERS`: 7 taggers, all `(inputs, axes)`, in the specified order. `memory_layout`, `layout` and `activation` read slots 7/8/9 when present.
- `SUPPORTED`: covers every gated axis, including `weight_dtype`, `bias_mode` and `activation`.
- `EXCLUSIONS`: empty. It is still correct, because SUPPORTED is single-dtype, so no mixed-precision cell can occur.
- `validate()`:
  - Reconstructs INPUTS, runs the taggers, then overrides `memory_layout` / `layout` / `activation` with the real values.
  - Checks SUPPORTED per axis, then EXCLUSIONS (`UnsupportedAxisValue` / `ExcludedCell`).
  - Structural refusals raise `ValueError`.
  - The entry point calls it first.
  - `prepare_*` gate through `_validate_prepare` on the axes prepare can see.
- The op file does **not** declare `INVALID`.
- No drift fixes were needed beyond `strided`. There was no XPASS, because validate refused strided before the probe.
- **INVALID audit** (`eval/golden_tests/conv2d_nhwc/feature_spec.py`): one entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}`.
  - It is the canonical bf8b + RM entry, and single-tensor (both axes describe the activation).
  - It is structural: a bf8b RM tensor cannot be built.
  - No cross-tensor entries. Not a norm-like op, so no weight-canonicalization cells apply.
  - Well formed.

## L1 Ledger Audit

- **Currency**: all 8 CBs have rows, and the sizes match the code. The ledger was updated for the column-clipped `window_sticks_max` bound and the two-tier budget (`CB_BUDGET_BYTES` preferred, `l1_cap_bytes()` hard).
- **Capacity vs live set**:
  - `cb_act_rm`, `cb_weight_tiles` and `cb_out_rm` exceed their live set by exactly their documented depth of 2.
  - `cb_act_tiles` (bm·bk), `cb_partials`/`cb_biased` (bm·bn) and `cb_bias_tiles` (core_n_tiles) are fully live.
  - The window spans m (affine in bm) and is reused across n and k.
  - No over- or under-sizing found.
- **Page format vs DEST**: `fp32_dest_acc_en` is on, and `cb_partials` / `cb_biased` / `cb_act_tiles` are Float32. Correct. `cb_act_rm`, `cb_act_window` and `cb_out_rm` carry input or output data, so their format is set by the tensor ABI, not by DEST. No finding.
- **Disjoint-lifetime pairs**: every row has a concrete sharing / no-sharing reason. The only alias is partials ≡ biased.
- **Bounds**: every symbol in the closed form has a bound and an establishing predicate. Nothing is unbounded: the window is bounded by the budget check, and the op raises rather than over-allocating.
- **Block-size defaults**: the interleaved default held. The work units are spread over the full grid first; then the coarsest bm, and the largest bk ≥ the floor, that fit. Departures from it come only from the fit check, never from a settled solve.
- **Per-core footprint** `F = window_sticks_max(bm)·tap_width·eb + 2·bk·T + bm·bk·T + 2·bk·bn·T_w + bm·bn·T_acc + has_bias·core_n_tiles·T_w + 2·bn·T`.
  - The window scales with bm (and with the clipped width). The weights scale with bk·bn. The partials scale with bm·bn.
  - Measured device peak over the golden run: 32 KiB – 1.27 MiB. Only SDXL 32×32×1280 s2 at fp32 exceeds the 768 KiB preference.
- **Data-movement budget**: present and consistent with R0. Input `g_n·ρ` crossings, weights `g_m·nb_m`, output 1. The cheapest-traffic split (resident window + halo + weight multicast) is a `deferred` regime row with its delta and a positive reason: it layers onto R0's unchanged CB set and arrives with the sharded refinements.
- **Ledger findings folded into the queue**: the footprint-vs-`achievable_l1_peak_bytes` targets are folded into Refinements 3, 6 and 9. There is no standalone ledger refinement.

## Precision Baseline

`tests/ttnn/unit_tests/operations/conv2d_nhwc/test_conv2d_nhwc_precision_baseline.py`: fp32, bias on, weights scaled by 1/√(C_in·k²).

| Shape (NHWC → C_out, k, p) | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP p50 / p99 | got/true ratio p5 / p50 / p95 |
|---|---|---|---|---|---|---|
| (1,8,8,32) → 64, k3 p1 | 1.0000000 | 6.8e-3 | 1.20e-3 | 1.23e-3 | 1.4e4 / 3.8e5 | 0.9949 / 0.9990 / 1.0029 |
| (1,28,28,64) → 64, k3 p1 | 0.9999994 | 8.5e-3 | 1.28e-3 | 1.14e-3 | 1.3e4 / 4.6e5 | 0.9948 / 0.9991 / 1.0030 |
| (1,14,14,256) → 512, k3 p0 | 0.9999998 | 8.5e-3 | 1.29e-3 | 1.18e-3 | 1.4e4 / 5.3e5 | 0.9945 / 0.9991 / 1.0031 |
| (1,112,112,32) → 64, k7 p3 | 1.0000000 | 9.8e-3 | 1.26e-3 | 1.19e-3 | 1.3e4 / 5.2e5 | 0.9946 / 0.9991 / 1.0032 |
| (1,128,128,512) → 128, k3 p1 (column clip) | 0.9999970 | 1.1e-2 | 1.29e-3 | 1.11e-3 | 1.3e4 / 4.8e5 | 0.9949 / 0.9991 / 1.0029 |

**Assessment**: the error is uniform across shapes and K depths (rel-RMS ≈ 1.1–1.2e-3), so it does not grow with K.
- The ratio is a broad cluster around 0.999. That is not a scale bug: the spread is ±0.3 %, not a tight non-1.0 constant.
- The 0.1 % downward median is the expected tf32 truncation of both matmul operands. The FPU reads SrcA/SrcB as tf32, and fast tilize truncates fp32 → tf32, by the `tilize_helpers.hpp` contract. This adds a −2⁻¹¹-per-operand bias.
- fp32 ULP error is therefore ~1e4 at the median. "fp32" here means tf32-input / fp32-accumulate.

**Recommended tolerances (fp32)**: PCC ≥ 0.9999, rel-RMS ≤ 5e-3, atol ≈ 2e-2 at unit-variance outputs. Golden uses PCC 0.999 / rel-RMS 0.02.

## Verifier CLI Summary (final run)

- supported_pass: 172 (Phase 0 as handed over: 133)
- xfail_expected: 2262
- invalid_skipped: 1104
- no_axes_found: 2174 (translated tests that carry no axes: 2145 xfail, 16 regression passes, rest skipped)
- supported_fail: 0 (initial run: 9)
- xpass_drift: 0
- xfail_wrong_mode: 0

**xfail_expected by blocking axis value** (a cell can count under several values):

| Axis value | Cells |
|---|---|
| dtype = bf16 | 1185 |
| weight_dtype = bf8b | 814 |
| weight_dtype = bf16 | 803 |
| c_in_unaligned | 273 |
| depthwise | 253 |
| both_unaligned | 240 |
| HEIGHT_SHARDED | 224 |
| dilated | 217 |
| TILE | 186 |
| grouped | 169 |
| BLOCK_SHARDED | 151 |
| c_out_unaligned | 86 |
| WIDTH_SHARDED | 78 |
| dtype = bf8b | 64 |
| relu | 58 |

Every one of these values is assigned to a refinement in `op_requirements.md`. Nothing in TARGET − SUPPORTED is omitted.

**Device timings (fp32, golden run)**, for orientation only: no perf target is fp32.
- 56×56×256→256 k3: 970 µs on 98 cores.
- 32×32×1280→1280 s2: 2.18 ms.
- 512×512×256→256 (tt-forge): 79 ms.
- The cost is dominated by the per-stick DRAM window reads, the per-(pixel, band) L1→L1 gather copies, and HiFi4 fp32 matmul.

## Recommendations

- **Queue shape.** There are 6 generality and 3 perf refinements, more than the usual 3–5. TARGET adds 13 axis values across 7 axes, and 4 of them are scheme-changes that must stand alone: BLOCK/WIDTH K-exchange, HEIGHT halo + weight multicast, depthwise compute, and TILE untilize-on-shard. Everything that was a knob-turn is bundled.
- **Order.** Refinements 1 and 2 land the perf-1 contract in full: BLOCK + bf16 × bf8b + bias for the flagged SDXL level-1 3×3. After that the order is hardest first, subject to the true dependencies. HEIGHT precedes the ResNet HEIGHT perf pass and the TILE refinement; TILE needs the sharded paths.
- **Fidelity policy (Refinement 2).** The public signature has no `compute_kernel_config`, so the op must choose fidelity from the dtypes. The two references disagree for the same dtype pair (SDXL HiFi2, ResNet LoFi). Measure PCC at LoFi first.
- **Interleaved R7 (technique 7).** Auto block-shard, runtime HEIGHT and kernel-stride fold are measured 10–25× wins on interleaved inputs. No flagged perf case is interleaved, so this is not a queue entry. Once Refinements 1 and 5 exist, R7 is only a window-fill swap and is worth doing as follow-up perf.
- **L1 pressure.** The fp32 SDXL 32×32×1280 s2 cell sits at 1.26 MiB, under the hard cap. Any fp32 shape with `5 · W · C_in · 4 B` above about 1.2 MiB per tile-row would need the remaining R10 output-row slicing. bf16 halves this.
- **Perf lamps to check first in the perf passes:**
  - L4: the writer drains output before prefetching the next block's weights.
  - L6: the window DRAM reads all start on the same bank (`bank_stagger`).
  - The gather issues one L1→L1 NoC read per (pixel, band) with a barrier per tile-row.
