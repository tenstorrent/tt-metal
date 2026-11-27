# Verification Report: g6_pad

Board: Blackhole, 110-core compute grid (`compute_with_storage_grid_size`). Op: pure data movement. There is no compute kernel and the gate is bit equality.

## Code Review

### Fixed in this pass
1. **The test directory did not collect.** `test_g6_pad_extra_cases.py` imported `_check_call` / `_make_input` / `_raw_rows`, which no longer exist in `test_g6_pad.py`. `test_g6_pad_knobs.py` monkeypatched `BAND_NOC0_SHARE` / `_PLAN_CACHE` / `_DESCRIPTOR_CACHE`, also gone. `test_g6_pad_profile.py` used `g6.last_plan()` (a function call on a dict attribute that doesn't exist on the package) and descriptor-module knobs that no longer exist. All three were leftovers from an earlier attempt.
   - `test_g6_pad_extra_cases.py` was rewritten on `run_case`. It keeps the 5 branch-coverage cases: case 69 with row/col remainder ≥ 16, row-only patches, and a faces-2/3-only band with a non-zero `c`.
   - `test_g6_pad_knobs.py` was rewritten against `g6_pad.KNOBS`. It now covers every non-default knob value: `band_lanes=1`, `band_block_tiles=2`, `min_units_per_core=32`, `block_tiles ∈ {1,3}` with `input_depth_blocks=2`, and `block_tiles=2` with depth 1, on 4 shapes. **This is the only coverage of the multi-block / depth-2 ring-wrap path**, because the default plan is one block per core on every INPUTS shape. All 24 pass.
   - `test_g6_pad_profile.py` was deleted; it was redundant with `test_g6_pad_perf_ab.py`, which gained p1 (fp32).
   - `zone_summary.py` filtered on a `G6PAD_` zone prefix the kernels don't use. It now keeps every kernel zone except firmware zones.
2. **DRY: the L1 cap restated the buffer depth as a literal.** `make_plan` computed `block_tiles_cap = (budget − tile) // (2 · tile)` while the CB is sized `input_depth_blocks · block_tiles`. The cap now divides by the `input_depth_blocks` knob, so a depth change lands in one place.
3. **SUPPORTED drift (`xpass_drift` = 1, then fixed).** The `sharded` refusal loose case is tagged `memory_config="l1"` by `axes.py`, since a height-sharded input lives in L1. `validate()` correctly raised `ValueError` for it, which the runner counted as an XPASS. A probe that widened SUPPORTED in memory ran every golden TILE cell then gated out through the golden `run_g6_pad`: **17/17 passed**. That covered fp32 extension ×10, p1 fp32 `leading`/`tile_aligned`, uint32 rank-2 `w` ×3 and the bf16 L1 cases ×3. int32 TILE has no golden cell; it passes in the new `test_g6_pad_dtypes.py`, on full-range values with poisoned padding, in both regimes. The TILE kernels are `elem_bytes`-generic and address through `TensorAccessor`, so SUPPORTED now honestly claims:
   - dtype {bf16, fp32, int32, uint32}, memory {dram, l1};
   - alignment {tile_aligned, w_non_aligned, h_non_aligned}, rank {2, 3, 4}, padded_dims {leading, h, w, hw}.
   - ROW_MAJOR stays out because its regime is not built.
4. **The L1 ledger was out of date.** The 32-bit cap read 163 where the code computes 164, the cap expression hardcoded the depth, and the page-format column read "bf16 Phase 0". All three are corrected.

### Reviewed and found conformant (no change)
- **Kernels.**
  - `void kernel_main()`, `api/dataflow/dataflow_api.h` and `TensorAccessor` everywhere; there is no `InterleavedAddrGen`.
  - CB sync balances: the reader reserves and pushes the nominal `block_tiles` per block, and the writer waits and pops `block_tiles` per block. A ragged last block or fill-only slots are simply not written, so every ring cycle closes at capacity.
  - Barriers: one read barrier per block; `writes_flushed` before each pop; `noc_async_write_barrier` at the end of every writing kernel (BB-03), including both band lanes.
- **Helpers.** `fill_l1_range` is used for the fill page and the sub-face-row column fragments. There is no compute phase, so no compute helper applies. The `mcast` helper is not applicable because there is no reuse-shared operand.
- **Blocking-model fidelity.**
  - `min_units_per_core`, `band_lanes`, `band_block_tiles`, `block_tiles` and `input_depth_blocks` are each defined once (`KNOBS`), carried into the plan, keyed in the descriptor cache, and passed to the kernels as CT args.
  - No CB scales with a whole-op dimension. `cb_input_tiles` is `depth · block_tiles`, clamped by the BB-05 cap.
  - Expression: reader, writer and band writer all issue a whole block of NoC transactions per synchronization point (one barrier or handshake per block, not per tile).
  - The half-turned-split check does not apply: the per-core run is one block, and the block is the whole per-core range.
- **Design conformance.**
  - Algorithm, regimes, the RISC/NoC ownership (reader NCRISC/NoC0, writer BRISC/NoC1, band split over both RISCs) and the flat `row_wise=True` splits all match `op_design.md`.
  - Grid fill: p2–p4 run on 110 cores and p1 on 110. `[1,3844,1536]` uses 48 cores (48 band tiles). `[1,3844,12]` uses 1 core because only one unit of work exists.
  - The extension output is the `ttnn.reshape(x, shape, padded_shape)` view. The acceptance and golden tests confirm under graph capture that it adds zero programs and keeps the buffer address.
- **Prompt `## Rules` (MUST rules).**
  - Bit-exact with no format detour: there is no unpack/pack.
  - Input tile padding never leaks: verified with poisoned-padding cases in both regimes and all dtypes.
  - Extension touches only the last tile row: `untouched_rows` cases pass.
  - new_storage writes each page once and never reads fill-only pages: the reader skips those slots.
  - One program per call; no host transforms; no OOM gated by EXCLUSIONS (EXCLUSIONS is empty).
  - BB-05: L1 is bounded by the cap. BB-07: the descriptor cache is keyed on structure plus knobs, and `fill_bits` and addresses are RT-patched; the loose `same_shape_two_values` case passes. BB-08: `g6_pad.last_plan`. BB-17: core ranges come from `num_cores_to_corerangeset(row_wise)`, at most 2 rectangles. BB-21: zones exist per block operation.

### Advisories (not fixed: they cost more than they buy at this stage)
- **RT-arg ABI duplicated as literals.** The host's `RT_ADDR=0` / `RT_FILL_BITS=3` and the kernels' `get_arg_val(0)` / `(3)` repeat the same indices. This is an ABI, not a block knob, so DRY drift is low risk, but a refinement that adds RT args must touch both sides.
- **Patch barrier drains the fill writes too.** In `patch_block` (new_storage writer), the barrier that makes local NoC patches land before the slot is sourced also drains the fill-only writes issued in `write_fill_tiles`. On patched shapes this reduces their overlap. A transaction-id-scoped wait would avoid it. It is p1-irrelevant, since p1 is tile-aligned with no patches.
- **BB-21 role dump is implicit.** Every core of a regime runs the same kernel set, and `last_plan` reports regime, `ncores` and block extents rather than a per-core role table.

## Registry Conformance
- `INPUT_TAGGERS` (6 taggers, all `(inputs, axes)`), `SUPPORTED` (6 axes, the same names as TARGET), `EXCLUSIONS = []` and `validate()` are present and correctly wired. `validate()` runs argument checks (`ValueError`, including sharded inputs), then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). It is the first line of `g6_pad()`.
- The op file does **not** declare `INVALID`.
- The SUPPORTED auto-fix is described above (Code Review item 3).
- **INVALID audit** (`feature_spec.py`): `INVALID = []`. That is well-formed: TARGET holds no block-float format, so the canonical bf8b + ROW_MAJOR entry does not apply. There are no cross-tensor entries, since there is a single input tensor, and no weight canonicalization, since this is not a norm op.
- **Tagger consistency**: `axes.py` reported one declared/captured mismatch (`dram` vs `l1`) on the `sharded` refusal case. That is the harness tagging a height-sharded L1 input, not an op tagger bug.

## L1 Ledger Audit
- **Currency**: the 3 CB rows (`cb_fill_tile` idx 0, `cb_input_tiles` idx 1, `cb_fill_tile_lane1` idx 2) match the descriptors. The size expressions match after the cap and format fixes above.
- **Capacity vs live set**: `cb_input_tiles` reserves fill-only slots that are never written. The ledger justifies this as the nominal-push ring closure, bounded by `block_tiles`. `cb_fill_tile` is 1 page with a live set of 1 that streams every axis. There are no over- or under-sized buffers.
- **Page format vs DEST**: there is no DEST, and every page is a dataflow-only payload in the input dtype. Audit 3 does not apply.
- **Disjoint lifetimes**: `cb_fill_tile` vs `cb_input_tiles` is concurrent (fill writes are issued while the reader fills slots), with the justification recorded. The two lane fill pages are concurrent on separate RISCs, also justified.
- **Bounds**: `block_tiles ≤ block_tiles_cap`, with the cap closed-form in `tile_bytes` and the depth knob. The total `tile_bytes · (1 + depth · block_tiles)` is ≤ 1,347,776 B by construction. The INPUTS maximum is ~61 KB (p1).
- **Traffic budget**: present and consistent with the implementation.
  - extension: 0 B read; only the band rows are written.
  - new_storage: input read 1× and output written 1×; fill-only tiles are never read.
  - There is no cheaper split (no dependent or reuse-shared axis), and the deferred `new_storage_rows` regime is deferred for scope, not traffic.
- **Block-size defaults**: held (interleaved: full-grid spread, coarsest block = the whole per-core range). Departures exist only as knobs, now correctness-covered.
- **Footprint**: `L1 = tile_bytes · (1 + input_depth_blocks · block_tiles)` for new_storage_tiles and `band_lanes · tile_bytes` for extension_band. Only `cb_input_tiles` scales, and only with `block_tiles` × depth.

## Precision Baseline
From `test_g6_pad_precision_baseline.py`. The input tile padding is poisoned with 1000. PCC via `comp_pcc`, max abs via `comp_allclose`, rel-RMS = RMS(err)/std(ref).

| Shape | Dtype / regime | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio p5/p50/p95 |
|-------|----------------|-----|-------------|--------------|------------------|---------|---------------------------|
| (1,1,2,62080) → 32 rows | bf16 / extension | 1.0 | 0 | 0 | 0 | 0 | 1/1/1 |
| (1,3844,1536) → 3872, c = 0.1 | bf16 / extension | 1.0 | 0 | 0 | 0 | 0 | 1/1/1 |
| (1,1,196,196) → 256², c = −2^127 | bf16 / new_storage (row + col patch) | 1.0 | 0 | 0 | 0 | 0 | 1/1/1 |
| (5,12,128,128) → (8,…) p1 | fp32 / new_storage | 1.0 | 0 | 0 | 0 | 0 | 1/1/1 |

**Assessment**: bit-exact on every measured shape and on every golden cell, including fp32 ±inf, −0 and subnormals, int32/uint32 range ends, and the `-2^127` and inexact `0.1` fills. There is no rounding anywhere, and the ratio distribution is a point mass at 1, so no scale bug exists.
**Recommended tolerances**: exact (`helpers.TOLERANCES` = `("exact", None)` for every dtype). Defaults hold.

## Verifier CLI Summary
Default runner (`eval/eval_test_runner.sh`, `EVAL_CAPTURE_L1=1`), saved in `verifier/verifier_report.json`:
- supported_pass: 74
- xfail_expected: 7 (all ROW_MAJOR: cases 260, 262, 263, 264, 265, 267 and loose special_values/262, all queued in Refinement 1)
- invalid_skipped: 0
- supported_fail: 4. **All are a harness artifact, not an op failure.** They are the four `refusal: "args"` loose cases: low ≠ 0, high < 0, 5 pairs on rank 4, and uint32 `value = 2**32`. The op raises `ValueError` in `validate()` as required and the test body returns normally. `eval/l1_profiling.py` then rejects the test because the graph-captured L1 peak is 0, which is always true for a refusal on a DRAM input since nothing is allocated. (The `sharded` refusal passes only because its sharded input occupies L1.) With `EVAL_CAPTURE_L1=0` (`verifier/verifier_report_no_l1_capture.json`) all 4 pass, giving **supported_pass 78, supported_fail 0**. This cannot be fixed inside the op without allocating L1 for no reason, so it needs a harness fix: `capture_for_item` should accept a zero peak when the graded call refused on the host.
- xpass_drift: 0 (was 1 before the SUPPORTED fix)
- xfail_wrong_mode: 0
- no_axes_found: 3 (`test_regression.py` regression tests, all passed)

## Performance snapshot (device kernel duration, `--profile`, 110-core BH)
| Case | Regime | Cores | Device ns |
|------|--------|-------|-----------|
| p1 `[5,12,128,128]` fp32 dim0 5→8 | new_storage_tiles | 110 | 28,327 (goal < 29,910; roofline 22,190) |
| p2 `[1,1,2,62080]` → 32 rows | extension_band | 110 | 11,894 |
| p3 `[1,1,8,62080]` → 16 rows | extension_band | 110 | 5,050 |
| p4 `[1,1,8,62080]` → 32 rows | extension_band | 110 | 10,189 |
| `[1,3844,1536]` → 3872 | extension_band | 48 | 1,981 |
| `[1,196,128]` → 256 | new_storage_tiles | 32 | 2,007 |
| `[1,1,196,196]` → 256² | new_storage_tiles | 64 | 3,007 |

## Recommendations
- **Queue**: TARGET − SUPPORTED is now `{layout: ROW_MAJOR}` only, and Refinement 1 (`new_storage_rows`) covers all of it. Every other TARGET value is in SUPPORTED with passing cells. INVALID is empty, so nothing is omitted. The perf-1 contract (p1) was already supported at Phase 0, so no generality refinement was needed to unlock it. After Refinement 1 the queue is all perf, and both perf slots target p1, the only `attention: PERF FOCUS` loose case.
- **p2–p4 headroom (not queued: not perf-flagged)**: these are the spec's other "tuned for" shapes. They run at ~290–310 GB/s of band writes, made of 2–3 small segments (≤ 1 KB) per tile, so they are likely NoC write-issue bound rather than DRAM bound. Candidate levers are the knobs already exposed (`band_lanes`, `min_units_per_core`; the implementer measured 32/64 as slower) and `bank_stagger` on the band-tile walk. They belong in the guard set of every p1 perf refinement. If a later loose case flags them, file a perf phase.
- **Ring-wrap/multi-block coverage**: keep `test_g6_pad_knobs.py` in the non-regression set of the p1 perf refinements, which will turn exactly those knobs.
- **Harness**: report the `l1_profiling` zero-peak rejection of host-side refusals to the eval owners. Until it is fixed, `supported_fail` reads 4 on a correct op.
