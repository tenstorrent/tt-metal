# Verification Report: tilize

Board: Blackhole p150b (110 compute cores, 8 DRAM banks). Results dir: `/tmp/tilize_verify/golden3` (trimmed `verifier_report.json` — summary + loud-category node ids — in this directory).

## Code Review

Fixed in place:
- **SUPPORTED drift (under-claimed) — 4 axes.** Probed bit-exact on device: `low_l1=True` (cap `LOW_L1_BLOCK_WIDTH_TILES=8`, ragged tail included, footprint O(1)), all four interleaved `buffer` transitions (TensorAccessor is buffer-type agnostic), `rank` 2/3/5/6 (leading-dim fold), and `tile_height` 16/8/4/2/1 (CB tile dims + the helper's `tilize_block` fallback). Added to SUPPORTED. Golden supported_pass 30 → 238.
- **Tiny tiles were unreachable from the entry point**: `tilize()` allocated the output without the requested `tile=`. Now allocates via `ttnn.TensorSpec(..., tile=out_tile)`.
- **validate() ordering bug (5 `xfail_wrong_mode`)**: the "H not a multiple of the output tile height without padding → ValueError" rule fired for TILE (retile) inputs, so `test_tilize_retile[... 1x1x16x128 ...]` got ValueError instead of a support refusal. A TILE input already carries its own tile padding; the rule is now scoped to ROW_MAJOR inputs and TILE inputs fall through to the SUPPORTED gate.
- **Runtime shard-side tagging**: `_side_dict` read `nd_shard_spec` first, but a live tensor's `memory_config()` fills both `shard_spec` and `nd_shard_spec` for a legacy shard, so every legacy call tagged `nd`/`ND_SHARDED`. Now prefers the 2D scheme when the layout is HEIGHT/WIDTH/BLOCK with a `shard_spec`. legacy vs nd is not recoverable at runtime (ND specs expressible in 2D are canonicalized), so Refinement 2 must land both together (noted there).
- **L1 ledger**: total footprint omitted the reader staging CB, and `stick_bytes` / `staging_tile_rows` / `num_banks` were missing from the symbol table. Added (see L1 Ledger Audit).
- Tests added: tiny-tile, buffer × low_l1 (ragged column tail), rank tests in `test_tilize.py`; `test_tilize_precision_baseline.py`.

Reviewed, no change needed:
- Registry wiring, `validate()` first line of the entry point, one `generic_op` dispatch per call, no layout-converting ttnn ops.
- Kernels: `void kernel_main()`, `api/dataflow/dataflow_api.h`, TensorAccessor (no InterleavedAddrGen). CB sync balanced: reader pushes `width` per tile-row = helper waits `width`; helper pushes `width` = writer pops `width`; tail column block on its own CB pair (17/1) so each ring has one quantum.
- Compute uses `compute_kernel_lib::tilize<width, in, out>(rows)` — one call per block, init/uninit once per core at Phase 0. No broadcast/reduce, so the broadcast checks do not apply. No mcast/semaphores (none needed).

### Design conformance
- Algorithm, pipeline topology and RISC ownership match `op_design.md`.
- Work distribution: `split_work_to_cores(grid, R, row_wise=True)` over the runtime grid; perf-focus shapes reach 110/110 cores. Both dataflow halves batched: the reader moves whole staging chunks (`num_banks` wide reads per chunk, double-buffered) and the writer issues a tile-row of writes per flush with one final barrier.
- **Deviation (improvement, accepted)**: the design's traffic ranking says interleaved RM sticks cannot be coalesced. The implementer added a bank-coalesced reader (sticks `p, p+num_banks, …` are contiguous in one bank → one wide read per bank per chunk + local reorder copies). Same DRAM bytes, far fewer transactions; 2.1–2.8× on the perf shapes. Gated on `num_col_blocks == 1`, interleaved source, aligned images, `cores ≥ 2·num_banks`. The design text should be updated by the planner on the next pass.
- Blocking-model fidelity: `block_height_tiles` (`BLOCK_HEIGHT_TILES`, default = whole assignment), `block_width_tiles` / `last_block_width_tiles` (`compute_blocking`, single source), `IN_/OUT_DEPTH_TILE_ROWS`, `STAGING_SLOT_BUDGET_BYTES` are all host parameters; every CB page count and trip count derives from them; `test_tilize_knobs.py` exercises non-trivial values. No CB scales with a whole-op dimension (staging is bounded because it exists only when one column block spans the stick). No collapsed or half-turned knob found. The coalesced reader ignores `block_height_tiles` (its quantum is the staging chunk) — consistent, since the push quantum per tile-row is unchanged.
- Expression: every phase works at tile-row granularity (the design's streaming quantum) with the block's width live; no per-tile handshakes.

### Prompt rules (`eval/prompts/tilize.txt`)
- MUST rules that apply now — in-kernel tilize of ROW_MAJOR input, no layout-converting ttnn op, grid read from the device at runtime, `SUPPORTED["tile_grid"]` only where the split reaches the grid, "none" sentinels in SUPPORTED, ValueError for malformed calls: all met.
- `low_l1=True` MUST be O(1) in tensor dims: holds (cap 8 tiles, staging only when C ≤ 8). Now SUPPORTED.
- Rules for `dtype=` casts, padding, sharding, retile re-arm with Refinements 1/2/4/5.

## Registry Conformance
- INPUT_TAGGERS (12, all `(inputs, axes)`), SUPPORTED, EXCLUSIONS (empty), validate() (SUPPORTED then EXCLUSIONS, typed `UnsupportedAxisValue` / `ExcludedCell`) present and wired. No `INVALID` in the op file.
- Auto-fixes to SUPPORTED: `low_l1 += True`, `buffer += dram_to_l1, l1_to_l1, l1_to_dram`, `rank += 2, 3, 5, 6`, `tile_height += 16, 8, 4, 2, 1`.

### INVALID audit (`eval/golden_tests/tilize/feature_spec.py`) — needs a spec update (not edited by the verifier)
- Existing entries are all `dtype × output_dtype` cast-family / integer-width rules: structural (the op's contract is value-preserving), documented. Canonical bf8b+ROW_MAJOR entry is not needed (bf8b/bf4b are not input dtypes in TARGET).
- **Missing structural entries**: `{dtype: fp8_e4m3, in_tile_height: h}` for h ∈ {32,16,8,4,2,1}. fp8_e4m3 is ROW_MAJOR-only (spec says so), so a TILE fp8 input cannot exist; today these 24 cells fail inside `from_torch` (`tensor_spec.cpp:144: layout == Layout::ROW_MAJOR`).
- **Unbuildable on this board, not declared**: fp8_e4m3 into any **sharded** input memory config — `from_torch` fails with `pytensor.cpp:167 ndarray_import failed` (32 cells). Either the harness should build sharded fp8 inputs another way, or `INVALID_FOR_ARCH["blackhole"]` should list them.
- These 56 cells are the entire `xfail_wrong_mode` bucket; they fail before the op is called, so nothing in the op can fix them. Please update via `/golden-tests`.
- Harness note: `axes.classify_call` has the same legacy/nd ambiguity as above (88 declared/captured `shard_api` mismatches), and the low_l1 A/B runs `False` first so captured `low_l1` is `False` on 71 rows. Both are harness-side.

## L1 Ledger Audit
- **Currency**: 5 CBs declared (0 `cb_rm_sticks`, 16 `cb_tiles_out`, 1/17 ragged-tail rings, 2 reader staging); all have rows; size expressions match `tilize_program_descriptor.py`. Total footprint was missing the staging term — fixed.
- **Capacity vs live set**: main and tail rings are 2× the live set by the declared double-buffer decision. Staging: 2 slots (one being copied out, one landing) — justified. Axis accounting: rings span `cols` (scales with `block_width_tiles`) and stream `rows`; no collapsed extent, no inflated window.
- **Page format vs DEST**: all pages bf16 with `fp32_dest_acc_en=False` — consistent. fp32/int pages arrive with Refinement 5 together with `fp32_dest_acc_en=True`.
- **Disjoint lifetime**: tail rings vs main rings have disjoint lifetimes; ledger justifies not aliasing (no CB alias API, ≤ main size, ragged shapes only). Staging and tail rings are mutually exclusive by predicate. Accepted.
- **Bounds**: every capacity symbol bounded (block cap 255 / L1 budget / `LOW_L1_BLOCK_WIDTH_TILES`; staging bounded via the `num_col_blocks == 1` predicate). No unbounded op dimension.
- **Block-size defaults**: held — interleaved split spreads tile-rows over the full grid, then takes the coarsest block (whole assignment × min(C, cap)).
- **Per-core footprint**: `block_width_tiles·(2·in_page + 2·out_page) + [coalesced] 2·slot_bytes + [ragged] tail rings` ≤ 512 KiB + 256 KiB (`low_l1=False`), ≤ 192 KiB (`low_l1=True`). Example `[1,1,16384,64]`: 16 KiB rings + ~32 KiB staging.
- **Data-movement budget**: present and consistent with the implemented split (each input byte crosses DRAM once, each output byte once; coalescing adds only local L1 copies). Cheapest-traffic split = rows-only, implemented. 2-D split is a `deferred` regime row with a positive reason (serves only short_wide/square_large, outside Phase 0) → now Refinement 1.
- No ledger finding needed a refinement.

## Precision Baseline

`test_tilize_precision_baseline.py` (bf16 → bf16, `randn·4`):

| Shape | tile_h | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio (median, p5–p95) |
|-------|--------|-----|-------------|--------------|------------------|---------|-------------------------------|
| (1,1,32,32) | 32 | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0–1.0 |
| (1,1,256,128) | 32 | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0–1.0 |
| (2,3,640,160) | 32 | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0–1.0 |
| (1,1,16384,64) | 32 | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0–1.0 |
| (1,1,4096,64) | 16 | 1.0 | 0 | 0 | 0 | 0 | 1.0, 1.0–1.0 |

**Assessment**: bit-identical on every path (coalesced and non-coalesced readers, fast tilize and `tilize_block`). No scale or precision issue.
**Recommended tolerances**: exact equality (`torch.equal`) for no-cast calls; `helpers._transition_tolerance` for `dtype=` casts.

## Verifier CLI Summary (final run)
- supported_pass: 238
- xfail_expected: 1925
- invalid_skipped: 2090
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 56 — all fp8_e4m3 cells that fail inside harness input creation (see INVALID audit); not fixable in the op
- no_axes_found: 81 — 71 skips in translated tests plus 10 `test_regression.py` calls (`test_integer_passthrough`, `test_extreme_magnitudes`, `test_pad_value_extremes`) refused by the dtype gate; they are covered by Refinement 5

## Perf snapshot (BH, device kernel ns from the golden run)

| LOOSE_CASE | ns | cores | BH reference |
|------------|----|-------|--------------|
| `[1,1,16384,64]` (PERF FOCUS) | 11 348 | 110 | 11 852 |
| `[1,1,16384,32]` | 8 022 | 110 | 7 193 |
| `[1,1,32768,64]` | 21 028 | 110 | 20 834 |
| `[1,1,128,64]` | 2 153 | 4 | 1 679 |

Other LOOSE_CASES are outside SUPPORTED until Refinements 1, 2, 5.

## TARGET − SUPPORTED coverage
| Axis | Missing values | Where |
|------|----------------|-------|
| tile_grid | short_wide, square_large | Refinement 1 |
| pad_mode / pad_value / alignment | auto, explicit / zero, positive, negative / w_, h_, hw_non_aligned | Refinement 1 |
| rank | 0, 1 | Refinement 1 (only reachable with a pad) |
| shard_api / out_scheme / orientation | legacy_2d, nd / HEIGHT, WIDTH, BLOCK, ND / ROW_MAJOR, COL_MAJOR | Refinement 2 |
| in_tile_height | 32, 16, 8, 4, 2, 1 | Refinement 4 |
| dtype | float32, fp8_e4m3, uint32, int32, uint16, uint8 | Refinement 5 |
| output_dtype | float32, bfloat8_b, bfloat4_b, uint32, int32, uint16, uint8 | Refinement 5 |

## Recommendations
- Ordering: R1 (2-D split + padding) must precede R2 (sharding) because `sharded_full_grid` is the 2-D split with sharded accessors; dtype widening goes last so it extends every finished path once.
- Tiny tiles on Wormhole: the translated suite skips them ("LLK for tiny tiles not fully supported on WH"); they pass on this Blackhole board. Re-check on WH is outside this board's reach.
- `[1,1,128,64]` runs on 4 cores (R = 4) at 2.15 µs vs reference 1.68 µs; launch latency dominates. The design lamp suggests the 2-D split / fewer-core variants. Worth measuring during the perf refinements, as a guard-set member.
