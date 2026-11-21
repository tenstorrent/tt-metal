# Verification Report: avg_pool2d

Target: Quasar (`quasar`) on the ttsim simulator (no clock: no timing figure below is a performance result).
Specification: `eval/specs/avg_pool2d/spec.py` (sha256 `bb27513b1abd`). Design: `op_design.md`, regime `col_split_stream`.

## Code Review

Files reviewed: `avg_pool2d.py`, `avg_pool2d_program_artifacts.py`, `kernels/avg_pool2d_{reader,compute,writer}.cpp`, `l1_ledger.md`, `op_design.md`, the acceptance and design tests.

**Fixed in place**
- **Scaler truncation (precision)**: the writer fills the scaler ring through `prepare_reduce_scaler`, whose `float_to_scaler_bits` keeps the top 16 bits of the fp32 value (`reduce_helpers_dataflow.inl:38`). So `1/(H*W)` was *truncated* to bfloat16, a downward scale error of up to 2⁻⁸ that depends on H*W (H*W = 15: ×0.9961, a median got/true ratio of 0.9963 measured). The host now rounds the scaler to the nearest bfloat16 (ties to even) before passing its bits (`_scaler_bits` in `avg_pool2d_program_artifacts.py`), which halves the worst case to 2⁻⁹. The bits are unchanged where truncation was already nearest (H*W = 49, 196). There is no kernel change. Measured on 3×5×96: relative RMS 0.0043 → 0.0036, ratio 0.9963 → 1.0035. Golden suite re-run: unchanged, clean.
- `avg_pool2d_program_artifacts.py`: the comment on `INPUT_DEPTH_CHUNK_ROWS` claimed two chunk rows of reads in flight. The reader barriers each chunk row before issuing the next, so the second slot is double buffering against compute, not a second batch on the wire. The comment now says what the code does. Keeping more reads in flight is Refinement 5.
- `op_design.md`: the Dataflow Strategy row for `cb_input_tiles` still described an implicit-sync reader with no padding fill. Added an *As built* note: the ring is explicit sync with the R1 zero fill. The next stage plans against the design, so it has to match the code; the ledger already carried the note.

**Checked, no change needed**
- Kernel syntax and APIs: `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor`, Metal 2.0 `ProgramSpec` + one `ttnn.generic_op` (one native dispatch per call).
- Ring balance: `in` gets `Ht × core_ct_tiles` pushed by the reader (batched per chunk row) and the same number popped one at a time by the helper. `scaler` gets 2 pushed by the writer, waited by the helper, and popped once by compute after the last block. `out` gets `core_ct_tiles` packed and drained by the writer's implicit-sync writes.
- Transaction tags: the reader is explicit sync and tags every transfer with user id 0. The writer sets its tags to 0 before `prepare_reduce_scaler`'s untagged `write_zeros` and hands the write tag back as its last statement. `test_one_device_session_before_and_after_other_programs` passes.
- Helper use: one `reduce<SUM, REDUCE_COL, ..., WaitAndPopPerTile, NONE>` per block with the partial scaler. `WaitAndPopPerTile` is the sanctioned reader↔compute streaming policy, and the block's live set (DEST accumulators of a chunk) is held across the positions walk, so this is not a per-unit expression failure. The 1/(H*W) rides in the scaler, so no post-op is needed. `reduce_mean<>` / `AccumulateViaAdd` are not used.
- Blocking-model fidelity:
  - `BLOCK_CT_TILES_CAP`, `INPUT_DEPTH_CHUNK_ROWS`, `OUTPUT_DEPTH_CHUNKS`, `THREAD_PLAN` and `DOUBLE_BUFFER_DEST` are each defined once on the host.
  - `dest_chunk_tiles` is derived from the same `ComputeHardwareConfig` the compute kernel gets, and `static_assert(DEST_AUTO_LIMIT == dest_chunk_tiles)` checks it.
  - The block loops run over `block_ct_tiles` (RT arg) in all three kernels, and no ring scales with an op dimension.
  - The per-core block is the whole assignment, so the split is turned on both count and size.
  - `test_several_chunks_and_blocks_per_cluster` turns the knobs (narrowed grid, block cap 3).
- Work distribution: channel tile columns over the full queried grid (`split_work_to_cores(..., row_wise=True)`), up to 32 clusters (64 tiles over 32 on the model's map). Both dataflow halves are batched: the reader per chunk row, the writer with implicit-sync tagged writes.
- Thread plan: `THREAD_PLAN = (1, 1, 1)` is one host constant passed as CT args; each kernel `static_assert`s it. The design states the op as bound by data movement, with the reason (one FPU `reduce_tile` per 2 KB tile moved once), which the target page accepts for (1, 1, 1). `test_thread_plan_and_channel_coverage` reads the plan off the artifacts and checks every cluster's channel range with a ramp. This backend has no timing model (`pipeline_logs/model_cycles.jsonl` is absent), so the plan was not measured against (1, 4, 1). That comparison belongs on `ttsim-perf`.
- Specification rules:
  - R1: rows ≥ H*W zeroed by the reader; `test_padding_rows_do_not_enter_the_mean` poisons them with 1000 / inf / nan.
  - R2: `default_compute_kernel_config()` is the only default; the lint passes.
  - R3: no gate on fidelity or approximation.
  - R4: device config read by attribute; the model test passes a `WormholeComputeKernelConfig`.
  - R5: both DEST widths are honoured; `dest_chunk_tiles` follows.
  - R6: lists and tuples, 2- or 4-value padding, any stride ≥ 1.
  - R7: one image in Phase 0.
  - R8: `output_dtype` and `output_layout` are gated by name.
  - R9: output is `MemoryConfig(INTERLEAVED, input buffer type)`, checked by `HOOKS["check"]` in every cell.
  - R10: width shards on any grid.
  - R11: `eval.memory.memory_axes` and the spec's `tag_window`.
  - R12: `UnsupportedAxisValue` / `ExcludedCell`, both `NotImplementedError`.
  - X1–X3: raise `ValueError` / `TypeError`; the golden rejection tests pass.
  - The structure-free rules are covered by cells; no rule carries a `structure` part.
- GUIDANCE advisories: all followed. One exception is placement: the GUIDANCE asks for one accessor path for every placement, which Phase 0 does, but it conflicts with the native-sharding requirement (below).

**Deferred to the refinement queue (architectural)**
- **Width-sharded inputs are gathered, not consumed in place.** A width shard keeps the reduction local: each shard holds every position of its channels. Phase 0 reads every width-sharded tile through the `TensorAccessor` onto a full-grid channel split, so the shard-holding clusters re-read their own resident shard over the NoC like an interleaved tensor. The design did this on purpose (32 clusters engaged instead of the model's 2 shards) and recorded the zero-copy regime `shard_in_place_width` as deferred. But its reason is occupancy alone, and the sharded block default (the whole resident shard) was departed from without a measurement. The claimed `sharding: width` value is therefore not implemented natively. This is **Refinement 1**. It is a real build, not a quick fix: a borrowed ring, an R1 mask in the reader, and a helper-order constraint, all listed in its notes.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from `eval.specs.avg_pool2d.spec`. No op-local axis function.
- `SUPPORTED` equals the specification's `PHASE0` exactly (lint `--phase0` passes: 0 errors). It gates every axis the kernel branches on: dtype, fp32_dest_acc_en, layout, memory_config, sharding, output_dtype, output_layout, window, batch, channel_alignment.
- `EXCLUSIONS = []`.
- `validate()` checks malformed calls first (X1–X3), then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). The entry point calls it on its first line, before any device work.
- The op file declares no `INVALID`.
- No SUPPORTED auto-fix: no XPASS evidence.
- INVALID audit (specification): the two entries `{memory_config: DRAM, sharding: height|width}` describe one tensor (the input's placement), encode a structural impossibility of the runner (no DRAM-sharded tensor is built), and pass the three sanity rules. There is no bf8b + ROW_MAJOR entry, and none is needed: `dtype` has bfloat16 only (Quasar has no bfloat8_b path). There are no weight axes, so the canonicalization rules do not apply.
- Specification lint: PASSED, 0 errors, 6 UNRESOLVED open items (numeric list mode, `governs`, `special_values`, `freedoms`, `PERFORMANCE`). None is a defect.

## L1 Ledger Audit

- **Currency**: three rings in the code (`in`, `scaler`, `out`), three rows in the ledger. Sizes match: `INPUT_DEPTH_CHUNK_ROWS × dest_chunk_tiles × C`, `2`, `OUTPUT_DEPTH_CHUNKS × dest_chunk_tiles × C` with C = 1, page `buffer_page_size()` = 2048 B. The explicit-sync change is recorded.
- **Capacity vs live set**:
  - `in`: capacity 2 chunk rows, compute live set 1 tile. The gap is the reader's batch plus double buffering, stated in the row. Not inflated over an axis: positions and channels stream.
  - `out`: 2 chunks, spans one DEST chunk of `channel`, and its capacity scales with `dest_chunk_tiles`. No collapsed extent.
  - `scaler`: constant 2.
  - No buffer is sized to a whole-op dimension.
- **Page format vs DEST**: all pages are Float16_b. With `fp32_dest_acc_en=True` the `out` page is the final bfloat16 result (output ABI), packed once after the whole sum is in DEST, so nothing is truncated mid-accumulation. No Float32 page exists with a 16-bit DEST.
- **Disjoint lifetimes**: none; all three rings live for the whole program and each "why not share" cell is filled.
- **Bounds / closed form**: footprint = `2048 × (INPUT_DEPTH_CHUNK_ROWS·dest_chunk_tiles + 2 + OUTPUT_DEPTH_CHUNKS·dest_chunk_tiles)` = 36,864 B (32-bit DEST) / 69,632 B (16-bit DEST) per cluster. Every symbol is bounded; no op dimension appears. The input-ring term scales with `INPUT_DEPTH_CHUNK_ROWS` and the DEST mode, the output-ring term with `OUTPUT_DEPTH_CHUNKS` and the DEST mode.
- **Data-movement budget**: present and consistent with the code (input once, output once, scaler built in place, no re-read). Finding: the cheapest-traffic split (`shard_in_place_width`, −1 NoC crossing of a width-sharded input, −262,144 B on the model's call) is deferred on occupancy alone, which is not a positive reason. It is **folded into Refinement 1**, which builds it and updates the ledger. No separate entry.
- **Block-size defaults**: the interleaved default holds (full grid, coarsest block = the whole assignment). The sharded default (one block = the resident shard) does not: departed from without measurement. Folded into Refinement 1.
- `positions_split_combine` (split the reduced axis for shapes whose channel tiles under-fill the grid: C = 32 runs on 1 cluster) stays deferred. The reason is positive: every such INPUTS entry is ≤ 64 KB (≤ 32 tiles), and a cross-cluster combine adds a partial tile per column and a semaphore round trip to a reduce of 2–32 tiles.

## Precision Baseline

`tests/ttnn/unit_tests/operations/avg_pool2d/test_avg_pool2d_precision_baseline.py`, 12/12 pass. After the scaler fix. Seed 1234. Normal inputs except the model config (uniform [0, 1), as the loose case). ULP is at bfloat16 against the reference rounded to bfloat16. The ratio is got/true over the half of the outputs farthest from zero.

| Shape (placement) | Config | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p50 / p99 / max | ratio p5 / median / p95 |
|---|---|---|---|---|---|---|---|
| 7×7×64 (DRAM) | HiFi4, 32-bit DEST | 0.999998 | 7.3e-04 | 2.0e-04 | 0.0022 | 0 / 1 / 1 | 0.9959 / 0.9984 / 1.0015 |
| 7×7×64 (DRAM) | HiFi4, 16-bit DEST | 0.999996 | 1.4e-03 | 3.4e-04 | 0.0034 | 1 / 4.4 / 5 | 0.9948 / 0.9970 / 1.0028 |
| 7×7×64 (DRAM) | LoFi, approx, 16-bit, uniform | 0.999508 | 4.3e-03 | 1.2e-03 | 0.0375 | 0 / 1 / 1 | 0.9935 / 0.9982 / 1.0031 |
| 7×7×2048 (L1 width, 2 cores) | HiFi4, 32-bit DEST | 0.999999 | 1.4e-03 | 1.9e-04 | 0.0020 | 0 / 1 / 10 | 0.9962 / 0.9988 / 1.0016 |
| 7×7×2048 (L1 width, 2 cores) | HiFi4, 16-bit DEST | 0.999996 | 2.3e-03 | 3.0e-04 | 0.0030 | 0 / 12 / 219 | 0.9946 / 0.9989 / 1.0035 |
| 7×7×2048 (L1 width, 2 cores) | LoFi, approx, 16-bit, uniform | 0.999438 | 5.0e-03 | 1.2e-03 | 0.0362 | 0 / 1 / 2 | 0.9938 / 0.9989 / 1.0040 |
| 14×14×32 (L1) | HiFi4, 32-bit DEST | 0.999998 | 3.1e-04 | 8.6e-05 | 0.0020 | 0 / 1 / 1 | 0.9967 / 1.0001 / 1.0017 |
| 14×14×32 (L1) | HiFi4, 16-bit DEST | 0.999976 | 1.6e-03 | 2.7e-04 | 0.0070 | 1 / 6.4 / 7 | 0.9908 / 0.9994 / 1.0060 |
| 14×14×32 (L1) | LoFi, approx, 16-bit, uniform | 0.995581 | 5.6e-03 | 1.5e-03 | 0.0930 | 0 / 2 / 2 | 0.9914 / 0.9988 / 1.0057 |
| 3×5×96 (DRAM) | HiFi4, 32-bit DEST | 0.999999 | 3.6e-03 | 7.7e-04 | 0.0036 | 1 / 1 / 1 | 1.0007 / 1.0035 / 1.0054 |
| 3×5×96 (DRAM) | HiFi4, 16-bit DEST | 0.999999 | 3.6e-03 | 7.7e-04 | 0.0036 | 1 / 1 / 1 | 1.0007 / 1.0035 / 1.0054 |
| 3×5×96 (DRAM) | LoFi, approx, 16-bit, uniform | 0.999933 | 4.0e-03 | 1.7e-03 | 0.0258 | 1 / 1 / 1 | 1.0002 / 1.0038 / 1.0067 |

**Assessment**:
- With a 32-bit DEST the result is within 1 bfloat16 ULP at p99. A 16-bit DEST adds a few ULP on the longer sums (Ht = 7: p99 6.4).
- **Scale triage**: PCC is high on every case and relative RMS is low (≤ 0.007 on normal inputs), so there is no scale-bug signature. A small uniform scale remains, set by H*W alone: the scaler is `1/(H*W)` in bfloat16, now the nearest one. Ratios: 15 positions 1.0035 (1/15 → 0.06689, ×1.0034); 49 positions 0.9988 (1/49 → 0.020386, ×0.9989).
- 196 positions gives 1.0001. That is the bfloat16 scaler (×0.9989) offset by the FPU reduce's tf32 accumulation bias, about +1.7e-4 per tile over 7 tiles.
- Before the fix the scaler was truncated, not rounded (15 positions: 0.9963); see Code Review.
- The bfloat16 scaler is also most of the loose case's 0.036 relative RMS on uniform inputs, whose outputs sit near 0.5 with a spread of 0.04.
- The 14×14×32 LoFi uniform case has a relative RMS of 0.093: the outputs of 196-term means spread by only ~0.02, so the RMS denominator is small. The spec grades that configuration by PCC only, and PCC is 0.9956 ≥ 0.99.

**Recommended tolerances**: the defaults hold, `NUMERIC["legacy_tolerances"]` (bfloat16, True) = (0.995, 0.04) and (bfloat16, False) = (0.99, 0.13). The smallest margins are relative RMS 0.0070 vs 0.13 (16-bit DEST) and PCC 0.99558 vs the loose 0.99.

## Verifier CLI Summary

Golden run `/tmp/ap_golden2` after the scaler fix (8 sim workers, 824 tests; the run before it, `/tmp/ap_golden1`, gave the same counts), report copied to `ttnn/ttnn/operations/avg_pool2d/verifier_report.json`.

- supported_pass: 53 (plus the loose case `resnet50_global_lofi`, PCC 0.99946; 4 numeric regression cases; 3 rejections X1–X3; 56 passed in total)
- xfail_expected: 496
- invalid_skipped: 272
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- supported_marked_xfail / invalid_unexpected: 0 / 0
- no_axes_found: 3 (the rejection tests; not registry-driven)
- Queue coverage (`--queue op_requirements.md`): 5 values of TARGET − SUPPORTED, 0 gaps.

Acceptance + design tests: 45/45. Precision baseline: 12/12.

## Recommendations

- **Queue shape**:
  - Refinement 1 completes the claimed width placement natively before anything builds on it. Refinement 2 (batched images) is the remaining scheme-change, hardest first.
  - Refinement 3 (perf) puts four NEOs to work on Refinement 1's in-place path for the model's call. No loose case is perf-flagged, so the model's call is free-selected as the region.
  - Refinement 4 bundles the reader/writer data-access values (row-major in and out, non-tile-aligned C, height sharding).
  - Refinement 5 (perf) keeps reads in flight on the interleaved reader.
  - Every perf gate is a structural count, because this backend has no clock.
- **Occupancy trade of Refinement 1 (finding for a timed backend)**: in place, the model's call reduces on its 2 shard clusters instead of 32. Whether in-place plus four NEOs (Refinements 1 and 3) beats the 32-cluster gather can only be timed on `ttsim-perf` or hardware. If it loses there, a predicate-gated hybrid is the follow-up: in place when the shard grid spans at least the clusters the channel split would engage. This is not a Done-when of any refinement here.
- **Scaler rounding (precision, no failing cell)**: after the fix `1/(H*W)` is the nearest bfloat16, a uniform scale error of up to 2⁻⁹ that depends on H*W. A Float32 scaler page would remove it (`prepare_reduce_scaler` supports Float32). Check first that a bf16 SrcA with an fp32 SrcB runs on ttsim at HiFi4. Not a refinement: no cell fails, and every tolerance has margin.
- **Partial scaler on Quasar**: kept; R1 rests on the reader's zero fill alone. Every refinement that adds a new input path (borrowed shard, row-major sticks, image boundaries inside a tile) must mask in the data-movement kernels, not with the scaler.
- **Hardware constants in host code**: `NEOS = 4` and `FREE_DM_HARTS = 6` in `validate_thread_plan` restate the target page; no run-time query exists for them. Harmless while the op is Quasar-only.
