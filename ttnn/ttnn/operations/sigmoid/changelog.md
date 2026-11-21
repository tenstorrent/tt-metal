# Changelog: sigmoid

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  R-stream: one `ttnn.ProgramSpec` / one `ttnn.generic_op` per call, full 32-cluster grid, thread plan (2, 4, 2),
  fused `eltwise_chain(CopyTile → SigmoidAccurate → PackTile)` per NEO.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[DRAM, L1],
  sharding=[interleaved], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4]
- **Accuracy achieved**: PCC ≥ 0.99986 (0.999991 on N(0,1)), max_abs_err=1.95e-3, rel_rms_err ≤ 1.7e-2
  (4.2e-3 on N(0,1)), **0 bf16 ULP on every element** (measured on 4 shapes × 3 distributions + 2 config
  variants via test_sigmoid_precision_baseline.py)
- **Golden suite at Phase 0**: 80 / 599 passing (74 PHASE0 cells + 6 NUMERIC regression), 518 xfail_expected,
  0 loud (per `verifier_report.json`)
- **Issues encountered**: verifier fix — `compute_kernel_config.dst_full_sync_en` was dropped; now mapped to
  `double_buffer_dest = not dst_full_sync_en`, with `block_tiles = DEST_TILES[(fp32_dest_acc_en,
  dst_full_sync_en)]`. No drift fixes needed.
- **Tests added**: test_sigmoid.py (harness), test_sigmoid_threads.py, test_sigmoid_precision_baseline.py
- **Structural perf counts (no clock on ttsim)**: 4 NEOs × up to 32 clusters at work, ⌈n/128⌉ pages max per NEO
  (the minimum), one DRAM crossing per tensor, one ring handshake per DEST block (4 tiles), 57 SFPU instructions
  per DEST row in the sigmoid element (compiled math TRISC).

### Phase 0 implementer notes

Built regime **R-stream** of `op_design.md` as designed: one `ttnn.ProgramSpec`, one `ttnn.generic_op`
dispatch per call.

- **Work split**: the flat page axis in whole rounds of C pages per cluster over the full grid
  (`split_rounds`, `sigmoid_program_artifacts.py`). Each input and output page crosses DRAM (or the L1 banks)
  once; no cross-core traffic. Matches the ledger's data-movement budget.
- **Thread plan** (R, C, W) = (2, 4, 2) from the one host constant `THREAD_PLAN`; counts reach the kernels as
  CT args; loops sized from them and `get_my_thread_id()`.
- **Knobs** (each defined once): `THREAD_PLAN`, `BUFFER_DEPTH` = 2, `DEST_TILES[fp32_dest_acc_en]` ->
  `block_tiles` = 4. Ring entries = `BUFFER_DEPTH * block_tiles * C` = 32 per ring (64 KiB bf16). L1 inventory
  is the ledger's, unchanged: two rings, no intermediate.
- **Compute**: one `eltwise_chain` per NEO, `CopyTile -> SigmoidAccurate -> PackTile`, PerBlockSize sync
  blocks of `block_tiles` (one DEST window each). `SigmoidAccurate` is an op-local chain element:
  clamp to [-88, 88] then `_sfpu_sigmoid_<DST_ACCUM_MODE>` (accurate exp + Newton reciprocal), init through
  `sigmoid_init<false>` (programs vConstFloatPrgm0 = 2.0). The library `Sigmoid<>` is the LUT path on Quasar
  and is rejected (reason at the head of `sigmoid_compute.cpp`).
- **Reader / writer**: implicit-sync tagged NoC transfers, one per page, tag handed back as the last statement.

Tests:
- acceptance `test_sigmoid.py`: 7/7.
- golden suite: 80 passed (74 PHASE0 cells + 6 NUMERIC regression), 0 failed, 518 refused cells xfail.
- `test_sigmoid_threads.py`: every NEO ran (odd NEOs tagged with sigmoid(-x)); plans (1,1,1), (1,4,1),
  (2,4,2), (4,4,2) on 1, 3, 39 and 789 pages; one device session tilize -> sigmoid -> sigmoid -> untilize.
- Probe: 0 bf16 ULP against the correctly-rounded float64 reference on the probed shapes.

Deviations from the design (advisory items only):
- Kernel files are `kernels/sigmoid_{reader,compute,writer}.cpp` (task prompt naming), not
  `{reader,compute,writer}.cpp`.
- The session test uses `ttnn.to_layout(..., TILE_LAYOUT)` as the program before the op instead of
  `ttnn.add`: `ttnn.add` is a Metal 1 op and is refused on Quasar (`kernel.hpp:605`).
- `dst_full_sync_en` was not mapped onto `double_buffer_dest` (fixed by the verifier, above).

Performance: this backend has no clock, so no timing is claimed. Structural counts: 4 NEOs per cluster at work,
up to 32 clusters, DEST-sized sync blocks (one ring handshake per 4 tiles), one DRAM crossing per tensor. The
design's perf lamps (plan alternatives, block size 1, BUFFER_DEPTH 3, whole-round split on tiny tensors) are
left for a timed backend.
