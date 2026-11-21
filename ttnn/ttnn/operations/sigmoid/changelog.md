# sigmoid changelog

## Phase 0 (ttnn-implementer)

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
- `dst_full_sync_en` is not mapped onto `double_buffer_dest` (the design's config table lists three fields);
  `block_tiles` assumes half-sync DEST accordingly.

Performance: this backend has no clock, so no timing is claimed. Structural counts: 4 NEOs per cluster at work,
up to 32 clusters, DEST-sized sync blocks (one ring handshake per 4 tiles), one DRAM crossing per tensor. The
design's perf lamps (plan alternatives, block size 1, BUFFER_DEPTH 3, whole-round split on tiny tensors) are
left for a timed backend.
