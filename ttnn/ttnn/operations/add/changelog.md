# Changelog: add

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  `stream_flat` of op_design.md: Metal 2.0 `ProgramSpec`, one `ttnn.generic_op` per call, Quasar on ttsim.
  - Work distribution: flat page ranges over the full grid (`split_work_to_cores`, row_wise, in rounds of C
    pages); clusters engaged = min(num_tiles, 32) (tested: 13, 52, 263, 1050 tiles).
  - Thread plan (1, 1, 1) from host constant `THREAD_PLAN` (data-movement bound); plans (2,2,2), (4,4,2), (1,4,1)
    tested correct with a NaN-prefilled output (a missing thread leaves NaN pages).
  - Knobs `BLOCK_TILES`=4, `RING_DEPTH_BLOCKS`=2, `MIN_TILES_PER_CORE`=1 are live and test-turned (block 1/2/4
    with ragged tails, depth 2/3/4, min tiles 1/4/8). The host also clamps `block_tiles` to the config's DEST
    section (mirrors `dest_helpers.hpp`), so a turned knob cannot trip the kernel's static_assert.
  - Reader/writer: implicit sync, one tagged transfer per page, tag handed back; session test with tt-metal's
    `to_layout` before and after passes.
  - Compute: one `compute_kernel_lib::add` call per NEO (PerBlockSize wait/pop/reserve/push, Block mapping).
  - L1: 3 rings, 49,152 B; a, b, out cross their tier once each (`l1_ledger.md`).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[DRAM, L1],
  sharding=[interleaved], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4],
  broadcast=[none], output_dtype=[None]
- **Accuracy achieved**: PCC=0.999998, max_abs_err=0.0156, rel_rms_err=0.0018, ≤ 1 bf16 ulp (≈1e-5 of elements at
  1 ulp, rest 0), got/true ratio median 1.0000 (measured on 4 shapes via test_add_precision_baseline.py)
- **Golden suite at Phase 0**: 87 / 4694 passed (86 supported_pass + rejection X1), 3046 xfail_expected,
  1560 skipped (INVALID / unbuildable bf8b), 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode, 0 hangs
  (per `verifier_report.json`)
- **Issues encountered**: verifier fixes only — replaced an unused `NEOS` constant and a duplicated `(1, 2, 4)`
  literal with one `COMPUTE_THREAD_COUNTS`; corrected two stale `l1_ledger.md` symbol-table bounds
  (`block_tiles` DEST clamp, `RING_DEPTH_BLOCKS` predicate). No drift fixes.
- **Tests added**: test_add.py (generated acceptance), test_add_plan.py (thread plans, knobs, session, structural
  counts, refusals), test_add_precision_baseline.py
- **Performance**: no time figure — ttsim has no clock; perf is structural on this backend.

## Refinement 0 — Height-sharded operands in L1 (the YOLOv8 bottleneck add)
- Date: 2026-11-21
- What was done: `"height"` added to `SUPPORTED["sharding"]` (both operands height-sharded in L1; R9, R10). No
  kernel, ring or program-spec change. Reused: the Phase 0 `stream_flat` path unchanged. The reader and writer
  address every tensor through its `TensorAccessor` by logical page id, so a sharded operand's pages are fetched
  from whichever core holds them. The work split stays the full-grid flat page split, independent of the shard
  grid. The output is allocated from `a.spec`, so it is height-sharded in L1 with a's shard shape, grid and
  orientation (R9). Added: the SUPPORTED value, comments in `add.py`, an `l1_ledger.md` note (same crossings,
  tier = the shard holders' L1 over the NoC), and the tests below.
- Accuracy achieved: the result equals the fp32 sum rounded once to bf16 (`torch.equal` on [1, 1, 64, 128]
  sharded over two cores, the traced YOLOv8 call). All sharded golden cells meet the 1-ulp bf16 requirement.
  Unit tests check allclose(1e-2) against the single-rounded sum on [1,1,64,128], [1,2,400], [1,1,50,128],
  [128,100], [1,1,1024,64] and [1,3,320,320].
- Golden test progress: targeted slice: 39/39 buildable `sharding=height` × bfloat16 × TILE × broadcast=none ×
  output_dtype=None cells pass; [1, 2, 33600] is skipped by the runner (its 2,150,400 B shard does not fit one
  core's L1). The interleaved representatives (DRAM and L1; [1,2,400], [1,1,1600,128], [17,64]) still pass, and a
  sharded broadcast cell is still refused (xfail). Full-suite count: the harness's run.
- Issues encountered: None. Finding (not done here, as the refinement asked): the shards are not consumed in
  place. Every page still crosses the NoC once from its holder's L1, and the op's cores are not aligned with
  the shard grid. A zero-copy ring backed on each local shard (`cb_descriptor_from_sharded_tensor` /
  the ProgramSpec equivalent), with the core set taken from the shard grid, would remove those crossings. That
  is its own performance refinement.
- Tests added: `test_add_sharded.py`. It checks: R9 placement (height-sharded L1, a's shard shape, grid and
  orientation) on 4 runner-placed shapes; shard grids of one core, two cores, the full grid and an offset pair of
  cores; a and b on different shard grids; thread plans (1,1,1) and (4,4,2) on sharded operands into a
  NaN-prefilled sharded output; one device session (`interleaved_to_sharded` before, add twice, `to_layout` and
  `sharded_to_interleaved` of the sharded result after, then an interleaved add); and refusal of width
  sharding by name (R10).
