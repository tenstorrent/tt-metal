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

## Refinement 20 — `memory_config`, a given output dtype, width-sharded operands, a second operand of lower rank (Llama, Qwen)
- Date: 2026-11-21
- What was done: host-side only; no kernel, ring or program-spec change. A probe first (gate bypassed, per the
  operator note) showed the `stream_flat` path already serves width-sharded operands and a lower-rank b with
  broadcast "none" (equal padded page sequences), bit-exact on 5 shapes.
  - Reused: the Phase 0 / Refinement 0 page walk through each tensor's `TensorAccessor` (by logical page id, so
    the work split and kernels do not depend on how a tensor is sharded), the full-grid split, `validate()`.
  - Added: the keyword-only `memory_config=None` (signature now equals the stub; `validate_contract` passes);
    the axis `memory_argument` derived with the specification's own `canonical_memory` (None / "input" /
    "other"; "other" is refused by name, R11); SUPPORTED `memory_argument` = [None, "input"],
    `output_dtype` += bfloat16 (R8), `sharding` += "width" (R9, R10). The output is allocated by
    `_output_spec`: a's memory config (interleaved in a's buffer, or sharded with a's shard shape, cores and
    orientation), the broadcast shape (a lower-rank a against a rank-4 b now gets the rank-4 result, not a's
    shape) and the requested dtype; it is `a.spec` itself when shape and dtype equal a's.
- Accuracy achieved: <= 1 bf16 ulp of the single-rounded fp32 sum on every shape tested ([1,1,32,2048],
  [1,1,32,256] width-sharded over two cores, [1,1,128,512]+[128,512] and +[1,128,512] in DRAM/L1/height/width,
  [1,1,64,1024] width-sharded over 1/2/32/offset cores, [1,3,320,320] at plans (1,1,1) and (4,4,2)); most
  elements bit-exact, rare elements 1 ulp off torch's rounding (as Phase 0's baseline recorded).
- Golden test progress: targeted slices: all 572 cells of this refinement's region (bfloat16, TILE, fp32 DEST,
  broadcast none, with `memory_argument=input`, `output_dtype=BFLOAT16` or `sharding=width`) plus the 16
  `[1,1,128,512]+[128,512]` cells and prior interleaved/height representatives: 584 passed, 5 skipped by the
  runner (shards that do not fit one core's L1, e.g. [1,2,33600]), 0 failed. Full-suite count: the harness's run.
- Issues encountered: None. Finding (unchanged from Refinement 0, now also for width): shards are not consumed
  in place; every page crosses the NoC once from its holder's L1 and the op's cores are not aligned with the
  shard grid. A zero-copy ring borrowed from each local shard (design regime `sharded_in_place`) is its own
  performance refinement.
- Tests added: `test_add_models.py` — the models' calls with placement asserted (Llama DRAM call with
  `memory_config=DRAM` -> DRAM interleaved; Qwen width-sharded call with its own config and `dtype=bfloat16` ->
  width-sharded [32, 128] on the same two cores), the lower-rank b in every placement, a lower-rank a (output has
  the broadcast shape), `memory_config=input` == None in every placement, refusal of another config and of
  `dtype=float32`, width shard-grid independence (1, 2, full, offset cores; a and b on different grids), plans
  (1,1,1)/(4,4,2) into a NaN-prefilled width-sharded output, and one device session (`interleaved_to_sharded`
  before, `to_layout` and `sharded_to_interleaved` after). `test_add_sharded.py`: the width-refusal test became a
  block-sharding refusal test (R10).

## Refinement 1 — Broadcasting: tensor broadcast and a Python-number operand
- Date: 2026-11-21
- What was done: SUPPORTED["broadcast"] += over_w, over_h, over_hw, over_leading, number (all five). Scheme change,
  design regimes `bcast_resident_b` and `number_scalar` (op_design.md, *Refinement 1 — broadcast as built*).
  - Reused: the full-grid flat page split, the three-kernel skeleton, the implicit-sync a reader and the writer
    (unchanged), the eltwise chain helper, `validate()` (plus a refusal of two operands that both broadcast).
  - Added: `plan_operands` (host swaps the operands when a broadcasts, R4 — FPU broadcast is srcB-only — and picks
    the regime: `col` / `row` / `scalar` / `leading` / `number`); `kernels/add_bcast_walk.hpp`, the run walk the
    reader and compute share (mirrored by `bcast_runs` in Python for ring sizing and tests); the output allocated
    with the broadcast shape (a's memory config, or b's when a broadcasts and is sharded, R9).
  - `leading`: reader remaps b's page over b's size-1 leading dims; compute is the flat helper call.
  - `col` / `row` / `scalar`: each b window is read once (explicit sync on b's ring) and held resident while
    every out tile that reuses it is added (FPU srcB `BroadcastDim::Col / Row / Scalar`); a and out sync in
    uniform caller-managed blocks of `block_tiles`; each run∩block piece is one chain call (`TileAddressing::Offset`,
    every lifecycle `None`), the first owning the setup and the rest `InitReconfigOwner::Caller`.
  - `number` (R7): the number's float32 bits are a compute runtime arg; `CopyTile(a)` + `FillBitcast` +
    `AddBinary` in the fp32 DEST, packed once — never rounded to bf16 first. Helper substitution vs the design's
    `unary<AddUnary>`: `add_unary_tile` is not built for Quasar (`binop_with_scalar.h`, Mul only in the Quasar
    LLK); an FPU scalar tile would hold the number at most as tf32, which misses 1 ulp on cancelling elements.
  - Thread plan: kept (1, 1, 1) (data-movement bound). `flat`, `leading`, `number` run at any plan (tested at
    (4, 4, 2) into NaN-prefilled outputs); the resident regimes are pinned to (1, 1, 1) by a host check — dealing
    a resident window to C > 1 NEOs needs the ALL access pattern, out of scope at C = 1.
- Accuracy achieved: ≤ 1 bf16 ulp of the exact sum rounded once on every shape tested; PCC=0.9999979–0.9999989,
  relative RMS=0.0014–0.0019 on [2,4,128,512]+[2,4,1,512], [4,8,32,256]+[4,8,32,1], [4,128,512]+[4,1,1],
  [2,512,1024]+[1,512,1024], [1,1,32,1]+[1,1,32,64], [4,8,32,256]+number, [1,1,17,50]+number (number and
  over_hw bit-exact; a number cancelling a to 1% stays ≤ 1 ulp).
- Golden test progress: targeted slices: all 416 cells of this refinement's region (bfloat16, TILE, fp32 DEST,
  broadcast ≠ none, every placement incl. L1 height/width sharded, memory_argument None/input, output_dtype
  None/bfloat16) pass, plus 40 sampled prior broadcast=none cells: 456/456. Full-suite count: the harness's run.
- Issues encountered: two Quasar mechanism facts, found by the stress tests (ranges crossing rows and images,
  turned knobs), shaped the resident regimes: (1) a ring window waited at once must not straddle the ring's end —
  row-blocked `PerBlockSize` chains and
  variable-size b windows read past the ring; fixed with uniform caller-managed a/out blocks, one-tile Col/Scalar
  windows and a never-wrapping Row ring. (2) Implicit-sync credits are posted per batch (half the ring), so a sparse
  b window could wait behind a reads the reader could not issue: deadlock; fixed by explicit sync on b's ring.
  Findings (not done here): b's total crossings are one per cluster that needs it, so `row` with one tile-row per
  cluster ([2,4,128,512]+[2,4,1,512]) reads b once per out page in total while no cluster re-reads a tile — an
  image-aligned split would cut it at the cost of occupancy; the reader barriers each Col b tile (one per row)
  before its a reads, which a one-window-ahead prefetch would hide. Gains in time are unmeasurable on ttsim.
- Tests added: `test_add_broadcast.py` — regime pinning for every broadcast value and both operand orders; the
  Done-when count (per cluster, b pages read == distinct b pages needed, for every broadcast INPUTS entry; runs
  cover the range; b's ring size); walk stress over min-tiles-per-core / window-cap / block / depth knobs into
  NaN-prefilled outputs; number exactness on cancellation; plans (4,4,2) for leading/number, refusal of C > 1 for
  resident regimes; refusal of two broadcasting operands; R9 placement of sharded broadcast results; one device
  session (tt-metal untilize before, every regime, `to_layout` after, add of an add). `test_add_plan.py`: the
  refusal test now refuses two broadcasting operands instead of a number / over_h b.
