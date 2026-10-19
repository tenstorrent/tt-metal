# combine_rf (Python ProgramDescriptor ops; not registered as ttnn.bringup.<name>)

- Source: mstaletovic/galaxy-unified-k3-v41-combine @ e4d239c6d81 (imported with flat_routed_expert_ttnn, synced in commit dc556fa6726); first imported @ f080c8df021 in 9003c0fd2ca.
- New ops, no TTNN source on this tree.

## Changes

imported, unchanged.

### Clear refusal when the accumulators' fp32 sums do not fit the arena (2026-10-19, deepseek_v41_flash_d_p P.8 follow-up)
- What: `sparse_combine_rf_prepare` with an arena raises ValueError when the arena cannot hold the accumulators' state,
  one ring slot per source, the CBs (the fp32 sums of ceil(S / A) home tokens) and the local ring. The message gives
  the sums per accumulator, the largest S per call at this A and the smallest A for this S. Before, the same case
  failed with "no room for one ring slot per source chip", which blamed the rings. Default behaviour unchanged: every
  configuration that built before builds the same program.
- Why: the "255-tile hang" (an accumulator holding more than ~255 fp32 tiles, S = 512 home tokens on 9 accumulators)
  was not an 8-bit limit. It was the local hand-off ring's grant batching, fixed in the entry below: grants went out
  in batches of 4, and a ring of fewer than 4 slots deadlocked. At S = 512 with a 1252 KB arena the ring had 2 slots;
  at S = 468 it had 9. With that fix reverted, S = 512 hangs at 2 slots (writers in se_yrm issue(), accumulators
  waiting on their local ring) and passes at 8 slots with the same 285 tiles. With the fix in place it passes at 2
  slots. The port readers' grants (batches of 8) also fire when the whole ring is granted, so they progress at any
  ring size (SPC_QP=4 passes). The real limit is L1 capacity: a home token's fp32 row stays in L1 for the whole
  call. At W 5120 that is 20 KB a token; with 9 accumulators of <= 1356 KB, one call takes <= 558 home tokens
  (310 tiles an accumulator). 1024 home tokens need >= 17 accumulators (1322 KB each), which is above the op's
  A <= 16.
- Results: deepseek_v41_flash_d_p/tt/perf_experts.py: test_weighted_combine_routes 30 / 30 (64-512 tokens, one pass,
  arenas auto / 1252 KB), test_weighted_refuses_small_arena (512 tokens in a 1156 KB arena: refused with "<= 468 home
  tokens or >= 10 accumulators"), test_weighted_passes_cost.
- Files: sparse_combine_rf.py

### Column-axis combine, fp32 output, passes, deterministic one-core planner (2026-10-19, deepseek_v41_flash_d_p P.8)
- What (every option off by default: the programs before, unchanged; one bug fix, identical for rings >= 4 slots):
  - `sparse_combine_rf_prepare(axis=0)`: the combine along each mesh column only (a model whose tokens are replicated
    over the mesh row: DeepSeek-V4.1's dispatch groups = mesh columns): a chip is chip r of its column's R-chip group
    (headers' home = r, rings R - 1, chip ids / credit ports / accumulator sources of its own column), Y ports only.
  - `out_dtype=ttnn.float32` (the accumulators write their fp32 sums; CB_OUT fp32 pages), `out_row0` (a pass's first
    output row: writer arg), `acc_unpack_fp32` (UnpackToDestFp32 on the accumulator CB for the final copy: the fp32
    sums no longer pass srcA at TF32), `zero_unseen` (a home token none of whose rows came writes zeros: with axis=0 a
    token whose experts all sit in other columns; its accumulator tiles were stale L1).
  - `accumulator_bytes(N, A, W, S, ...)`: a caller's L1 sizing of the accumulators (passes).
  - `combine_planner_prepare(axis=0, id_offset=, tokens=, token_base=, ids_uint32=, tile_in=)` (the one-core planner):
    plan as chip r of an R x 1 group, ids of another column skipped (id - id_offset outside [0, R EPC)), a pass of S
    tokens from token_base of `tokens` per home (token index = home x tokens + token), uint32 ids, ids / weights read
    as TILE [tokens, K] (no untilize / reshape: ttnn.reshape of a [2048, 6] row-major tensor to one page took ~1 ms),
    only the pass's tiles; `run.programs` / `run.cores` / `run.tensors` for the one-core planner too (several passes'
    planners merged into one program). `det=` now also with the one-core column planner: the order table from every
    source chip's flat schedule (se_dyn_from_counts), per (source, expert) entry lists. `plan_reference(id_offset=,
    token_base=)`: the host model of the column / pass options. `port_luts(axis=0)`, `axis_ports(chips, axis)`.
  - Fix: the accumulator granted its local L1 hand-off ring only in batches of 4 slots; a ring of 2-3 slots (what a
    near-full arena leaves) never reached a batch after its first grant and deadlocked with the flat y writers (hang:
    writers in se_yrm issue() waiting for grants, accumulators in take_local). Now a batch is min(4, ring slots).
- Why: deepseek_v41_flash_d_p P.8 (EXPERTS_COMBINE=weighted): the flat expert + this weighted combine in one program
  per chip on the model's layout (dispatch / combine along the column, column partials reduce-scattered over the row),
  fp32 per-token sums, bit-identical calls (the component tests' determinism check).
- Results (2x2 QuietBox, layer 0, chunk 2048, 1024 home tokens per chip in 2 passes of 512, 9 accumulators + 1 port
  cell, deepseek_v41_flash_d_p/tt/perf_experts.py::test_weighted_vs_separate): vs float64 on 128 tokens (NUMERICS=
  native, full precision, fp32 out) rel 0.0029, bias +2e-5 (separate path 0.0023 / +2e-5); deterministic: 0 of 10.5M
  values differ between calls (arrival order before: 149 values at ~1e-8); test_weighted_combine_routes (fake weights,
  local / remote / random routing, 64-512 tokens) pass; the frozen test_c_*_experts pass with it.
- Needed by: deepseek_v41_flash_d_p P.8 (tt/weighted_combine.py)
- Files: sparse_combine_rf.py, combine_plan.py
