# combine_rf (Python ProgramDescriptor ops; not registered as ttnn.bringup.<name>)

- Source: ttnn/ttnn/bringup/combine_rf on mstaletovic/combine-lean-sender-llk @ 856d97d5864 (three-way sync from
  e4d239c6d81, 2026-10-27, entry below); before: mstaletovic/galaxy-unified-k3-v41-combine @ e4d239c6d81 (imported with
  flat_routed_expert_ttnn, synced in commit dc556fa6726); first imported @ f080c8df021 in 9003c0fd2ca.
- New ops, no TTNN source on this tree.

## Changes

imported, unchanged.

### Sync with Marko @ 856d97d5864 (2026-10-27)
- What: a three-way merge of this folder, base e4d239c6d81 (our last sync; an ancestor of his branch), ours bab52a1a844,
  theirs origin/mstaletovic/combine-lean-sender-llk 856d97d5864. Every change of ours is kept (column-axis combine,
  fp32 output / out_row0 / acc_unpack_fp32 / zero_unseen, acc_dtype, accumulator_bytes and the arena refusal, the
  one-core column planner and the column multi-core planner, the rebase ports, tests/, examples/).
- Taken from Marko (his 38 combine_rf commits since e4d239c6d81):
  - sparse_combine_rf: the landing pool (each accumulator's depth x (N - 1) slots carved per call into one ring per
    source by demand, layouts forwarded by the home's ports; ccb61f5c92f), typed options instead of env knobs
    (inj_slots, ring_depth, pool_slots, local_keep, CombineProbes; 078879e6229, b10e0b79632), piggybacked credits
    (default on where they apply; off with deterministic and ports_per_direction; 251859a48d9, ee4f7c72cbe,
    50b36072a18), idle credit batching at >= 16 sources (c2e4fbbc4cd), shared landing with slot grants
    (landing="shared", 75916a7168b), snap_q (e94d370c790), shared_rows / dispatch / barrier / fill_sink and the
    probes (HW hang words, BK, FILLY), combine_l1_layout / combine_auto / fused_arena_reserve. His copy of our P.8
    ring-grant fix (the same `(QL < 4 ? QL : 4)`) merged into one.
  - combine_plan: the V2 multi-core planner's bucket / per-instance row sizing (16c0771dcff, eb4e1db54a9); the one-core
    and column planners are untouched by it.
  - l1_zero (l1_zero_kernels), sparse_all_gather_2d (zero kernels, dispatch_phase, memoized mesh_plan),
    dispatch_prep (folded / row-mate masks, own routing), layout_ops (src_tile_of gathers, persistent dst),
    router_topk (fold, split=None), fabric_all_gather_py: as on his branch.
  - New modules: routing_a2a.py, routing_pack.py, kv_push_plan.py, glu_act.py, FUSED_SHARED_EXPERT_PLAN.md (for his
    MoE block; DeepSeek-V4.1 does not use them).
  - `ttnn.freeze_program_hashes(desc)` where a module caches a descriptor (1bc1002c13d): combine_plan run(),
    l1_zero_program, layout_ops, dispatch_prep, sparse_all_gather_2d, fabric_all_gather_py, routing_a2a,
    routing_pack, glu_act. It only stores each program's structural hash in custom_program_hash (the same value
    generic_op would compute per call), so a reused descriptor is not re-hashed.
- Conflicts (git merge-file per file) and their resolution:
  - sparse_combine_rf.py, 5 hunks: (1) our _ACC_COMPUTE_WZ (zero_unseen) next to his state-layout comment: both kept,
    _ACC_COMPUTE_WZ now derives from his _ACC_COMPUTE_W (SNAP lines; its anchors still match, module asserts pass);
    (2) the local ring grant comment (same code both sides): his comment; (3) the signature: his new arguments, then
    ours (axis ... acc_dtype; every caller passes keywords); (4) the docstring: his text, then our column-option
    block; (5) the arena: our refusal (ValueError with the largest S / smallest A) kept, now with his `local_keep`
    in `keep` and stated as one landing slot per source, then his landing-pool sizing (PSL) instead of our fixed
    depth D.
  - layout_ops.py (4 hunks) and router_topk.py (1): ours was only the repo formatter's reflow; his side taken (it is
    black-clean).
  - Clean, both sides changed: combine_plan.py (our column planners, his MC2 sizing + freeze), l1_zero.py,
    fabric_all_gather_py.py, sparse_all_gather_2d.py (both carried the same rebase port); sparse_combine_l1.py
    identical on both sides. Only ours: sum_ops.py (formatting). Only his: dispatch_prep.py.
- Adjustments made in the merge (beyond picking sides):
  - sparse_combine_rf: his idle-credit-batching rule counts the combine group's chips (N; axis=0: the column) instead
    of the mesh's (R x C): identical for the whole mesh, and DeepSeek's 8-chip Galaxy column stays unbatched as before.
  - sparse_combine_rf axis=0 refuses shared_rows, dispatch, barrier and landing="shared" (whole-mesh options; their
    port / barrier wiring is not ported to the column group, and nothing tests them there).
  - sparse_all_gather_2d: the descriptor-cache hit called `ttnn.bringup.prepared_program.generic_op` (his fork, not on
    this tree; ImportError on the second call); it falls back to `ttnn.generic_op` on the same cached descriptor.
  - accumulator_bytes unchanged in form: the pool of depth x (N - 1) slots takes the L1 our fixed rings did, and its
    value equals his combine_l1_layout's acc_used (checked for N 2 / 4 / 8, A 9-16, W 5120-7168). His state block is
    larger (layouts, U0, piggyback words): +896 B a core at N 2, +2944 B at N 8 (A 9). One-pass capacities at a
    1356 KB arena are unchanged (bf16 / fp32: 1125 / 558 tokens at N 2 A 9, 1062 / 531 at N 8 A 9).
  - his `local_keep` defaults to 8 (was 2 slots for the L1 hand-off): deepseek_v41_flash_d_p tt/weighted_combine.py
    and tests/unit/test_acc_dtype.py pass local_keep=2, so the model's arena split and refusal limits stay as its
    pass sizing (accumulator_bytes, local_slots 2) assumes.
- What changes for DeepSeek-V4.1 (deterministic, ports_per_direction, axis=0, depth 4): its rings are carved from the
  pool per call (2x2: one source, all 4 slots as before; 8x4 column: up to 28 slots over 7 sources by demand
  instead of 4 each); piggyback, credit batching, shared landing and snap stay off; the deterministic order (so the
  sums) is unchanged. Not taken: generic_op holding the caller's descriptor by reference (1bc1002c13d, every generic_op
  caller), the prepared_program fork, vendored multicast helpers; examples/ left as they were (reference only).
- Core binding carried: `ttnn.freeze_program_hashes` only (from 1bc1002c13d), in its own files:
  ttnn/cpp/ttnn/operations/generic/freeze_program_hashes_nanobind.{hpp,cpp}, plus one line each in ttnn/sources.cmake,
  ttnn/ttnn/types.py and ttnn/ttnn/__init__.py, and the include + call in ttnn/cpp/ttnn-nanobind/__init__.cpp
  (beside bind_generic_operation, module ttnn._ttnn.operations.generic). It needs nothing else:
  generic_op already returns custom_program_hash when set, and the copy generic_op makes keeps it. No upstream PR.
- Why: keep this fork equal to its source (smaller future syncs); his landing pool keeps a Galaxy's ring L1
  independent of the source count.
- Results: not run here (no build, no device). Merged modules import with their module-level kernel-string asserts
  passing; black clean.
- Needed by: deepseek_v41_flash_d_p (expert ops sync, plan 2026-10-27 step 3)
- Files: every module above; tests/unit/test_acc_dtype.py (local_keep=2); README.md

### Column multi-core planner: combine_planner_prepare(axis=0, cores=...) (2026-10-20, deepseek_v41_flash_d_p P.11)
- What: the one-core planner's column / pass options (axis=0, id_offset, tokens / token_base, ids_uint32, tile_in,
  det) on several cores, a new kernel `_PLANNER_MCC` with the V2 multi-core planner's structure (Marko Staletovic's
  `_PLANNER_MC2`, whole-mesh only): M = N ceil(S / CT) cores (`_mcc_geometry`: chunks of 32-token multiples inside one
  home, <= the `cores` given), each reads its chunk's tiles (faces 0 / 2 of the TILE ids / weights), counting-sorts its
  local-expert rows into a bucket of final table words, sends counts (+ det: its rows per column expert) to all; expert
  owners total and broadcast each chunk's first row; every core pulls its 1/M of the row tiles from the buckets and
  writes them, and its share of the token index / table tail as zeros; n_inj / n_ret / n_local / want as atomics on
  core 0's plan; det: each chunk of this chip's home tokens computes its tokens' order lists itself (a row's place in
  its expert's list = the rows of the chunks before it, the source schedules from the totals; packed sums, per-expert
  entry lists), the accumulators' owners (core a % M) assemble the pages. Chunk order = the one-core planner's (home,
  token, k) scan order, so every output is bit-identical. `run.programs` / `run.cores` / `run.tensors` (+ dord) as the
  one-core column path; `run.mcc = (M, CT)`; `run.tdbg` = per-core phase timestamps. Default `cores=None` with axis=0:
  the one-core planner, its program unchanged; the whole-mesh planners (`_PLANNER`, `_PLANNER_MC`, `_PLANNER_MC2`)
  unchanged.
- Why: deepseek_v41_flash_d_p P.11 (owner): the one-core column planner took 1573 us of a ~3.4 ms experts call a layer
  at 1024 home tokens per chip (P.9, DS41-73).
- Results (2x2 QuietBox p300c): tests/unit/test_column_planner_mc.py 20 / 20 (12 skips by design: uint16 ids one pass,
  32 cores): every output word identical to the one-core planner's, both mesh columns, S 512 / 1024 one pass, 1024 in
  passes of 512 and 342, det and arrival order, 32 and 8 cores, uint32 / uint16 ids, uniform / hot / one-column
  routings, outputs pre-filled with garbage. Model (deepseek_v41_flash_d_p tt/perf_experts.py::test_combine_planner,
  layers 0 / 24, chunk 2048, profiler op mode): planner 1572.7 -> 68.8 us (32 cores x 64 tokens); experts ops 3427 ->
  1912 us (L00), 3426 -> 1927 us (L24); synced experts call 3.66-3.68 -> 2.10-2.16 ms; experts output 0 of 10.5M values
  differ. Whole fork suite with the option off: tests/unit 28 passed (test_column_planner 4/4, test_acc_dtype 4/4: the
  baseline) + the new file.
- Needed by: deepseek_v41_flash_d_p P.11 (tt/weighted_combine.py, setting COMBINE_PLANNER)
- Files: combine_plan.py, tests/unit/test_column_planner_mc.py

### acc_dtype: bf16 running sums, twice the home tokens a call (2026-10-20, deepseek_v41_flash_d_p P.9)
- What: `sparse_combine_rf_prepare(acc_dtype=ttnn.bfloat16)` keeps every home token's running sum in bf16 (CB_ACC
  Float16_b, 2048 B tiles) instead of fp32 (4096 B): the packer's L1 accumulation adds each weighted row (fp32 DEST,
  HiFi4 multiply unchanged) onto the bf16 sum. Same kernels, same summation order (deterministic stays bit-identical),
  same output formats. `accumulator_bytes(..., acc_dtype=)` and the arena refusal (ValueError numbers, "bf16 tiles")
  size with it; `acc_page(acc_dtype)`. Default None = ttnn.float32: every program before is unchanged (same CB format,
  sizes and kernel sources).
- Why: deepseek_v41_flash_d_p P.9 (owner, option B): with fp32 sums 9 accumulators hold <= 558 home tokens, so the
  model's 1024 home tokens per chip (2x2, chunk 2048) ran in 2 passes and the flat expert streamed every expert's
  weights twice a layer; with bf16 sums one pass holds <= 1143.
- Results (2x2 QuietBox p300c): tests/unit/test_acc_dtype.py 4 / 4: sizing (bf16 1143 vs fp32 567 tokens a call at 9
  accumulators, 1356 KB), refusal (1024 tokens in 1156 KB: bf16 "<= 963 home tokens or >= 10 accumulators", 963
  builds, 972 refused; fp32 "<= 477"), precision on the standalone combine (6 bf16 rows a token, 2^-6..2^6 scales, vs
  the float64 sum): fp32 rel 7e-9, bf16 rel 2.7e-3 (one rounding of the exact sum: 1.66e-3), bias -1.2e-5, worst row
  3.9e-3. Model (tt/perf_experts.py::test_combine_acc, layers 0 / 24): one pass; full precision vs float64 fp32 0.0029,
  bf16 0.0041, bf16 vs fp32 0.0029; 0 of 10.5M values differ between calls; experts call 5.74 -> 4.29 ms a layer
  (shipped glm precision).
- Needed by: deepseek_v41_flash_d_p P.9 (tt/weighted_combine.py, setting COMBINE_ACC)
- Files: sparse_combine_rf.py, tests/unit/test_acc_dtype.py

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

### Rebase onto llk_helper_library 6b80daf43e3: kernel API port (2026-10-27)
- What: the four inline fabric kernels include `hostdev/fabric_edm_packet_header.hpp` (#60172); `_chip_maps` passes
  `mesh_device` to `ttnn.cluster.get_chip_unique_id_from_fabric_node_id` (#60180). Host side, by the overseer:
  `ttnn.get_neighbor_eth_directions` takes `mesh_device` first (#59551).
- Why: kernel compile error (DeepSeek-V4.1's weighted combine included); TypeError in automatic core placement.
- Needed by: rebase 2026-10-27
- Files: `fabric_all_gather_py.py`, `sparse_all_gather_2d.py`, `sparse_combine_l1.py`, `sparse_combine_rf.py`
