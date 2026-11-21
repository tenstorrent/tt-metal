# Operation Design: paged_scaled_dot_product_attention_decode

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (two matmuls + online softmax + cross-core combine), one `ttnn.generic_op` dispatch |
| Goal | One decode step of attention over a paged K/V cache: every user's 32-row query tile attends positions `0..cur_pos[u]` of its own key/value heads, gathered through its page-table row. |
| Math | `out[0,u,h,:] = Σ_p softmax_p(scale·q[0,u,h,:]·key(u,h//g,p) + mask[u,p]) · value(u,h//g,p)`, `key(u,c,p) = K[page_table[u, p//block_size], c, p%block_size, :]`, `mask = 0 (p ≤ cur_pos[u]) / excluded (p > cur_pos[u])`, `g = NH//NKV` |
| References | Spec (authoritative): `eval/specs/paged_scaled_dot_product_attention_decode/spec.py`. Target: `.claude/references/targets/quasar.md`. Backend: `.claude/references/backends/ttsim.md`. Host program: `.claude/references/quasar_op_examples/` (`toy_row_passes` = per-NEO units, padded rounds, compute-owned rings, `pack_init`; `toy_thread_plan/kernels/reader_mixed.cpp` = both ring protocols in one kernel). Semaphore / remote-write host pattern (not Quasar-validated, read for API shape only): `ttnn/ttnn/operations/toy_variance/toy_variance_sharded_program_artifacts.py`. Rings: `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A. Blocking: `.claude/references/blocking-model.md`. Ledger: `l1_ledger.md` (this directory). |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (probe on this backend: `compute_with_storage_grid_size()` = 8-4). Op code queries `device.compute_with_storage_grid_size()` | target page + probe |
| L1 per core | 4,194,304 B per cluster, shared by its 4 NEOs and 8 DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 − 65,536 = 3,897,792 B on this target (query value from a probe on this backend) | probe |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.ProgramSpec` / `KernelSpec` / `DataflowBufferSpec` / `SemaphoreSpec` / `ScratchpadSpec` / `WorkUnitSpec`, dispatched by `ttnn.generic_op`) — `.claude/references/quasar_op_examples/` | target page |
| Threads per core | C ∈ {1,2,4} NEOs (one compute thread each); R + W ≤ 6 user DM harts; on STRIDED rings R | C and W | C. 32 DFBs per cluster, ≤ 24 active per NEO, ring ≤ 1,048,560 B; 16 semaphores per core; kernel binary ≤ 48 KiB DM / 24 KiB compute | target page |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / legacy device-op program factory — refused at kernel creation. (2) `bfloat8_b`, `bfloat4_b` — no path. (3) `matmul_block_helpers.hpp` (and `matmul_block` helper) — not used: the matmuls are written with LLK calls (`matmul_tiles`). (4) `reduce<>` `ReduceFp32Mode::Accurate` — static_assert (`reduce_helpers_compute.inl:76-93`). (5) `reduce<MAX, REDUCE_ROW>` with `Accumulate` — static_assert (`reduce_helpers_compute.inl:797-806`). (6) reduce helper indexed policies (`WaitUpfrontNoPop`, `BulkWaitBulkPop`) under C > 1 — not used: indexed reads pass `tile_index · ring.get_consumer_stride_tiles()`. (7) raw `matmul_block` with ct/rt/kt > 1 (internal contiguous stepping) under C > 1 — not used: matmuls are issued one tile at a time (`matmul_tiles`). (8) `eltwise/binary/sfpu/minmax.hpp` (`BinaryMax` chain struct) — not used: the max is the raw `binary_max_tile`. (9) `post_reduce_op` — not used: the recip is its own chain. (10) `UnpackToDest` — not used. (11) uint16/uint32 DFB formats — refused (`program_spec/validation/resource/dfb.cpp:148`). (12) `exp_tile` `scale_en`/`scale`/clamping other than defaults — static_assert (`ckernel_sfpu_exp.h:334-355`). (13) `sfpu_*_bcast_col` with 16-bit DEST (needs fp32 DEST, `sfpu_binary_bcast.h:58-60`). (14) `SemaphoreSpec` initial value ≠ 0 — refused (`resource/semaphore.cpp:19-24`). (15) A DM kernel both producer and consumer of one DFB — refused (`resource/dfb.cpp:646-651`). (16) compute `fpu_math_fidelity` — set on every compute KernelSpec. (17) `ttnn.Mcast` with a multi-threaded kernel (`mcast_spec_adapter.cpp:125-126`) | target page + headers |

Every worked number below is for this target (32 clusters, 3,897,792 B budget). Every run-time predicate uses the queries.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor_q` | Tensor | yes | [1, B, NH, D], NH ≤ 32 (X8), D % 32 == 0 | — | tensor |
| `input_tensor_k`, `input_tensor_v` | Tensor | yes | [blocks, NKV, block_size, D]; equal shapes (X6); same D as q (X7); block_size % 32 == 0 (X9); NH % NKV == 0 (X5); DRAM interleaved, TILE, q's dtype (R13) | — | tensor |
| `page_table_tensor` | Tensor | yes | [B, P] int32 ROW_MAJOR DRAM interleaved; rows == B (X3) | — | tensor |
| `cur_pos_tensor` | Tensor (kw-only) | yes | [B] int32 ROW_MAJOR DRAM interleaved; len == B (X4). Values NOT checked (R6) — clamped on device | — | tensor |
| `scale` | float | no | any float (sign included) | `1/sqrt(D)` (R4) | common RT (fp32 bits) |
| `sliding_window_size` | int | no | None only (R15 → NotImplementedError, X1) | None | host |
| `memory_config` | MemoryConfig | no | DRAM interleaved (`canonical_memory` → `output_memory` axis, R14); non-config object → TypeError (X10) | None ≡ DRAM | host |
| `program_config` | SDPAProgramConfig / None | no | anything; accepted and not read (R11) | None | — |
| `compute_kernel_config` | ComputeConfigDescriptor / WormholeComputeKernelConfig | no | fields `math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode` read as attributes (R10); float32 + `fp32_dest_acc_en=False` refused (R7) | `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, math_approx_mode=False (R8) | CT (kernel config) |

`validate()` gates on the spec's TARGET axes by their names: `dtype`, `fp32_dest_acc_en`, `layout`, `memory_config` + `sharding` (from `eval.memory.memory_axes(input_tensor_q)`, R12), `output_memory` (`canonical_memory`), and `INPUT_TAGGERS = AXES` (`alignment`, `kv_heads_mode`, `batch`) imported from the spec, never redefined. Phase 0 `SUPPORTED` = spec `PHASE0`. Every refusal outside SUPPORTED is `NotImplementedError` (R19). REJECTIONS X1–X10 are checked before the support gate.

## Tensors

### Input

| Property | q | K / V caches | page table | cur_pos |
|----------|---|--------------|------------|---------|
| Shape | [1, B, NH, D] | [blocks, NKV, block_size, D] | [B, P] | [B] |
| Dtype | bfloat16 (Phase 0) | = q | int32 | int32 |
| Layout | TILE | TILE | ROW_MAJOR | ROW_MAJOR |
| Memory | L1 height-sharded (Phase 0); read through its `TensorAccessor`, so any shard grid / interleaved works (R12) | DRAM interleaved | DRAM interleaved (one page = one user row) | DRAM interleaved (one page = all B) |

Tile geometry (alignment-aware, per image): `tensor_d_tiles = ceil(D/32)`; user u's q tile row = pages `u·tensor_d_tiles + j`; `block_pos_tiles = block_size/32` (X9 makes it exact); `tensor_pos_tiles = P · block_pos_tiles`; K/V page of (block b, head c, block-local tile-row r, d-tile j) = `((b·NKV + c)·block_pos_tiles + r)·tensor_d_tiles + j`; position tile t of user u, head c → `b = page_table[u, t // block_pos_tiles]`, `r = t % block_pos_tiles`.

### Output

| Property | Value |
|----------|-------|
| Shape | = q logical shape [1, B, NH, D] (rows ≥ NH of the tile hold anything, R17) |
| Dtype | = q |
| Layout | TILE |
| Memory | DRAM interleaved: `ttnn.allocate_tensor_on_device(TensorSpec(q.shape, q.dtype, TILE, DRAM_MEMORY_CONFIG), device)` (R14) |

Scratch (allocation, not a dispatch): `partials` = **L1 interleaved** TILE tensor (knob `partials_memory`, Phase 0 `ttnn.L1_MEMORY_CONFIG`; `ttnn.DRAM_MEMORY_CONFIG` is the fallback the L1 predicate selects when its per-bank share does not fit) `[1, 1, B·max_workers_per_user·32, (tensor_d_tiles+2)·32]`, dtype `float32` when `fp32_dest_acc_en` else `bfloat16`. Page `slot·(tensor_d_tiles+2) + i`, slot = `u·max_workers_per_user + w`; i < tensor_d_tiles: unnormalized O tile; i = tensor_d_tiles: row max m; i = tensor_d_tiles+1: row sum l.

## Blocking Model

The op is a parameterized flash-decode. **The scheme:** for each user, the key/value heads' position tiles are concatenated into one *virtual key sequence* of `N_u = NKV · user_pos_tiles(u)` units (unit v = head `c = v // user_pos_tiles`, tile `t = v % user_pos_tiles`). Every query row uses only its own head's units: a unit of head c adds `NEG_BIG` to the scores of rows outside group c, so they contribute exactly 0 after the softmax (the heads' rows share one output tile, so the head axis is folded into the sequence instead of being written as sub-tile rows). The virtual sequence is split over `W_active` workers (a worker = one NEO of one cluster); each runs an online softmax over its contiguous range and emits a partial (O unnormalized, m, l); the user's combiner cluster merges the partials (the merge is associative and commutative) and writes the output.

`user_pos_tiles(u) = clamp(ceil((cur_pos[u]+1)/32), 1, tensor_pos_tiles)` is known **only on the device** (R6: the op does not read device tensors on the host), so the split is evaluated on the device from `cur_pos` by one shared function (`decode_plan.hpp`, used by reader, compute and writer — single source); the host sizes everything for the maximum `N_max = NKV · tensor_pos_tiles`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| user (B) | independent — R5: user u's result reads only user u's row of q / page table / cur_pos | `users_per_block` | 1 — no operand is shared across users (each has its own page-table row, so K/V tiles differ), so a multi-user block amortizes nothing but a 4-byte position read and multiplies the per-NEO state; a user's own work is already spread over `G_u·C` workers | host constant | users dealt to cluster groups: `B ≤ floor(grid/G_u)` → one group of `G_u` clusters per user; else `G_u = 1` and `ttnn.split_work_to_cores(grid, B)` deals users to clusters (user rounds) | knob-turn |
| kv_head (NKV) | independent in the math (R3), but all heads' rows share one output tile per d-tile → folded into the virtual sequence as **head segments** (row-masked), so it inherits the sequence's dependent character for assignment | segment = the run of a worker's range inside one head (bounded by `k_chunk_tiles` per chunk) | whole run | `decode_plan.hpp` (derived) | part of the virtual-sequence split | head-aligned split (regime table, deferred) |
| position tile (`user_pos_tiles`, dynamic ≤ `tensor_pos_tiles`) | dependent — softmax and the weighted sum span positions | `k_chunk_tiles` (one online-softmax step) | `min(tensor_pos_tiles, ceil(N_max / max_workers_per_user))`, clamped by the L1 predicate: LLAMA_CAPTURED 32, PAIRS_OF_HEADS 4, BLOCKS_OF_TWO_TILES 6, QWEN_TWO_KV_HEADS 4, QWEN_ONE/LLAMA_SPLIT 2, ONE_TILE_WIDE 1 (= the whole per-worker assignment in every Phase 0 cell) | host function `k_chunk_tiles(...)` → CT arg | virtual sequence split over `W_active = G_active·C` workers (contiguous balanced ranges), merged by the combiner | knob-turn (bigger chunk, more workers); tree combine = scheme-change (deferred) |
| head_dim D (`tensor_d_tiles`) | dependent in QK (contracted, accumulated in DEST over d-tiles); independent in PV / O | `o_block_tiles` (d-extent of the O state per pass) | `tensor_d_tiles` (whole; DEST sub-blocks are internal) | host constant derived = `tensor_d_tiles` | phase 1: not split (QK needs all of D); phase 2 (combine): d-tiles dealt to the combiner cluster's C NEOs, `d_share_tiles = ceil(tensor_d_tiles / C)` each, short last round padded | wide-D regime (deferred) |
| query-head rows (NH ≤ 32) | independent, but they are the 32 rows of one tile — not a blocking axis | — | the whole tile | X8 | rows never split | — |
| partial (phase 2, `W_active` per user) | dependent — merged into one result | `partials_per_merge` | 1 — each merge step costs a fixed (stats + d_share) per partial; a p-partial block would need p stat tiles resident and saves only p−1 rescales of a `d_share_tiles` state (perf lamp "batched merge") | CT constant | merged by the C NEOs of the combiner cluster (each its d-share) | knob-turn |

**Stages re-blocked.** Scores / probabilities of a chunk `[32 × k_chunk_tiles·32]` have the position-tile axis (block = `k_chunk_tiles`, held for the chunk) and the row axis (one tile). The partials `[W_active × (tensor_d_tiles+2)]` have the partial axis (dependent, merged sequentially) and the d axis (independent) — **assigned across the combiner cluster's C NEOs**, so the combine does not sit on one NEO while three idle. Clusters other than the combiner idle during phase 2; the √N balance rule below sizes `W_active` so phase 2 (≈ W partial merges) and phase 1 (≈ N/W units per worker) cost the same order — this is the communication-role cost of the combiner made explicit, not left implicit.

**Worker-count rule (single source `decode_plan.hpp`, mirrored once in the host for `N_max`).** `clusters_for(N) = clamp(ceil(isqrt_ceil(ceil(N / merge_cost_units)) / C), 1, G_u)`: total time ∝ N/W (phase 1) + `merge_cost_units`·W (phase 2), minimized at W = √(N/κ). Host: `G_u = clamp(clusters_for(N_max), 1, max(1, floor(num_clusters / B)))`, `max_workers_per_user = G_u·C`. Device: `G_active(u) = min(G_u, clusters_for(N_u))`; clusters of the group with `rank ≥ G_active(u)` skip user u entirely (no ring traffic on any kernel). Worker w = `rank·C + neo` gets `base + (w < rem)` units starting at `w·base + min(w, rem)`, `base = N_u // W_active`, `rem = N_u % W_active`; a worker with 0 units emits a neutral partial (O = 0, l = 0, m = `NEG_EMPTY`).

Phase 0 values on this target (C = 4, κ = 1, B = 1): LLAMA_CAPTURED N_max = 1024 → W* = 32 → **G_u = 8 clusters**, 32 workers × 32 units at cur_pos = 4095; PAIRS_OF_HEADS 64 → 2 clusters; BLOCKS_OF_TWO_TILES 48 → 2 clusters; QWEN_TWO_KV_HEADS 16 → 1; QWEN_ONE_KV_HEAD / LLAMA_SPLIT 8 → 1; ONE_TILE_WIDE 2 → 1 (2 workers busy, 2 neutral).

**A worker with 0 units** (small N) reads no q, masks, K or V and skips phase 1; it emits the neutral partial so every writer rotation stays balanced.

**Chunking inside a worker.** The range is cut at head boundaries (a chunk never straddles heads, so one mask tile serves a chunk's body tiles) and into chunks of `n ≤ k_chunk_tiles` tiles. The head-last tile (t = `user_pos_tiles−1`) is always the last tile of its chunk and gets the position mask.

**Every knob is a parameter.** `k_chunk_tiles`, `kv_depth_tiles`, `mask_depth`, `partial_depth`, `merge_cost_units`, `G_u` (with the test override `_MAX_CLUSTERS_PER_USER`), `partials_memory`, `THREAD_PLAN`, `l1_reserve_bytes` are each one host constant / host function passed as a CT or RT arg; every page count, loop trip count and slot index is derived from them. Kernels loop `for (chunk_idx = 0; chunk_idx < num_chunks_this_worker; ++chunk_idx)` with the extent from `decode_plan.hpp`, never over an inlined size.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_k_tiles`, `cb_v_tiles` | `kv_depth_tiles` (per NEO) | `2·tensor_d_tiles` (two position tile-rows in flight; 4–8 tiles for D = 64–128) | reader keeps one tile-row of reads in flight while compute consumes the other (catalog `double_buffer`: 4–8 reads before a barrier is the knee) |
| `cb_score_masks` | `mask_depth` | 4 = 2 segments × 2 nominal masks | reader builds the next segment's masks without stalling its K/V reads |
| `cb_partials_in` | `partial_depth` | `2·(2 + d_share_tiles)` (two partials) | combiner reader prefetches the next partial during a merge |
| `cb_q_tiles`, `cb_output_tiles` | fixed 2 | 2 | streaming, front-only |
| compute-owned chunk rings (`cb_scores`, `cb_probs`) | 1 chunk | `k_chunk_tiles` | none needed: produced and consumed by the same NEO in program order |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Indexed AB unpack under C > 1 (`llk_unpack_AB_matmul_api.h:192-194`, `llk_unpack_AB_reduce_api.h:122-125`) | every indexed tile read in a matmul / reduce | kernel passes `tile_index · ring.get_consumer_stride_tiles()` (`dataflow_buffer.h:240`); matmuls issued one tile at a time (`matmul_tiles`) so no LLK-internal contiguous stepping | — |
| Indexed access must not straddle a ring's end | blocks read by index (`cb_q_scaled`, `cb_scores`, `cb_probs`, `cb_score_masks`) | **nominal block push/pop**: `cb_q_scaled` exactly `tensor_d_tiles` per user; `cb_score_masks` exactly 2 per segment (index 0 body, 1 last); `cb_scores` / `cb_probs` reserve + push and wait + pop exactly `k_chunk_tiles` per chunk, the work narrows to the chunk's `n` valid tiles, pad entries are never read; capacity = the nominal block | a ragged chunk starts the next block mid-ring and an index wraps past the ring end |
| DEST capacity: score tiles of one matmul pass / O tiles accumulated across a chunk's t-loop | QK pass width `s_s`; PV output sub-block `s_o` | `s_s, s_o ≤ DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`; 4 with fp32 DEST half-sync, 8 with 16-bit); `s_o = min(o_block_tiles, DEST_AUTO_LIMIT)`; the V reader order is (o-sub-block, t, j ∈ sub-block) so V stays front-only | DEST overflow: corrupted scores / O |
| Writer hart serves C/W = 2 NEOs in strict rotation | per-NEO entry count on `cb_partial_out`, `cb_output_tiles` | every active NEO pushes exactly `tensor_d_tiles+2` partial tiles; every combiner NEO pushes exactly `d_share_tiles` output tiles (short last round: NEO recomputes d-tile `q·C` and the writer stores it again, same bytes) | writer blocks on one NEO forever (hang) |
| Reader streams differ per NEO (different unit counts, combiner-only partial reads) | per-NEO entry count on reader rings | **R = C** in the plan so each reader hart feeds exactly one NEO (STRIDED P = C: one tile counter per producer/consumer pair) | rotation across NEOs with unequal counts hangs |
| One output tile holds all heads of a user | NH | NH ≤ 32 (X8) | — refused |
| Block of the cache is whole tiles | block_size | `block_size % 32 == 0` (X9) | — refused |
| Device-tensor values unchecked (R6) | `cur_pos`, page-table entries used as indices | device clamps `cur_pos` to `[0, tensor_pos_tiles·32−1]` and each page-table entry to `[0, blocks−1]` before forming a page id | NoC read outside K/V → hang / fault ("nothing else" in R6 forbids that) |
| Finite masks | `NEG_BIG` (score mask) and `NEG_EMPTY` (neutral partial m) | `NEG_BIG = −1e9`, `NEG_EMPTY = −1e30`, both one constant in `decode_plan.hpp`; never −inf | −inf makes an all-masked row `exp(−inf+inf) = NaN` (other-head rows in a head segment, padding rows) |
| Matmul source formats must match (`matmul.h:99-102`, issue #46769) | page format of `cb_q_scaled`, `cb_probs` | both = input dtype format (Float16_b in Phase 0), like `cb_k_tiles` / `cb_v_tiles` | asymmetric src tile sizes not re-programmed by `matmul_init` |
| L1 budget | `k_chunk_tiles`, `tensor_d_tiles` | host predicate `C · per_neo_bytes(k_chunk_tiles, tensor_d_tiles) + scratchpads + partials_bank_share ≤ budget` (`l1_ledger.md`; `partials_bank_share = ceil(partials_bytes / num_l1_banks)` when `partials_memory = L1`, else 0 — if it does not fit, `partials_memory` falls back to DRAM first); `k_chunk_tiles` lowered to the largest value that fits; if `k_chunk_tiles = 1` does not fit → `NotImplementedError` naming the wide-D regime | allocation failure at launch |
| Semaphores: 16 per core, initial value 0 only | combine signalling | one semaphore `sem_partials_ready`; consumed back to 0 by the readers' `down` (see Dataflow) | stale count on the next launch |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM; minimum = each attended K/V tile once, q once per user, output once) | What a bigger block buys |
|--------|--------|-----------|-------|----------------------------------------------------------|--------------------------|
| **R-split: virtual-sequence flash-decode, √N workers, flat combine** | **built** | always (covers every TARGET shape whose per-NEO footprint fits; `G_active = 1` is the same regime on one cluster) | chunk = `k_chunk_tiles` position tiles × all D of one head segment; partial = `tensor_d_tiles+2` tiles | DRAM: minimum — each attended K/V tile once (tiles 0..`user_pos_tiles−1` of each head; up to 31 positions past cur_pos in the last tile), output once. Cross-core only: q read `W_active` times from its shard (reuse-shared); partials `W_active·(tensor_d_tiles+2)` tiles written to + `C·W_active·(2 + d_share_tiles)` tiles read from the L1-interleaved scratch | per chunk: one stats step (max fold, 2 exp tiles), one α-rescale of the `tensor_d_tiles`-tile O state, the matmul / reduce / exp re-inits between phases, ring handshakes — paid `ceil(units/k_chunk_tiles)` times per worker (intended: once per chunk) |
| R-tree: two-level combine (cluster-local merge, then group merge) | deferred — positive reason: the largest `N_max` in INPUTS is 1024, where the flat √N rule gives `W ≤ 32` and phase 2 ≈ phase 1; a tree only pays above W ≈ 32. Reachable: the merge op and the scratch-slot + semaphore contract are the level-2 contract; a tree adds one level of combiners that write a partial instead of the output | `N_max > 1024` | — | partial reads drop from W·(2C+…) to ≈ (C + G)·(…) | fewer rescales at the root |
| R-head-aligned: whole heads per worker, no score row-mask | deferred — positive reason: the row mask costs ≤ 1 tile-op per unit against ≥ 2·`tensor_d_tiles`+3 per unit, and R-split covers every shape; head alignment needs one partial per head segment (variable count per NEO, which breaks writer rotation). Reachable: masks are already per-segment and the combiner already merges masked rows | `NKV ≥ W_active` and `user_pos_tiles ≥ k_chunk_tiles` | — | same | drops one add per score tile |
| R-q-reuse: q read once per cluster (ALL-consumer ring to 4 NEOs) or multicast to the group | deferred — positive reason: q traffic is `W_active·tensor_d_tiles` tiles against `2·N_u·tensor_d_tiles` for K/V, i.e. `W/(2N) = 1/(2√N)` of it (≤ 1/6 for N ≥ 9), and from L1 not DRAM. Reachable: `cb_q_tiles` becomes an ALL-pattern ring fed by one hart (DataflowBuffer.md "ALL") or an `ttnn.Mcast` channel; compute is unchanged | — | — | removes `(C−1)/C` of q reads | — |
| R-q-in-place: the cluster holding user u's q shard consumes it zero-copy (`borrowed_from`) | deferred — positive reason: the workers are not the shard's core (one shard on core (0,0) for all of a user's ≥ 4 workers), R12 forbids depending on which cores hold the shard, and the saving is ≤ `tensor_d_tiles` tiles per cluster | shard core ∈ group | — | — | — |
| R-wide-D: O state blocked along D (`o_block_tiles < tensor_d_tiles`), QK recomputed per d-block or d split over NEOs | deferred — positive reason: only the batched D = 4096 entry (not Phase 0) needs it; Phase 0 D ≤ 128. Reachable: `o_block_tiles` is already a named knob and PV is already walked in d-sub-blocks | per-NEO footprint with `k_chunk_tiles = 1` > budget/C | `o_block_tiles` | QK repeated `ceil(tensor_d_tiles/o_block_tiles)` times | — |
| R-single-NEO: one worker per user, no combine | rejected — superseded by R-split, which at `W = 1` is this regime; with C = 4 NEOs per cluster it would leave 3 NEOs and 31 clusters idle for every traced (B = 1) call | — | — | minimum | — |
| R-subtile-rows: each head's worker writes its g rows of the output tile directly | rejected — a row of a tile is 2 × 32 B face rows in bf16; for g = 1 the writes are 32 B at 32 B offsets, below the 64 B DRAM alignment, and two workers would write one tile concurrently. Superseded by the row-masked virtual sequence | — | — | — | — |
| R-resident-KV: K/V of a worker's range held in L1 | rejected — nothing to save: R-split already reads each K/V tile once (the minimum); residency only adds L1 | — | — | — | — |
| R-d-split-QK: split D across clusters | rejected (dead end) — QK contracts over D, so each cluster would re-read every K tile: K crosses DRAM `G` times | — | — | K × G | — |

### Traffic ranking

Per user, `N = N_u` units, `T = tensor_d_tiles`, tile = 2 KB (bf16) / 4 KB (fp32 partials):

| Candidate split | DRAM crossings | Cross-core | Rank |
|-----------------|----------------|-----------|------|
| users only (one cluster per user) | K, V once (2NT tiles) | q once | cheapest bytes, but B = 1 in every traced call → 1 of 32 clusters, and its 4 NEOs still need a combine |
| **virtual sequence across `W = G·C` workers + flat combine (built)** | K, V once (2NT) — the DRAM minimum | q W× (L1 shard → NoC); partials W(T+2) written + C·W(2+d_share) read in the L1 scratch; 1 semaphore inc per writer hart | **cheapest-traffic split that fills the grid**: same DRAM bytes as users-only, up to 8 clusters × 4 NEOs engaged |
| same, partials in DRAM (`partials_memory = DRAM`) | K, V once + partials W(T+2) + C·W(2+d_share) through DRAM (LLAMA_CAPTURED fp32: 2 MiB on top of 8 MiB) | q W× | fallback only |
| heads only, sub-tile writes | K, V once | — | rejected (alignment) |
| D across clusters | K G times | — | rejected (dead end) |

Chosen: the virtual-sequence split with L1 partials — DRAM bytes at the minimum, and the only split that engages more than one cluster for a single user.

### Block schedule

```cpp
for (uint32_t user_round = 0; user_round < num_user_rounds_this_cluster; ++user_round) {
    load_position_block(user_round);          // cur_pos[u] → plan (every kernel); inactive cluster: continue
    load_q_block(); scale_q_block();          // q' = q · scale, Dt tiles, resident for the user
    for (uint32_t seg_idx = 0; seg_idx < num_segments_this_worker; ++seg_idx) {
        load_mask_block(seg_idx);             // body mask (row groups) + last-tile mask, resident for the segment
        for (uint32_t chunk_idx = 0; chunk_idx < num_chunks_in_segment; ++chunk_idx) {
            load_k_block(); score_block();    // S = q'·Kᵀ + mask, k_chunk_tiles nominal
            rowmax_block(); stats_block();    // m_chunk; m_new, α  (first chunk: m_run = m_chunk)
            exp_block();                      // P = exp(S − m_new)
            load_v_block(); pv_block();       // O_chunk = P·V ; l_chunk = rowsum(P)
            merge_block();                    // l_run = α·l_run + l_chunk ; O_run = α·O_run + O_chunk
        }
    }
    emit_partial_block(); store_partial_block();   // + semaphore signal to the combiner
    if (cluster_rank == 0) {                       // combiner cluster, all C NEOs, each its d-share
        wait_partials_block();
        for (uint32_t partial_idx = 0; partial_idx < W_active; ++partial_idx) {
            load_partial_block(partial_idx); merge_partial_block();
        }
        normalize_block(); store_output_block();
    }
}
```

| Operation | Block shape | Resident across it | Intended frequency of inits / barriers / signals |
|-----------|-------------|--------------------|---------------------------------------------------|
| load_position_block | 1 int32 + 1 page-table row | — | 2 tagged reads + 1 barrier per user per hart |
| load_q_block / scale_q_block | `tensor_d_tiles` | `cb_q_scaled` for the whole user | one SFPU scalar-mul init per user |
| load_mask_block | 2 tiles nominal (body, last; an entry the segment does not need is pushed unwritten and never read) | for the segment | built in L1 by the reader hart once per segment |
| score_block | `[1 × n]` score tiles, contraction `tensor_d_tiles` | `cb_q_scaled`, masks | one `matmul_init(transpose=1)` per chunk; DEST passes of `s_s` tiles |
| rowmax_block | `[1 × n]` → 1 | `cb_scores` (not popped) | one `reduce_init<MAX,ROW>` per chunk |
| stats_block | 2 → 2 (col-vector tiles) | — | one DEST section per chunk (copy ×2, max, sub, exp, 2 packs) |
| exp_block | `[1 × n]` | `cb_max_run` (not popped) | one sub-bcast-col + exp init per chunk; pops `cb_scores` nominal block |
| pv_block | `[1 × n] · [n × tensor_d_tiles]` → `[1 × tensor_d_tiles]` + rowsum | `cb_probs` across d-sub-blocks | one `matmul_init` + one `reduce_init<SUM,ROW>` per chunk |
| merge_block | 1 + `tensor_d_tiles` tiles | `cb_alpha` | one chain init per chunk |
| emit_partial / store_partial | `tensor_d_tiles+2` | — | once per worker per user; one barrier + one semaphore inc per writer hart |
| wait_partials | — | — | once per user per combiner reader hart |
| merge_partial_block | 2 + `d_share_tiles` per partial | state (`cb_max_run`, `cb_sum_run`, `cb_out_acc`) | one stats DEST section + 2 chains per partial |
| normalize / store_output | `d_share_tiles` | — | once per user per combiner NEO |

**Stall shadow.** (1) The combiner waits for every partial: its own phase-1 range is scheduled before the wait (built), and its own NEOs' partials are merged with everyone else's. The merge is associative and commutative, so merging partials in arrival order instead of slot order would fill the wait — it reorders floating-point sums, so it is a perf lamp gated on the precision baseline. (2) In `load_position_block` the q read and the page-table read do not depend on `cur_pos`: each reader hart issues the cur_pos read, the page-table row read and the q reads back to back and barriers once. (3) First K reads depend on the page-table values: nothing else in the reader is independent of them except q (already in flight).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (`merge_cost_units` κ = 1) | κ is a paper estimate (merge ≈ Dt+6 tile-ops vs unit ≈ 2Dt+4); the √N rule may over- or under-engage clusters | κ ∈ {0.5, 2, 4}; `G_u` forced to 1 (host override knob `_MAX_CLUSTERS_PER_USER`) |
| Overlap (`k_chunk_tiles` = whole per-worker assignment) | one big chunk serializes QK → max → exp → PV inside a NEO; a smaller chunk lets V reads of chunk i overlap scores of chunk i+1 | `k_chunk_tiles` halved / quartered at LLAMA_CAPTURED |
| Thread plan (4,4,2) | decided from the bound, not measured (no timing model on ttsim) | (1,1,1) and (4,4,2) on `ttsim-perf`, one constant `THREAD_PLAN` |
| Arrival-order merge | combiner waits for the slowest worker before merging anything | per-slot ready flags (slot-written word + `wait_min`) and merge in arrival order; precision gate |
| α·O folded into DEST before PV | merge packs O_chunk then re-reads it | `mul_tiles_bcast_cols(O_run, α)` into DEST, then PV `matmul_tiles` accumulates on top (`eltwise_l1_vs_dest_accumulate`), drops `cb_out_step` |
| Partial scratch placement | L1-interleaved spreads a slot's pages over all banks, so the combiner gathers from many cores | combiner-local slots (workers write straight into the combiner cluster's L1 slot array, one hop) vs DRAM (`partials_memory`) |
| Pack-into-destination on the last chunk | `emit_partial` copies `tensor_d_tiles+2` tiles | last merge packs straight into `cb_partial_out` |
| Stats on one column | stats tiles are column vectors but ops run on the whole tile | `VectorMode` limited to column 0 for recip / exp / max (`sfpu_tile_scope`) |
| fp32 matmul operands | `cb_q_scaled`, `cb_probs` are bf16 to keep matmul sources symmetric | Float32 for both with K/V reconfig, after #46769 |
| Row sum via ones-matmul | reduce SUM accumulates through tf32 | `l_chunk = P · ONES` as an extra PV output column |

### Catalog evidence (`ttnn/ttnn/operations/examples/master.md`; all measured on Wormhole / Blackhole, none on Quasar)

| Entry | Informed | How |
|-------|----------|-----|
| `double_buffer` | `kv_depth_tiles` | 4–8 reads in flight before a barrier is the knee → two tile-rows of K/V (4–8 tiles for D = 64–128) |
| `compute_block_size` | `k_chunk_tiles`, chain fusion | fixed per-pass cost → one chunk = the whole per-worker assignment; sub + exp fused in one chain; no reconfig where the format is constant |
| `compute_fusion` | stats_block | SFPU follow-ups (max, sub, exp) stay in one DEST section |
| `tensix_all_reduce` | combine shape | tiny payload → root combine; tree only on a busy grid (R-tree deferred) |
| `shared_input_reuse`, `mcast_topology` | R-q-reuse (deferred) | multicast win far below the read cut; q is ≤ 1/(2√N) of K/V bytes |
| `noc_placement` | Work Distribution | engaged clusters taken row-wise |
| `eltwise_l1_vs_dest_accumulate` | perf lamp "α·O folded into DEST" | DEST accumulation beats re-reading a partial |
| `sfpu_tile_scope` | perf lamp "stats on one column" | recip / exp on column-vector tiles |
| `matmul_output_subblock` | (none — subblocking is not a knob) | DEST passes `s_s`, `s_o` follow `DEST_AUTO_LIMIT` |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| cur_pos[u], page_table[u] → reader/writer scratchpads | int32 bytes | `noc.async_read<TXN_ID>(accessor, scratchpad, …, {.trid = 0})` + barrier on trid 0; read through `MEM_L1_UNCACHED_BASE` alias | per-hart offset in the kernel's scratchpad; values clamped (R6) |
| cur_pos → compute | Int32 tile | `cb_cur_pos`, **explicit** (hart writes word 0); compute `read_tile_value` (`dataflow_buffer.h:387`) then `dummy_unpack` (`tile_move_copy.h:102`) before `pop_front` (TEN-4746) | one entry per user per NEO |
| reduce scaler | bf16 | `prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>` (value 1.0 serves MAX and SUM ROW), **explicit** | once per kernel; popped by compute at kernel end |
| q tiles (L1 shard core) → `cb_q_tiles` | bf16 | **implicit** tagged reads through q's `TensorAccessor` (sharded or interleaved, R12) | `tensor_d_tiles` per user per NEO |
| K tiles (DRAM) → `cb_k_tiles` | bf16 | **implicit**, one tagged read per tile, order (t, j) of the chunk | page ids from the clamped page table |
| V tiles (DRAM) → `cb_v_tiles` | bf16 | **implicit**, order (o-sub-block, t, j ∈ sub-block) | matches pv_block's loop |
| masks → `cb_score_masks` | intermediate format | **explicit**: hart writes 1024 elements in tile face order (`face = (r/16)·2 + c/16`, offset `(r%16)·16 + c%16`): body mask `0` if row r ∈ group c else `NEG_BIG`; last mask additionally `NEG_BIG` for columns `c > cur_pos % 32`; rows ≥ NH belong to no group | body mask only when `has_row_groups = (NKV > 1 or NH < 32)` (CT) |
| compute ↔ compute | intermediate / bf16 | compute-owned rings, explicit (packer → unpacker) | `pack_init(ring)` before any helper whose output ring differs from the previous pack's |
| partial → `cb_partial_out` → scratch slot (L1 interleaved) | partial format | writer **explicit** ring, tagged writes trid 0, barrier, `pop_front`; then `Semaphore::up(noc, combiner_x, combiner_y, C · neos_served)` + `noc.async_atomic_barrier()` | writer hart w serves NEOs w, w+W in rotation |
| scratch slots → `cb_partials_in` (combiner) | partial format | each combiner reader hart: `Semaphore::down(W_active)` (consumes exactly; total ups = C·W_active so the count returns to 0), then **implicit** reads `[m_k, l_k, O_k[d-share]]` per partial | only cluster_rank 0 |
| final → `cb_output_tiles` → output | bf16 | writer **explicit**, tagged, barrier | d-share per NEO; padded round re-stores the same tile |
| end of every DM kernel | — | reader: `noc_async_read_set_trid(0, noc.get_noc_id())`; writer: `noc_async_write_set_trid(0, …)` | hand the sticky tag back (target page) |

Protocol per kernel: reader = mixed (implicit `cb_q_tiles`, `cb_k_tiles`, `cb_v_tiles`, `cb_partials_in`; explicit `cb_cur_pos`, `cb_reduce_scaler`, `cb_score_masks` listed in `disable_dfb_implicit_sync_for`, every explicit-ring and scratchpad transfer with `{.trid = 0}`) as in `reader_mixed.cpp`; writer = explicit for both rings (`disable_dfb_implicit_sync_for_all`), every transfer tagged.

**Placement scheme per TARGET value** (the blocking scheme seen as the `memory_config` / `sharding` axes of the queries): `L1/height` (Phase 0) and `DRAM|L1/interleaved` are all one read through q's accessor — **knob-turn** unlocks (placement only), because q is reuse-shared and tiny; no cut of q crosses the dependent axis. The caches, page table and positions are DRAM interleaved only (R13). `output_memory` has the one value DRAM (R14). No sharded scheme changes the algorithm.

**Tensix-to-Tensix contract (built):** slot `u·max_workers_per_user + w` in `partials` is written only by worker w of user u; the combiner reads slot k only after `down(W_active)` returns; writers signal only after their slot writes are barriered. `partials_memory` (L1 / DRAM / combiner-local) and the deferred R-tree keep this contract and change only the slot address space / add a level.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a (user, worker) range of the virtual key sequence; at host level a *cluster group* of `G_u` clusters per user |
| Grid | `device.compute_with_storage_grid_size()` (32 on this target); engaged clusters = `num_groups · G_u` row-wise from the grid origin (catalog `noc_placement`: rows beat columns) |
| Per-core work | cluster rank r of a group: NEO n is worker `w = r·C + n`; units from `decode_plan.hpp` on the device; combiner = rank 0 (phase 2 for its group's users) |
| Remainder | users: `split_work_to_cores(grid, B)` groups 1/2 when `G_u = 1`; units: `base + (w < rem)`; d-share: `ceil(tensor_d_tiles / C)` rounds, short round padded; `user_pos_tiles = ceil((cur_pos+1)/32)`, tile counts all `ceil` |
| Bound | **compute** — per unit 2·`tensor_d_tiles` matmul tiles + mask, sub, exp (SFPU), two reduce passes and the per-chunk merge, against 2·`tensor_d_tiles` tiles read: several passes plus SFPU work per tile (target page "Choosing the plan": a matmul / SFPU work) |
| Thread plan | **(R, C, W) = (4, 4, 2)**, host constant `THREAD_PLAN`; counts passed as CT args; validated `C ≤ 4, R + W ≤ 6, R | C, W | C` **and `R == C`** (mechanism cap: per-NEO reader streams differ in length) |
| Unit dealt to compute threads | a worker range (much larger than a ring entry). Every ring is counted per NEO × C. Reader hart n feeds only NEO n (R = C), so unequal ranges need no padding on reader rings; on the writer rings every NEO pushes equal counts (partial `tensor_d_tiles+2`; output `d_share_tiles`, last round padded by recomputing d-tile `q·C`). Whole rounds of C workers are dealt to clusters (a cluster is engaged with all C NEOs or not at all) |
| Tests the plan needs | (1) every thread ran: a debug hook (`_DEBUG_KEEP_SCRATCH` module flag keeps the last `partials` tensor; the test pre-fills it via the hook's sentinel) checks every active worker slot was written, at C = 4 and a forced C = 1 plan; (2) one device session: the op after another program, and `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` after the op |

**Regime-pinned tests** (one built regime, but its topology depends on N and the grid): `G_active = 1` (QWEN_ONE_KV_HEAD), `G_active > 1` (LLAMA_CAPTURED at cur_pos 4095 → 8 clusters; PAIRS_OF_HEADS → 2), the same LLAMA_CAPTURED call with `_MAX_CLUSTERS_PER_USER = 1` (R18: one cluster serves everything), neutral partials (ONE_TILE_WIDE at cur_pos 5: 1 unit, 3 empty workers), chunks cut at head boundaries (PAIRS_OF_HEADS, BLOCKS_OF_TWO_TILES), the matmul / reduce realizations at C = 1 and C = 4 (target page: those helpers are unproven under a plan).

## Circular Buffers

Formats: **I** = intermediate = Float32 when `fp32_dest_acc_en` else Float16_b; **B** = Float16_b (q dtype); Int32 for `cb_cur_pos`. All capacities below are **per NEO**; `num_entries` = value × C. All rings STRIDED. Sizes in `l1_ledger.md`.

| Semantic Name | Index | Page Size | Num Pages (per NEO) | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---|---|---|---|---|---|---|---|---|---|
| `cb_cur_pos` | 0 | Int32 tile | 1 | one scalar per user | Int32 | reader | compute | explicit | per user |
| `cb_reduce_scaler` | 1 | B tile | 1 | constant | B | reader | compute | explicit | kernel |
| `cb_q_tiles` | 2 | B tile | 2 | streams q into scale_q | B | reader | compute | implicit | per user |
| `cb_k_tiles` | 3 | B tile | `kv_depth_tiles` | streams over (t, j); front-only | B | reader | compute | implicit | phase 1 |
| `cb_v_tiles` | 4 | B tile | `kv_depth_tiles` | streams over (sub-block, t, j); front-only | B | reader | compute | implicit | phase 1 |
| `cb_score_masks` | 5 | I tile | `mask_depth` | 2 masks per segment (nominal), two segments | I | reader | compute | explicit | per segment |
| `cb_partials_in` | 6 | I tile | `partial_depth` | streams `[m, l, O share]` per partial | I | reader | compute | implicit | phase 2 |
| `cb_q_scaled` | 7 | B tile | `tensor_d_tiles` | spans d (indexed matmul in0), nominal block | B | compute | compute | — | per user |
| `cb_scores` | 8 | I tile | `k_chunk_tiles` | spans the chunk (indexed max + exp), nominal block | I | compute | compute | — | per chunk |
| `cb_probs` | 9 | B tile | `k_chunk_tiles` | spans the chunk (indexed PV across d-sub-blocks, then rowsum), nominal block | B | compute | compute | — | per chunk |
| `cb_max_run` | 10 | I tile | 2 | m state (phase 1 and 2) | I | compute | compute | — | per user |
| `cb_max_step` | 11 | I tile | 2 | m_chunk (phase 1) / β (phase 2) | I | compute | compute | — | per step |
| `cb_alpha` | 12 | I tile | 2 | α; 1/l in normalize | I | compute | compute | — | per step |
| `cb_sum_run` | 13 | I tile | 2 | l state | I | compute | compute | — | per user |
| `cb_sum_step` | 14 | I tile | 2 | l_chunk (phase 1) / β·l_k (phase 2) | I | compute | compute | — | per step |
| `cb_out_acc` | 15 | I tile | `tensor_d_tiles` | O state (phase 2: `d_share_tiles ≤ tensor_d_tiles`), in-place per tile | I | compute | compute | — | per user |
| `cb_out_step` | 16 | I tile | `tensor_d_tiles` | O_chunk (phase 1) / β·O_k (phase 2) | I | compute | compute | — | per step |
| `cb_partial_out` | 17 | I tile | `tensor_d_tiles + 2` | one partial | I | compute | writer | explicit | per user |
| `cb_output_tiles` | 18 | B tile | 2 | streams d-share | B | compute | writer | explicit | phase 2 |

Push = wait for every ring: chunk rings push/pop exactly `k_chunk_tiles` per chunk; `cb_score_masks` exactly 2 per segment; `cb_q_scaled` exactly `tensor_d_tiles` per user; state rings 1 tile per update (pop then push); `cb_partial_out` `tensor_d_tiles+2` per user per active NEO; `cb_output_tiles` `d_share_tiles` per user per combiner NEO; `cb_partials_in` `W_active·(2 + d_share_tiles)` per user per combiner NEO. Each ring has exactly one producer kernel and one consumer kernel. Semaphore: `sem_partials_ready`, `target_nodes` = combiner clusters; bound by writer (remote `up`) and reader (`down`). Scratchpads: reader `R·(align_up(4·P, 64) + align_up(4·B, 64))` B, writer `W·align_up(4·B, 64)` B.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|---|---|---|---|---|---|
| 1 | load_position_block (compute side) | 1 | raw | `cb_cur_pos` (1, wait → `read_tile_value` → `dummy_unpack` → pop) | — | plan computed; inactive → skip user |
| 2 | scale_q_block | `tensor_d_tiles` | helper | `cb_q_tiles` (front, per tile pop) | `cb_q_scaled` (`tensor_d_tiles`) | q' resident for the user |
| 3 | score_block | `[1×n]`, contraction `tensor_d_tiles` | raw | `cb_q_scaled` (indexed j·stride), `cb_k_tiles` (front, pop per tile), `cb_score_masks` (front 0/1, held) | `cb_scores` (reserve/push `k_chunk_tiles`, write n) | masks popped at segment end |
| 4 | rowmax_block | `[1×n]` → 1 | raw | `cb_scores` (indexed t·stride, not popped), `cb_reduce_scaler` | first chunk: `cb_max_run`; else `cb_max_step` | — |
| 5 | stats_block (not on first chunk) | 2 → 2 | raw | `cb_max_run` (pop), `cb_max_step` (pop) | `cb_max_run` (m_new), `cb_alpha` (α) | — |
| 6 | exp_block | `[1×n]` | helper chain | `cb_scores` (indexed, then pop nominal), `cb_max_run` (bcast Col, held) | `cb_probs` (reserve/push nominal, write n) | `cb_scores` empty |
| 7 | pv_block | `[1×n]·[n×tensor_d_tiles]` | raw | `cb_probs` (indexed t·stride), `cb_v_tiles` (front, pop per tile) | first chunk: `cb_out_acc`; else `cb_out_step` | — |
| 7b | rowsum (in pv_block, after PV) | `[1×n]` → 1 | helper | `cb_probs` (WaitAndPopPerTile n, then pop pad), `cb_reduce_scaler` | first: `cb_sum_run`; else `cb_sum_step` | `cb_probs` empty |
| 8 | merge_block (not on first chunk) | 1 + `tensor_d_tiles` | helper chain | `cb_sum_run`/`cb_out_acc` (pop per tile), `cb_alpha` (bcast Col, pop at end), `cb_sum_step`/`cb_out_step` (pop) | `cb_sum_run`, `cb_out_acc` | — |
| 9 | emit_partial_block | `tensor_d_tiles+2` | helper | `cb_out_acc`, `cb_max_run`, `cb_sum_run` (pop) — or fill for a neutral partial | `cb_partial_out` | state rings empty |
| 10 | merge_partial_block | 2 + `d_share_tiles` | raw stats + helper chains | `cb_partials_in` (front), state rings | state rings; β via `cb_max_step`, β·l_k via `cb_sum_step`, β·O_k via `cb_out_step` | first partial initializes the state by copy |
| 11 | normalize_block | `d_share_tiles` | helper chain | `cb_sum_run` (recip → `cb_alpha`), `cb_out_acc` (pop) | `cb_output_tiles` | all empty |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|---|---|---|---|---|---|---|---|
| (kernel start) | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>` | `compute_kernel_hw_startup.h:59` | (cb_q_scaled, cb_k_tiles, cb_scores) once; Reverse because matmul maps in0→SrcB (`matmul.h:94-97`); every later op re-inits | — | — | — |
| load_position_block | raw_api | `DataflowBuffer::read_tile_value<uint32_t>` + `dummy_unpack` | `dataflow_buffer.h:387`; `tile_move_copy.h:102` | (0, 0) | `cb_cur_pos` | — | — |
| scale_q_block | helper | `compute_kernel_lib::unary<MulUnary<>, …>` | `eltwise/api/convenience.hpp:86`; `MulUnary` `eltwise/unary/scalar.hpp:32`, `scalar.inl:56` → `mul_unary_tile` Quasar path `binop_with_scalar.h:61-63` | `IterationShape::tiles(tensor_d_tiles)`, scale fp32 bits (common RT) | `cb_q_tiles` | `cb_q_scaled` | `tensor_d_tiles` |
| score_block | raw_api | `matmul_init(cb_q_scaled, cb_k_tiles, /*transpose=*/1)`, `matmul_tiles` | `matmul.h:113`, `:172` | in1 (K) transposed at unpack (SrcA, `llk_unpack_matmul.h:121`); in0 index `j·stride`, in1 index 0 (front); DEST pass `s_s ≤ DEST_AUTO_LIMIT` | `cb_q_scaled`, `cb_k_tiles` | `cb_scores` | `k_chunk_tiles` |
| score_block (mask) | raw_api | `add_reuse_dest_tiles` / `binary_reuse_dest_init` (DEST += mask) | `eltwise_binary.h:468`, `:55` (Quasar path `:430-436`) | mask index 0 (body, only if `has_row_groups`) / 1 (last, head-last tile) of the segment, × stride | `cb_score_masks` | (DEST) | — |
| rowmax_block | raw_api | `reduce_init` / `reduce_tile<MAX, REDUCE_ROW>` / `reduce_uninit` | `reduce.h:66`, `:147`, `:96` | tile index `t·cb_scores.get_consumer_stride_tiles()`, scaler tile 0 | `cb_scores`, `cb_reduce_scaler` | `cb_max_step` / `cb_max_run` | `k_chunk_tiles` |
| stats_block / merge_partial stats | raw_api | `copy_tile`, `binary_max_tile`, `sub_binary_tile`, `exp_tile`, `pack_tile`, `pack_init` | `tile_move_copy.h:126`; `binary_max_min.h:126`, `:145`; `eltwise_binary_sfpu.h:131`; `exp.h:85`, `:40`; `pack.h:87`, `:32` | D0 = m_run, D1 = m_step, D2 = max(D0,D1); D0 = exp(D0−D2) [α]; phase 2 also D1 = exp(D1−D2) [β]; ≤ 3 slots ≤ fp32 DEST limit | `cb_max_run`, `cb_max_step` / `cb_partials_in` | `cb_max_run`, `cb_alpha` (+ `cb_max_step` for β) | — |
| exp_block | helper | `eltwise_chain(BinaryFpu<Sub, input(cb_scores, None, None), input(cb_max_run, BroadcastDim::Col, …, PopPolicy::None)>, Exp<approx>, PackTile<output(cb_probs)>)` | `eltwise/api/chain.hpp:562-572` (BinaryFpu), `eltwise/unary/math.hpp:22` (Exp); FPU bcast Col on Quasar `llk_unpack_binary_broadcast_operands.h:45` | `IterationShape::tiles(n)`, `Approx::Fast` iff `math_approx_mode`; nominal `wait_front/pop_front(k_chunk_tiles)` and `reserve_back/push_back(k_chunk_tiles)` issued around it | `cb_scores`, `cb_max_run` | `cb_probs` | `k_chunk_tiles` |
| pv_block | raw_api | `matmul_init(cb_probs, cb_v_tiles, 0)`, `matmul_tiles` | `matmul.h:113`, `:172` | in0 index `t·stride`, in1 index 0; t-loop inside an o-sub-block of `s_o` DEST tiles | `cb_probs`, `cb_v_tiles` | `cb_out_step` / `cb_out_acc` | `k_chunk_tiles`, `o_block_tiles` |
| pv_block rowsum | helper | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, cb_probs, cb_reduce_scaler, out, WaitAndPopPerTile>` | `reduce_helpers_compute.hpp:654-681`; `ReduceInputBlockShape::row(n)` `:300-311` | front-only (index 0); pad entries popped after (dummy unpack first) | `cb_probs` | `cb_sum_step` / `cb_sum_run` | `k_chunk_tiles` |
| merge_block | helper | `eltwise_chain(BinaryFpu<Mul, input(cb_out_acc), input(cb_alpha, Col, …, PopPolicy::None)>, DestReuseBinary<Add, input(cb_out_step), DEST_TO_SRCA>, PackTile<output(cb_out_acc)>)`; same for l with `cb_sum_*`; `cb_alpha` popped after both | `chain.hpp:579-583` (DestReuseBinary), `:562` (BinaryFpu); Quasar dest-reuse `eltwise_binary.h:430-436` | `tiles(tensor_d_tiles)` / `tiles(1)`; `pack_init` before each | `cb_out_acc`, `cb_alpha`, `cb_out_step` | `cb_out_acc` | `tensor_d_tiles` |
| emit_partial_block | helper | `compute_kernel_lib::copy` / `eltwise_chain(FillScalar<>, PackTile)` (neutral) | `convenience.hpp:113`; `eltwise/generators/fill.hpp:20` | O tiles, then m, then l | state rings | `cb_partial_out` | — |
| merge_partial_block (β·O_k, β·l_k) | helper | `mul<input(cb_partials_in), input(cb_max_step, Col), output(cb_out_step)>` then merge_block's chain | `convenience.hpp:70` | `tiles(d_share_tiles)` | `cb_partials_in`, `cb_max_step` | `cb_out_step` / `cb_sum_step` | `d_share_tiles` |
| normalize_block | helper | `eltwise_chain(CopyTile<cb_sum_run>, Recip<>, PackTile<cb_alpha>)`; `mul<input(cb_out_acc), input(cb_alpha, Col), output(cb_output_tiles)>` | `eltwise/unary/math.hpp:34`; `convenience.hpp:70` | recip as its own chain (not a reduce post-op) | `cb_sum_run`, `cb_out_acc` | `cb_output_tiles` | `d_share_tiles` |
| reader: scaler | helper | `prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb, 1.0f)` | `reduce_helpers_dataflow.hpp:70` | bf16 packed; MAX and SUM ROW both use 1.0 | — | `cb_reduce_scaler` | — |

**Helpers considered and rejected (every raw_api row):**

| Raw block op | Helper considered | Mismatch (file:line) | What closes the gap |
|---|---|---|---|
| score_block, pv_block | `matmul_block` helper | not used: the matmul is written with LLK calls | — |
| score_block, pv_block | raw `matmul_block` (ct/kt > 1) | Quasar asserts `idst == 0` (`matmul.h:297`) | — |
| score_block mask | eltwise `add<>` helper | the mask must be added in the same DEST section as the matmul accumulation (a chain owns its own acquire, `chain.hpp` §5); a separate pass would re-read and re-pack S | — (alternative: separate `add<>` pass into a second ring, costs one ring + one pass) |
| rowmax_block | `reduce<MAX, REDUCE_ROW>` with `WaitUpfrontNoPop` | not used under C > 1; `WaitAndPopPerTile` would pop S, which exp_block still needs | — |
| stats_block | `binary_sfpu<BinaryMax>` | `minmax.hpp`: not used | — |
| load_position_block | — (no helper reads a scalar from a ring) | — | — |

Compute config: `ComputeHardwareConfig(fpu_math_fidelity = math_fidelity (always set), enable_32_bit_dest = fp32_dest_acc_en, …)`; `math_approx_mode` → `Exp<Approx::Fast>` CT flag. Built from the resolved config's three attributes (R10); `default_compute_kernel_config()` is the only default (R8).

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|------------------|------------------|---------------|
| exp_block | sub | `cb_scores`: All | `cb_max_run`: Col0 (REDUCE_ROW output) | Col |
| merge_block | mul | `cb_out_acc`: All | `cb_alpha`: Col0 (elementwise of Col0 tiles) | Col |
| merge_block | dest-reuse add | DEST: All | `cb_out_step`: All | None |
| merge_block (l) | mul / add | `cb_sum_run`: Col0 | `cb_alpha` / `cb_sum_step`: Col0 | None (Col0 meaningful, rest finite) |
| merge_partial | mul | `cb_partials_in` O tile: All | `cb_max_step` (β): Col0 | Col |
| normalize | mul | `cb_out_acc`: All | `cb_alpha` (1/l): Col0 | Col |
| score_block | dest-reuse add | DEST S: All | `cb_score_masks`: All | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Indexed unpack under C = 4 | matmul (q', P) and reduce (S) use indexed reads | multiply every index by `get_consumer_stride_tiles()`; test C = 1 vs C = 4 on the same cell first |
| Writer rotation deadlock | a writer hart serves 2 NEOs in strict rotation; a NEO that emits later or fewer tiles blocks the other | equal per-NEO counts on both writer rings; the combine is d-split over all C NEOs so no NEO waits on its own writer partner's partial |
| Device-only position | host cannot know `user_pos_tiles`; any host-sized loop reads K/V past cur_pos | all three kernels derive the plan from cur_pos with `decode_plan.hpp`; compute gets cur_pos through `cb_cur_pos` |
| −inf masks → NaN | other-head rows and padding rows are fully masked within a segment | finite `NEG_BIG` / `NEG_EMPTY`; never −inf |
| Scale sign / value | R4: any float; folding scale after the mask would flip `NEG_BIG` for scale < 0 | scale applied to q before QK (`scale_q_block`) |
| Implicit/explicit tag hygiene | reader is mixed, writer explicit, semaphores atomics | every explicit-ring, scratchpad and atomic transfer tagged trid 0; trid reset as each DM kernel's last statement; run after another program and before `ttnn.to_layout` in one session |
| Semaphore left non-zero | a later launch would pass its wait early | ups total `C·W_active`, each of the C combiner reader harts `down(W_active)`: ends at 0 |
| Out-of-range device values (R6) | an invalid page-table entry would address outside K/V | clamp cur_pos and every page-table entry on the device |
| `reduce_tile<MAX>` with 32-bit DEST on Quasar | tt-metal's own Quasar fork avoids a 32-bit DEST block max (spec GUIDANCE) | Phase 0 uses `reduce_tile` (compiles per `reduce_helpers_compute.inl`); if it misbehaves at fp32, fall back to `sfpu_reduce<MAX, Float32, REDUCE_ROW>` (`compute_kernel_api.h:791-822`) on a copied tile |
| Compute kernel binary limit 24 KiB | many phases, raw + chains, two DEST widths | one CT branch per `has_row_groups`; reuse the same chain instantiations in phase 1 and phase 2 (same rings) |
| Pack target | almost every op changes output ring | `pack_init(out_ring)` before every helper/pack whose ring differs from the previous pack's |
| TEN-4746 wait→pop without unpack | `cb_cur_pos` and nominal pad pops | `dummy_unpack` before any pop not preceded by a real unpack |
| DestReuseBinary on Quasar untested | merge_block relies on it | fallback: `mul` into `cb_out_step`-sized temp then `add` (two helper calls), costs one extra ring of `tensor_d_tiles` |
| Partial format vs output format | in fp32 mode partials are Float32, output bf16 | separate writer rings `cb_partial_out` (I) and `cb_output_tiles` (B) |

## Structural impossibilities

None to add: the spec's `INVALID` (`{memory_config: DRAM, sharding: height}`) is complete for this design.
