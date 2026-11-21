# L1 Ledger: paged_scaled_dot_product_attention_decode

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Design: `op_design.md` (this directory).
Target: `quasar` on ttsim — 4 NEOs per cluster, budget `ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes` = 3,963,328 − 65,536 = 3,897,792 B per cluster on this target.

Every capacity is **per NEO**; the ring's `num_entries` is the capacity × C (thread plan (R, C, W) = (4, 4, 2), STRIDED rings). Formats: **I** = Float32 when `fp32_dest_acc_en` else Float16_b (tile `I_T` = 4,096 / 2,048 B); **B** = Float16_b (q dtype, tile `B_T` = 2,048 B); Int32 tile = 4,096 B.

Block axes accounted in every row: **user** (`users_per_block` = 1), **seg** (head segment), **pos** (`k_chunk_tiles`), **d** (`tensor_d_tiles` / `d_share_tiles`), **part** (partials, phase 2).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_cur_pos` | 1 (constant: one scalar per user) | 1 | {user: streams → 1, seg: streams, pos: streams, d: streams, part: streams} | Int32 | reader | compute | user | Not shared: the only Int32 ring; a tile-sized entry because compute reads it after a real `dummy_unpack` (TEN-4746) |
| `cb_reduce_scaler` | 1 (constant) | 1 | {user: streams, seg: streams, pos: streams, d: streams, part: streams} | B | reader | compute | kernel | Not shared: live for the whole kernel (every rowmax and rowsum reads it; also the datacopy that primes both DEST banks at kernel start) |
| `cb_q_tiles` | `tensor_d_tiles` (one block) | `tensor_d_tiles` | {user: streams, seg: streams, pos: streams, d: spans → `tensor_d_tiles`, part: streams} | B | reader | compute | user (start) | Not shared: concurrent with every compute-owned ring of the user. One block because the reader writes q as one explicit-sync block (one barrier); **implemented: was 2 in the design** |
| `cb_k_tiles` | `kv_depth_tiles` = 2·`tensor_d_tiles` | `tensor_d_tiles` (one tile-row block) | {user: streams, seg: streams, pos: streams (t), d: spans one row → block, part: streams} | B | reader | compute | phase 1 | Not shared with `cb_v_tiles`: both in flight at once; capacity 2 rows = one row read while compute consumes the other (catalog `double_buffer`). Blocks of `tensor_d_tiles` divide the capacity, so a block never wraps |
| `cb_v_tiles` | `kv_depth_tiles` = 2·`tensor_d_tiles` | `pv_subblock_tiles` (one block) | {user: streams, seg: streams, pos: streams (t), d: sub-block order, part: streams} | B | reader | compute | phase 1 | as `cb_k_tiles`; `pv_subblock_tiles` is a divisor of `tensor_d_tiles` so V blocks never wrap |
| `cb_score_masks` | `mask_depth` = 4 (2 segments × 2 nominal) | 2 (one segment's body + last mask) | {user: streams, seg: spans → 1 segment (+1 prefetched), pos: streams, d: streams, part: streams} | I | reader | compute | segment | Not shared: written by a DM hart (explicit sync) while compute holds the previous segment's masks |
| `cb_partials_in` | `partial_depth` = 2·(2 + `d_share_tiles`) | 1 (front-only) | {user: streams, seg: streams, pos: streams, d: streams → `d_share_tiles`, part: streams → 2 partials} | I | reader | compute | phase 2 | Not aliased with K/V (format differs, ≤ 24 KiB). Order per partial **[m, O share, l]** (implemented; design had [m, l, O]) so the β product streams `d_share_tiles + 1` tiles in one call |
| `cb_q_scaled` | `tensor_d_tiles` (nominal block, indexed) | `tensor_d_tiles` | {user: spans → 1, seg: streams, pos: streams, d: spans → `tensor_d_tiles`, part: streams} | B | compute | compute | user (phase 1) | Not shared: resident across every chunk of the user. B: matmul in0 must match in1's format (`matmul.h:99-102`) |
| `cb_scores` | `k_chunk_tiles` (nominal block) | `k_chunk_tiles` (n valid ≤ it) | {user: streams, seg: streams, pos: spans → `k_chunk_tiles`, d: streams (contracted), part: streams} | I | compute | compute | chunk (score → exp) | Not shared with `cb_probs`: exp reads S and writes P in the same pass |
| `cb_probs` | `k_chunk_tiles` (nominal block) | `k_chunk_tiles` | {user: streams, seg: streams, pos: spans → `k_chunk_tiles`, d: streams (re-read per sub-block), part: streams} | B | compute | compute | chunk (exp → pv → rowsum) | Not shared: see `cb_scores` |
| `cb_max_run` | 2 | 1 | {user: spans → 1, seg: streams, pos: streams, d: streams, part: streams} | I | compute | compute | user (state) | Not shared: m state across every chunk and partial, held by exp while col_step changes; capacity 2 = pop/push overlap |
| `cb_col_step` | 2 | 1–2 | {user: streams, seg: streams, pos: streams (chunk), d: streams, part: streams} | I | compute | compute | step | **Shares by role** (implemented, replaces the design's `cb_max_step` + `cb_alpha`): m_chunk → α (phase 1, stats pops m_chunk then pushes α); [β, α] (phase 2); 1/l (normalize). Merged because a NEO has 16 Tensix-only tile counters and a compute-owned ring takes 2 (`TILE_COUNTERS_PER_INTRA_TENSIX_DFB`): at most 8 such rings |
| `cb_sum_run` | 2 | 1 | {user: spans → 1, seg: streams, pos: streams, d: streams, part: streams} | I | compute | compute | user (state) | Not shared: l state, concurrent with `cb_max_run` |
| `cb_out_acc` | `tensor_d_tiles` | `tensor_d_tiles` (phase 2: `d_share_tiles`) | {user: spans → 1, seg: streams, pos: streams, d: spans → `tensor_d_tiles`, part: streams} | I | compute | compute | user (state) | Not shared: O state across all chunks; updated in place one tile at a time |
| `cb_step` | `tensor_d_tiles + 1` | `tensor_d_tiles + 1` | {user: streams, seg: streams, pos: streams (chunk), d: spans → `tensor_d_tiles`, part: streams (β·[O share, l])} | I | compute | compute | step | **Shares by role** (implemented, replaces `cb_out_step` + `cb_sum_step`): [O_chunk, l_chunk] (phase 1) / [β·O_k share, β·l_k] (phase 2); the merge reads O first, then l. Same 8-ring cap as `cb_col_step` |
| `cb_partial_out` | `tensor_d_tiles + 2` | `tensor_d_tiles + 2` | {user: spans → 1, seg: streams, pos: streams, d: spans → `tensor_d_tiles`, part: spans → 1 (own)} | I | compute | writer | end of phase 1 | Not shared with `cb_output_tiles`: formats differ in fp32 mode and each writer ring needs its own equal per-NEO count |
| `cb_output_tiles` | 2 (streaming window) | 1 | {user: streams, seg: streams, pos: streams, d: streams → `d_share_tiles`, part: streams} | B | compute | writer | phase 2 | as above; capacity 2 > live 1 = double buffer |

Audit notes:
- **Page format vs DEST (audit 2).** I rings follow `fp32_dest_acc_en`. B rings with fp32 DEST: `cb_q_scaled`, `cb_probs` are matmul operands whose format must equal K / V's (`matmul.h:99-102`, #46769) — a mechanism reason; `cb_q_tiles`, `cb_k_tiles`, `cb_v_tiles` are dataflow payloads of the bf16 input that never went through DEST; `cb_output_tiles` is the output ABI; `cb_reduce_scaler` is the reduce scaler's required packed bf16 format. No Float32 page exists under 16-bit DEST (I = Float16_b then).
- **Capacity vs live set (audit 1).** Every gap is a stated window (double buffer / prefetch / pop-push overlap); chunk rings are exactly the nominal block.
- **Disjoint lifetimes (audit 3).** Phase-1 / phase-2 pairs reuse rings by role (`cb_col_step`, `cb_step`, `cb_out_acc`); `cb_partials_in` vs `cb_k_tiles` / `cb_v_tiles` not aliased (format and protocol differ, ≤ 24 KiB).
- **Count.** 17 rings ≤ 24 active per NEO, ≤ 32 per cluster; 8 of them compute-owned (the cap: 16 Tensix-only tile counters per NEO, two per compute-owned ring).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| C | 4 (`THREAD_PLAN`) | host `validate_thread_plan`: C ∈ {1,2,4}, R + W ≤ 6, R | C, W | C, R == C |
| `tensor_d_tiles` | Phase 0 ≤ 4 (D ≤ 128); any value satisfying the L1 predicate at `k_chunk_tiles` = 1 (fp32: ≤ 35 on this target) | L1 predicate; larger → `NotImplementedError` (R-wide-D, deferred) |
| `d_share_tiles` | `ceil(tensor_d_tiles / C)` | derived |
| `k_chunk_tiles` | `min(tensor_pos_tiles, ceil(N_max / max_workers_per_user))`, then lowered until the L1 predicate holds (≥ 1) | host function `k_chunk_tiles(...)` |
| `kv_depth_tiles` | 2·`tensor_d_tiles` | derived |
| `mask_depth` | 4 | constant |
| `partial_depth` | 2·(2 + `d_share_tiles`) | derived |
| P (page-table width) | scratchpad only: `align_up(4·P, 64)` B per reader hart | P ≤ blocks (R2); Phase 0 ≤ 128 |
| B (users) | scratchpad only: `align_up(4·B, 64)` B per DM hart (64 B per hart at B ≤ 16; 128 B at B = 32) | no predicate: the positions page grows 4 B per user; B ≤ 32 in INPUTS. A host bound (or a per-user read of one element) is needed before B reaches tens of thousands |
| `partials_bytes` | `B·max_workers_per_user·(tensor_d_tiles+2)·I_T` | its per-bank share counts against the budget when `partials_memory = L1` |
| budget | `ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes` (3,897,792 B on this target; `l1_reserve_bytes` = 65,536 host constant) | run-time query |

## Footprint

`per_neo = Int32_T + B_T·(1 + tensor_d_tiles + 4·tensor_d_tiles + tensor_d_tiles + k_chunk_tiles + 2) + I_T·(4 + 2·(2+d_share_tiles) + k_chunk_tiles + 2 + 2 + 2 + tensor_d_tiles + (tensor_d_tiles+1) + (tensor_d_tiles+2))`
(computed by the host from the same ring list it builds the program from: `_rings()` / `_ring_bytes()`).

`per_cluster = C·per_neo + R·(align_up(pos_page,64) + align_up(table_page,64)) + W·align_up(pos_page,64) + partials_bank_share ≤ budget`

Worked, this target, fp32 DEST (I_T = 4,096), from `_rings()`:

| Cell | `tensor_d_tiles` | `d_share_tiles` | `k_chunk_tiles` | per NEO | per cluster (×4) | of budget |
|------|-----|-----|-----|---------|-------------|-----------|
| LLAMA_CAPTURED | 2 | 1 | 32 | 333,824 B | 1,335,296 B | 34% |
| QWEN_TWO_KV_HEADS | 4 | 1 | 4 | 210,944 B | 843,776 B | 22% |
| BLOCKS_OF_TWO_TILES | 3 | 1 | 6 | 198,656 B | 794,624 B | 20% |

## Data-movement budget

Per user, `N = NKV·user_pos_tiles`, `T = tensor_d_tiles`, `W = W_active`; worked for LLAMA_CAPTURED at cur_pos = 4095 (N = 1024, T = 2, W = 32, fp32 partials).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| K cache | 1 (attended tiles only: N·T = 2,048 tiles = 4 MiB) | streamed once through `cb_k_tiles`; each unit belongs to exactly one worker; tiles past cur_pos's tile are never read | — |
| V cache | 1 (4 MiB) | as K | — |
| q | 0 (L1 shard) | — | W·T = 64 tiles (128 KiB) from the shard core: reuse-shared, read once per worker (R-q-reuse deferred) |
| page table | 1 row per reader hart per engaged cluster (4·8·512 B = 16 KiB) | each reader hart needs the row in its own scratchpad region | — |
| cur_pos | 1 page per DM hart per engaged cluster (6·8·4 B) | every kernel derives the plan itself | — |
| partials (scratch) | 0 (`partials_memory = L1`) | — | W·(T+2) = 128 tiles written (512 KiB) + C·W·(2+d_share) = 384 tiles read (1.5 MiB); 2 semaphore incs per writer hart |
| output | 1 (T tiles = 2 tiles; + padded re-stores when T < C: 2 more) | written once by the combiner cluster's d-share | — |

Total: DRAM ≈ 8 MiB (K + V; the minimum) + < 0.1 MiB; cross-core ≈ 2.1 MiB (partials + q).

> Cheapest-traffic split considered: the virtual-sequence split with partials in L1 — DRAM at the minimum (every attended K/V tile once), cross-core `W(T+2) + C·W(2+d_share)` partial tiles + `W·T` q tiles, which at W = √N grow as O(√N) against K/V's O(N). Implemented: the same split. It is the cheapest-traffic split that engages more than one cluster per user; a users-only split moves the same DRAM bytes but engages one cluster for every traced (B = 1) call.

> Implementation note (first phase): the DRAM and cross-core counts above are as built. What changed is the transfer
> shape on the reader side: every reader ring is explicit-sync, one barrier per block (one K tile-row of
> `tensor_d_tiles` tiles, one V d-sub-block, all of q, one partial piece), because implicit sync's ISR batches span all
> producer harts and the per-NEO streams differ in length. Transfers per barrier: K `tensor_d_tiles`, V
> `pv_subblock_tiles`, q `tensor_d_tiles`. The writer stores each served NEO's whole partial (`tensor_d_tiles + 2`
> tiles) under one barrier (verifier fix: it was one tile per barrier); the output stays one tile per barrier
> (`d_share_tiles` per NEO per user, = 1 at D <= 128: `pop_front` is what moves a writer hart to its next NEO's tile
> counter, so the rotation cannot batch across NEOs).
