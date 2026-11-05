# L1 Ledger: nomic_moe_route_dispatch

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Regime: `R1 nd_push_l1` (the only
built regime). Block axes (from `op_design.md`, Axes): `token` (`rps`), `hidden` (`ht`), `expert` (1 tile),
`choice` (2), `out_row` (1 tile row per block), `out_piece` (`w`), `out_block` (`num_blocks_this_core`),
`slot` (rows pushed). `T4 = 4096` (fp32 tile), `T2 = 2048` (bf16 tile).

Every CB is used for exactly one block per call (one block per core per phase), so live set and
capacity coincide unless stated; there is no streaming window anywhere.

## Implemented (Phase 0, commit after 620edd4ffe6)

> **Perf 1:** `cb_mask_keep` (4), `cb_masked_logits` (5), `cb_row_max` (6), `cb_exps` (7), `cb_recip_sum` (8),
> `cb_max_scaler` (10) and `cb_sum_scaler` (11) were removed by the fused SFPU softmax. At rps = 4 that frees
> 4 x 4 x 4 KB + 4 KB + 2 x 2 KB = 72 KB a core. `cb_bytes(plan)` / `circular_buffer_bytes()` follow automatically.
>
> **Perf 2:** `cb_router_weight` (1) holds faces 0 and 2 only: 12 x T4 (48 KB, was 96 KB), read through
> `cb_w_faces` (23), a 1 KB-page view of the same bytes (no allocation). `cb_own_partial` (2) and the new
> `cb_own_partial_west` (22) are 16 B handshakes: compute packs its partial straight into its own region of
> `cb_partials`. About 64 KB a core freed at rps = 4. `cb_scores` (9) now carries one top-2 code tile a score
> tile (same size), published a tile at a time.

The implementation departs from the planned inventory in four places, all advisory items:
aliases A, B and D are not built (every CB below is its own allocation: the fit at model occupancy
has > 600 KB of headroom at 8x512, and alias A carried a cross-core ordering hazard the design had to
argue away); the two row-staging CBs are one (`cb_x_rows`, so the push split is token granular and
rps = 1 shards use both DM RISCs); the pad list is not stored (the writer derives its pad rows from
the region header in `cb_slot_rows`); `weight_tiles` is its own CB (alias C dropped: zeroing it in
the W-fetch shadow took the zero-fill off the selection path). Two 16 B handshake CBs carry the
split selection between the DM RISCs.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_tiles` (0) | `rps*ht` x T2, **backed on x's shard (0 CB bytes)** | `rps*ht` | {token: spans -> rps, hidden: spans -> ht, others: n/a} | Float16_b | reader | compute | whole call (2 publish rounds) | Zero-copy over the input shard. Interleaved x (R2, Refinement 2): a real CB of exactly `rps*ht` = `rps*24` pages (G = 1), filled by one TensorAccessor burst of the core's tile rows; exactly the publish size so the second publish round re-exposes the same pages (no aliasing: it is live the whole call). |
| `cb_router_weight` (1) | 24 x T4 | slice `c` (ht tiles) | {hidden: spans -> 24 (every slice lands at its own offset), expert: 1 tile col} | Float32 | reader, or the writer on a core whose member slice travels on NOC1 (Refinement 4, `W_SLICE_NOCS`; one producer a core) | compute | until `partial_logits_block` | Not aliased with `cb_gathered_rows` (alias A dropped, see above). |
| `cb_own_partial` (2) | `rps` x T4 | `rps` | {token: spans -> rps, expert: 1} | Float32 | compute | reader | matmul -> exchange barrier | Separate (alias B dropped). |
| `cb_partials` (3) | `G*rps` x T4 | `G*rps` | {token: spans -> rps, hidden: G contributors} | Float32, UnpackToDestFp32 | reader (peers write it) | compute | exchange -> `logits_block` | Peers write it remotely before this core reserves it: never aliased. |
| `cb_mask_keep` (4) | 1 x T4 | 1 | {expert: 1 tile, others: constant} | Float32, UnpackToDestFp32 | reader | compute | whole call | Constant. |
| `cb_masked_logits` (5), `cb_row_max` (6), `cb_exps` (7), `cb_recip_sum` (8), `cb_scores` (9) | `rps` x T4 each | `rps` | {token: spans -> rps, expert: 1 (row max / recip: col 0)} | Float32 | compute (scores: compute) | compute (scores: reader + writer reads its tokens after `cb_select_go`) | softmax chain | Separate (aliases B, D dropped). |
| `cb_max_scaler` (10), `cb_sum_scaler` (11) | 1 x T2 each | 1 | constant | Float16_b | reader | compute | whole call | Pool-type-aware fills differ; the sum scaler covers only the E expert columns (Refinement 4: the exp skips faces 1/3). |
| `cb_x_rows` (12) | `rps*stage_tiles` x T2 (`stage_tiles` = `ht`, or the whole-piece staged width: 12 at G=3,k=2; 16 at G=2,k=3) | `rps*stage_tiles` on pushing members, 0 on a member that pushes nothing | {token: spans -> rps, hidden: spans -> stage_tiles} | Float16_b (row-major bytes) | compute (one untilize) | reader (writer reads its tokens by address after `cb_slot_rows`) | untilize -> end of pushes | One CB instead of the reader / writer pair: token-granular split (`PUSH_READER_SHARE`). |
| `cb_stage_tiles` (21), whole-piece push only (Refinement 3) | `rps*stage_tiles` x T2 | `rps*stage_tiles` (pushing members) | {token: spans -> rps, hidden: spans -> stage_tiles} | Float16_b | writer (NOC1 reads of its own and its group peers' resident x tiles; Refinement 4, was the reader on NOC0) | compute (untilize, after the softmax) | start -> untilize | Not packed from `cb_x_tiles`: the staged range spans peers' shards, so it is a NoC gather into one tile-row-major block the untilize helper can read (no strided untilize). Could alias any CB dead before the untilize ends (e.g. `cb_gathered_rows`, but remote pushes into it start only after the table, a cross-core ordering argument not worth 96 KB of headroom we have). |
| `cb_slot_rows` (14) | 1 page of `2*rps*32*4 + 128` B | same | {token: spans -> rps, choice: 2, header: starts/totals/end/staging/writer bases} | UInt32 | reader (writer fills its tokens' slots into the reserved page) | writer | selection -> pushes | Replaces the pad list (pad rows derived from the header). |
| `cb_gathered_rows` (15) | `max_blocks_per_core*w` x T2 | `num_blocks*w` | {out_block, out_piece -> w, out_row -> 1} | Float16_b (row-major bytes) | reader (publishes after its owner landing count, Perf R3.2; remote pushers fill it) | compute | pushes -> tilize | Not aliased (alias A dropped). |
| `cb_dispatched` (16) | `max_blocks_per_core*w` x T2, **backed on the output shards** | `num_blocks*w` | {out_block, out_piece, out_row} | Float16_b | compute | (output) | tilize | Zero-copy over the output. |
| `cb_dispatched` (16), DRAM out (R4) | `DRAM_OUT_BLOCKS_IN_FLIGHT*w` x T2 (2*w; 24 KB at w=6) | `w` per block | {out_block: streams, out_piece -> w, out_row -> 1} | Float16_b | compute (tilize, one block a push) | writer (DRAM, one flush a block) | tilize -> DRAM write | Replaces the zero-copy output CB when `dispatched` is in DRAM; depth 2 so the tilize of block b+1 overlaps the writes of block b. Not aliased with `cb_gathered_rows`: tilize reads that CB while packing into this one. Counted by `cb_bytes(plan)` / `circular_buffer_bytes(..., buffer_type=DRAM)`; `_l1_room` reserves no `dispatched` bytes for DRAM. |
| `reader_scratch` (17) | 1 page: table `round64(num_groups*16)` + drain 64 + counts 64 B | same | {group x expert table} | UInt32 | reader (+ coordinator multicast) | reader | whole call | Multicast target at a fixed address on every core. |
| `cb_select_go` (18) | 1 x 16 B | 1 | handshake | UInt32 | reader | writer | selection start | Handshake page. |
| `cb_writer_counts` (19) | 1 x 16 B | 1 | {expert: 8 x u8 counts} | UInt32 | writer | reader | selection end | Handshake page. |
| `weight_tiles` (20) | `2*rps` x T4 | `2*core_token_tiles` | {token: spans -> rps, choice: 2} | Float32 | reader + writer (column 0 of their tokens) | reader (leader, or every member under `WEIGHTS_SPREAD`: tiles c, c+G, ..., DRAM write) | zeroed at start -> after the landing increment | Own allocation (alias C dropped). |

Implemented total (`circular_buffer_bytes()` = `cb_bytes(plan)`, sharded CBs excluded): 8x512 608,032 B;
6x512 / 8x384 337,440; 4x512 / 8x256 296,224; 2x512 / 4x256 246,816. Fit at model occupancy, 8x512:
allocator base 112,576 + 608,032 = 720,608 <= 1,381,312 - 122,880 (dispatched) = 1,258,432.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `rps` | 1..4 | x placement == nd_norm (`nomic_residual_layer_norm` caps a shard at 4 tile rows), re-checked in `_plan()` (`MAX_ROWS_PER_SHARD`) |
| `G` | 2..3 | nd_norm hidden split; `_plan()` refuses G < 2 |
| `ht` | 8 or 12 (`24 / G`) | as `G` |
| `num_groups` | <= 55 | `G*num_groups <= 110` (`_plan()`) |
| `reader_tokens` | `round(PUSH_READER_SHARE * rps*32)` <= 128 | op constant `PUSH_READER_SHARE` |
| `w` | 8 or 12 | `nomic_moe_dispatch.plan(R)` for R in 2240..8512 |
| `max_blocks_per_core * w` | <= 63 tiles | dispatch budget `ceil(24*tr/110) + 4`, `tr = R/32 <= 266` |
| `REGION_WORDS` | 32 (>= 26 header words) | host constant; kernels `static_assert(kHeaderWords <= region_words)` |
| `core_token_tiles` | <= `rps` | per-core RT |

## Total per-core footprint (implemented; `circular_buffer_bytes()` = `cb_bytes(plan)`)

```
cb_bytes = 24*T4                                   # cb_router_weight            (constant: every W slice lands)
         + rps*T4                                  # cb_own_partial              (token)
         + G*rps*T4                                # cb_partials                 (token x contributor)
         + 2*rps*T4                                # weight_tiles                (token x choice)
         + T4                                      # cb_mask_keep
         + 5*rps*T4                                # masked logits, row max, exps, recip sum, scores (token)
         + 2*T2                                    # scalers
         + rps*stage_tiles*T2                      # cb_x_rows                   (token x staged hidden range)
         + [whole-piece push] rps*stage_tiles*T2   # cb_stage_tiles              (R3: G=3,k=2 and G=2,k=3)
         + [x interleaved] rps*24*T2               # cb_x_tiles (R2; nd_norm: zero-copy over the shard)
         + (2*rps*32 + 32)*4                       # cb_slot_rows                (token x choice + header)
         + max_blocks_per_core*w*T2                # cb_gathered_rows            (out_block x out_piece)
         + round64(num_groups*16) + 128            # reader_scratch              (group x expert)
         + 2*16                                    # handshake pages
```

| (B,S) | G | rps | total |
|-------|---|-----|-------|
| 8x512 | 3 | 4 | **608,032** (was 476,960 before Refinement 3's whole-piece push: +32,768 x_rows, +98,304 stage tiles) |
| 6x512 / 8x384 | 2 | 2 | **337,440** |
| 4x512 / 8x256 | 3 | 2 | **296,224** |
| 2x512 / 4x256 | 2 | 1 | **246,816** (was 205,856: +8,192 x_rows, +32,768 stage tiles) |

Terms that scale with a knob: `cb_gathered_rows` with the output plan (`max_blocks_per_core*w`, i.e. with R);
every token-axis CB with the pinned shard (`rps`, `G`, `ht`); nothing scales with T beyond the shard and R.
Fit at the model's occupancy, 8x512: allocator base 112,576 + 608,032 = 720,608 <= 1,381,312 - 122,880
(`dispatched`, 5*12*2048 per bank) = 1,258,432. The planned aliases A-D would save ~173 KB at 8x512; they are
not needed while this headroom holds (> 660 KB) and are the first lever if a later regime runs short.

## Planned (op_design.md) - superseded

Kept for the alias proofs (A-D) a later L1-pressure refinement can re-adopt. Not the built inventory: the
symbol table and the total below describe the implementation.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_tiles` | `rps*ht` x T2, **backed on x's shard (0 CB bytes)** | `rps*ht` | {token: spans -> rps, hidden: spans -> ht, expert: n/a, choice: n/a, out_row/out_piece/out_block: n/a, slot: n/a} | Float16_b (input dtype; read-only) | reader | compute | whole call (2 publish rounds) | Zero-copy over the input shard; nothing to share. |
| `cb_router_weight` | 24 x T4 (alias A capacity `max(24*T4, max_blocks_per_core*w*T2)`) | `ht` tiles used (slice c); all 24 slots are written by the G multicasts | {hidden: spans -> 24 (all G slices land: one multicast address per slice), token: n/a, expert: spans -> 1 tile col, others: n/a} | Float32 (weight dtype; fp32 DEST) | reader | compute | start -> end of `partial_logits_block` | **Alias A with `cb_gathered_rows`**: disjoint because senders barrier before signalling and any push needs the table, which needs every slice signal and every member's matmul. Capacity > live set (24 vs ht): the multicast lands every slice at its own offset; receiving only slice c would need non-rectangular receiver sets (lamp L5). |
| `cb_gathered_rows` | `max_blocks_per_core*w` x T2 (alias A) | `num_blocks_this_core*w` | {out_block: spans -> max_blocks_per_core, out_piece: spans -> w, out_row: spans -> 1 tile row (32 rows), token/hidden/expert/choice: n/a, slot: streams (rows land in arbitrary order)} | Float16_b (row-major bytes) | reader (publishes after `sem_go`; remote pushers fill it) | compute | first push landing -> tilize | Alias A (above). Cannot pack straight into `cb_dispatched`: a 32-row x w-tile block is not tilizable in place (packing tile 0 overwrites rows 0..2 that later tiles still read; `w` > fp32 DEST tiles). |
| `cb_dispatched` | `max_blocks_per_core*w` x T2, **backed on the output shards (0 CB bytes)** | `num_blocks_this_core*w` | {out_block: spans, out_piece: spans -> w, out_row: spans -> 1, others: n/a} | Float16_b | compute | (output tensor) | tilize | Zero-copy over the output; nothing to share. |
| `cb_own_partial` | `rps` x T4 (alias B capacity) | `core_token_tiles` | {token: spans -> rps, expert: spans -> 1, hidden: n/a (reduced), others: n/a} | Float32 (fp32 DEST) | compute | reader | matmul -> after the exchange barrier | **Alias B with `cb_masked_logits`, `cb_scores`**: popped by the reader before it publishes `cb_partials`, so before `logits_block` writes `cb_masked_logits`. Not packed straight into `cb_partials` region c: that CB is produced by the reader (peers write it); one producer per CB. |
| `cb_masked_logits` | `rps` x T4 (alias B) | `core_token_tiles` | {token: spans -> rps, expert: spans -> 1, others: n/a} | Float32 | compute | compute | `logits_block` -> chain 2 of `softmax_block` | Alias B. Not in place over `cb_partials`: different shape (`G*rps` in, `rps` out) and `cb_partials` has a dataflow producer. |
| `cb_scores` | `rps` x T4 (alias B) | `core_token_tiles` | {token: spans -> rps, expert: spans -> 1, choice: n/a, others: n/a} | Float32 | compute | reader | chain 3 -> end of `select_block` | Alias B: `cb_masked_logits` is popped at the end of chain 2, before chain 3 packs scores. Not in place over `cb_exps`: chain 3 reads exps and recip together and the reader is a second consumer. |
| `cb_partials` | `G*rps` x T4 (alias C capacity `max(G*rps, 2*rps)`) | `G*core_token_tiles` | {token: spans -> rps, hidden: spans -> G contributors (reduced partials), expert: spans -> 1, others: n/a} | Float32, UnpackToDestFp32 | reader | compute | exchange -> `logits_block` | **Alias C with `weight_tiles`** (reader scratch, `2*rps` fp32 tiles for `combine_weights`): `logits_block` pops partials before scores exist; the reader builds weight tiles after `select_block`. |
| `weight_tiles` (reader scratch) | `2*rps` x T4 (alias C) | `2*core_token_tiles` | {token: spans -> rps, choice: spans -> 2, others: n/a} | Float32 (bit copy of scores) | reader | reader (scratch -> DRAM) | `write_combine_weights_block` | Alias C (above). Whole block, not 2 rotating slots as routing_metadata: it is written in the count-exchange wait, where nothing else needs the L1, and alias C makes it free. |
| `cb_mask_keep` | 1 x T4 | 1 (constant) | {expert: spans -> 1 tile, all others: n/a (constant)} | Float32, UnpackToDestFp32 | reader | compute | whole call | Constant tile; read by every `logits_block` tile, so live the whole compute phase. A generated cond in DEST would need a column-index SFPU op the helpers do not expose. |
| `cb_row_max` | `rps` x T4 (alias D capacity) | `core_token_tiles` (col 0 valid) | {token: spans -> rps, expert: reduced -> 1, others: n/a} | Float32 | compute | compute | reduce max -> end of chain 2 | **Alias D with `cb_recip_sum`**: chain 2 pops max before reduce(sum) packs recip. |
| `cb_recip_sum` | `rps` x T4 (alias D) | `core_token_tiles` | {token: spans -> rps, expert: reduced -> 1, others: n/a} | Float32 | compute | compute | reduce sum -> end of chain 3 | Alias D. |
| `cb_exps` | `rps` x T4 | `core_token_tiles` | {token: spans -> rps, expert: spans -> 1, others: n/a} | Float32 | compute | compute | chain 2 -> chain 3 | Concurrent with `cb_masked_logits` (chain 2 reads one, writes the other) and with `cb_recip_sum` (chain 3 reads both); no disjoint partner left. |
| `cb_max_scaler` | 1 x T2 | 1 | {all axes: n/a (constant)} | Float16_b (packed reduce scaler) | reader | compute | whole call | Pool-type-aware scaler for MAX; SUM needs its own fill pattern (`prepare_reduce_scaler` overloads), so two tiles. |
| `cb_sum_scaler` | 1 x T2 | 1 | {all axes: n/a (constant)} | Float16_b | reader | compute | whole call | See `cb_max_scaler`. |
| `cb_x_rows_reader` | `rows_reader*ht` x T2 | same | {token: spans -> rows_reader, hidden: spans -> ht, slot: streams, others: n/a} | Float16_b (row-major bytes) | compute | reader | untilize -> end of the reader's pushes | Concurrent with every other buffer through the push phase (rows are copied out of it until the barrier). Split from `cb_x_rows_writer` because each CB has one consumer. |
| `cb_x_rows_writer` | `(rps-rows_reader)*ht` x T2 | same | {token: spans -> rps-rows_reader, hidden: spans -> ht, slot: streams, others: n/a} | Float16_b (row-major bytes) | compute | writer | untilize -> end of the writer's pushes | See `cb_x_rows_reader`. |
| `cb_slot_rows` | `2*(rps-rows_reader)*32 + 2*MAX_PAD_ROWS + 2` x 4 B | writer's slots + writer's pad pairs + 2 counts | {token: spans -> rps-rows_reader, choice: spans -> 2, slot: spans -> pad list bounded by MAX_PAD_ROWS, others: n/a} | UInt32 | reader | writer | `assign_rows_block` -> writer's pushes | Concurrent with the scratch it is computed from; too small to alias (< 4 KB). |
| `reader_scratch` | `16*(num_groups+4) + 16*rps*32 + 4*(2*rows_reader*32 + 2*MAX_PAD_ROWS)` B | same | {token: spans -> rps (ids + ranks per slot), choice: spans -> 2, slot: spans -> MAX_PAD_ROWS, others: n/a} | UInt32 | reader | reader (no handshake) | whole call | Holds the count table, which the coordinator's multicast writes at a fixed address on every core, so it cannot move into a CB that is recycled. |

## Data-movement budget (R1, 8x512: T = 4096, R = 8416)

Implemented as planned: the DRAM crossings below hold for the built kernels (W read once as full tiles by the G senders; combine_weights written once, by the leader).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (6.3 MB) | 0 | resident shard read in place (L1 reads twice: untilize, matmul) | every dispatched row once: R x 1536 B = 12.9 MB (inherent to the gather), as 2 whole-piece writes per row (G=3, k=2; was 4 intersection writes before Refinement 3); plus the whole-piece stage gather: members 0 and 2 each read the 4 tiles of member 1's slice that fall in their piece, 2 x rps x 4 x 2 KB = 64 KB a group, 2 MB total |
| W (96 KB incl. padding) | 1 | G senders read their slice once | multicast of 96 KB to 110 cores |
| partial logits | 0 | live only in L1 | `(G-1) * rps * 2 KB` per member: 3 MB total |
| count table | 0 | - | 32 x 16 B gather + 512 B x 109 multicast |
| combine_weights | 1 (write, 1 MB full tiles) | routing_metadata's exact tile contents | - |
| combine_index | 1 (write, 32 KB) | - | - |
| expert_counts | 1 (write, 32 B) | - | - |
| dispatched (12.9 MB) | 0 | tilized in place into its L1 shards | (counted under x) |
| x, interleaved in (R2) | 1 (dram_interleaved) / 0 (l1_interleaved, one NoC read) | each token core reads its `split_work_to_cores` tile rows once into `cb_x_tiles` (one barrier, shared with the W reads on the sender) | no partial exchange (G = 1); `k` writes per pushed row instead of `G*k`-ish intersections |
| dispatched, DRAM out (R4) | 1 (write) | each owner tilizes its round-robin blocks into the 2-deep stream and its writer writes the `w` tiles of a block to DRAM (one flush a block) | none beyond R1 (pushes unchanged) |

Totals: DRAM ~1.1 MB (96 KB read, ~1.06 MB written); NoC ~18 MB (12.9 MB rows, 3 MB partials, 2 MB stage
gather, W and table multicasts); L1 local: x read twice (12.6 MB), row staging written + read once each,
tilize reads + writes 12.9 MB each.

> Cheapest-traffic split considered: the pinned split (token groups x hidden slices) with push gather -
> no other split avoids moving x at least once more (re-splitting to whole rows: +6.3 MB NoC) or adds a
> global exchange (pull gather: + gather_index build and distribution). Implemented: the same split
> (R1). The implemented split is the cheapest.
