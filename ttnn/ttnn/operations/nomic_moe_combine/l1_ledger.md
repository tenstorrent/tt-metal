# L1 Ledger: nomic_moe_combine

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Regime: `owner_computes` (the only built regime).
Block axes (op_design.md, Blocking Model): `token_rows` (block = `core_row_tiles` ≤ r, whole shard, streamed per tile
row), `hidden` (block = W, whole shard width), `choice` (top_k = 2, whole). `fp32_dest_acc_en = True` (enforced), so
every page that carries a DEST value between phases is Float32; the rows and the output are bf16 by contract.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_rows_choice0` | `rows_depth_tile_rows · W` (Phase 0 `r · W`), 2048 B | up to `core_row_tiles · W` (the reader runs ahead of compute by design) | {token_rows: spans → rows_depth_tile_rows (= r), hidden: spans → W, choice: streams → choice 0 only} | Float16_b (bf16 row-major values, the input dtype; never a DEST value) | reader (choice 0) | compute | kernel | none: concurrent with every other CB. Capacity = live set by choice (whole shard resident so read issue, the bound, never waits on CB space); lamp L6 measures depth 2 |
| `cb_rows_choice1` | `rows_depth_tile_rows · W`, 2048 B | as above | {token_rows: spans → r, hidden: spans → W, choice: streams → choice 1 only} | Float16_b | reader_noc1 (choice 1) | compute | kernel | none: a second CB rather than a 2-choice CB because each choice has its own producer RISC (one producer per CB), and tilize-A addresses a block from the CB read pointer, so each choice needs its own pointer |
| `cb_weights_raw_choice0` | `r`, 4096 B | `core_row_tiles` (pushed one tile row at a time as each lands; R2: resident until kernel end, R1: popped after expansion) | {token_rows: spans → r, hidden: streams → none (per-token values), choice: streams → choice 0} | Float32 (the input's fp32; faces 0, 2 written) | reader (choice 0) | compute | kernel start → end of `expand_weights_block` | could alias `cb_rows_choice0`'s tail only if expansion ran after rows stopped arriving — it runs before; in-place expansion into this CB would make compute a second producer of a reader-produced CB (ownership rule). 2 of each 4 KB page unused (faces 1, 3): the column-broadcast unpack needs face 2 at +2 KB |
| `cb_weights_raw_choice1` | `r`, 4096 B | `core_row_tiles` | {token_rows: spans → r, hidden: streams → none, choice: streams → choice 1} | Float32 | reader_noc1 | compute | as above | as above |
| `cb_weights_bcast` (realization R1 only; not allocated under the default R2) | `2 · r`, 4096 B | `2 · core_row_tiles` (resident, tile `2·tb + k`) | {token_rows: spans → r, hidden: streams → none (broadcast inside the tile), choice: spans → 2} | Float32 (TF32-valued DEST result; Float32 so the multiply's srcB re-reads it losslessly — a 16-bit page would round the weights a second time and fail the precision bar) | compute (pack) | compute (unpack) | kernel | not aliased with `cb_weights_raw_*`: both are live during the expansion (unpack raw, pack bcast). Lamp L5 (per-tile-row expansion) would shrink it to 2 pages |
| `cb_bias_tiles` | `W`, 4096 B | `W` (resident; row 0 of faces 0, 1 of each tile; tile 0 bytes 2048..2048+W·128 hold the staged row until 2W local NoC reads place it — no RISC copy, which measured 2-3.5 us) | {token_rows: streams → none (constant over tokens), hidden: spans → W, choice: streams → none} | Float32 (bf16 bias widened exactly; same srcB format as the weights, so no srcB reconfig per DEST group — lamp L7 measures bf16 at half the L1) | reader (choice 0) | compute | kernel | the staging buffer lives inside it (tile 0, faces 2-3) instead of a separate scratch CB; not shared further: live for the whole kernel |
| `cb_index_choice0` | `r`, 128 B | `core_row_tiles` (32 uint32 per tile row) | {token_rows: spans → r, hidden: streams → none, choice: streams → choice 0} | UInt32 | reader (choice 0) | reader (choice 0) (private scratch, no handshake) | kernel | not mergeable with choice 1's: different producer RISC; 128 B a tile row, kept whole so no index is re-read |
| `cb_index_choice1` | `r`, 128 B | `core_row_tiles` | {token_rows: spans → r, hidden: streams → none, choice: streams → choice 1} | UInt32 | reader_noc1 | reader_noc1 | kernel | as above |
| `cb_output_shard` | `r · W`, 2048 B — **backed by the output tensor's shard** (`ttnn.cb_descriptor_from_sharded_tensor`), not counted against the CB budget | `core_row_tiles · W` | {token_rows: spans → r, hidden: spans → W, choice: streams → reduced away} | Float16_b (the one bf16 rounding of the fp32 DEST: the output dtype, by contract) | compute | — (the output tensor; never popped) | kernel | it **is** the destination (pack-into-destination, zero-copy); no writer, no intermediate output CB |
| `cb_rows_tilized` (realization R2 — **the implemented default**; 0 pages under R1) | `2 · W`, 2048 B | `2 · W` (one tile row, both choices) | {token_rows: streams → 1, hidden: spans → W, choice: spans → 2} | Float16_b (tilized copy of bf16 values: exact) | compute (pack) | compute (unpack) | per tile row | R1 (unpack-tilize pairing) is correct but measured compute-bound: 8x512 32.7 us vs 16.8 under R2 (tilize-A unpacks one 16-datum row per instruction, ~380 cycles a tile product). R2 replaces `cb_weights_bcast` (the raw weights feed `mul_tiles_bcast_cols` directly), so the two are never both allocated; knob `FUSED_TILIZE` |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `r` = `shard_row_tiles` | shard height in tiles (`nd_shard_spec.shard_shape[-2] / 32`) | 1 ≤ r; r ≤ 4 in every norm plan (`MAX_ROWS_PER_SHARD`); generically bounded by the budget predicate below | `validate()`: `circular_buffer_bytes(...) ≤ 384 KB` |
| `W` = `shard_col_tiles` | shard width in tiles | 24 % W == 0 and W ≤ 16 ⇒ W ∈ {1, 2, 3, 4, 6, 8, 12}; the norm uses 8 (G = 3) or 12 (G = 2) | `validate()` (bias-staging mechanism cap) |
| `core_row_tiles` | this core's shard tile rows | 1 ≤ core_row_tiles ≤ r | plan (shard geometry) |
| `rows_depth_tile_rows` | rows CB depth knob | Phase 0 = r | host constant |
| `batch_reads`, `batch_depth` | reads in flight per RISC | 8 x 2 (measured, op file `BATCH_READS` / `BATCH_DEPTH`); `batch_depth ≥ 2`, `32 % batch_reads == 0` | kernel `static_assert` |
| transaction ids | row batches + index + bias staging + bias placement + one per tile row's weight tile | `batch_depth + 3 + r ≤ 15` ⇒ r ≤ 10 at depth 2 | `validate()` (`_shard_geometry`: `NOC_MAX_TRID - BATCH_DEPTH - SIDE_TRIDS`) and the reader's `static_assert` |

## Total per-core footprint (CBs beyond the io tensors)

```
cb_bytes(r, W) = 2·r·W·2048            # cb_rows_choice0/1         (scales with r·W: token_rows x hidden)
               + 2·r·4096              # cb_weights_raw_choice0/1  (scales with r)
               + 2·r·4096              # cb_weights_bcast          (scales with r)
               + W·4096                # cb_bias_tiles             (scales with W)
               + 2·r·128               # cb_index_choice0/1        (scales with r)
             [ + 2·W·2048 ]            # cb_rows_tilized, realization R2 (implemented default)
             [ - 2·r·4096 ]            # cb_weights_bcast absent under R2
             R1: 4096·r·W + 16640·r + 4096·W      R2 (default): 4096·r·W + 8448·r + 8192·W
```

| Plan (example shape) | r | W | R1 bytes | R2 bytes |
|----------------------|---|---|----------|----------|
| G = 2, 1 row (2x512, 1x1000, 1x1024, 3x512, 4x288) | 1 | 12 | 114,944 (112 KB) | 155,904 (152 KB) |
| G = 3, 2 rows (4x512, 8x256) | 2 | 8 | 131,584 (129 KB) | 147,968 (145 KB) |
| G = 2, 2 rows (5x512, 6x512, 8x384) | 2 | 12 | 180,736 (177 KB) | 213,504 (209 KB) |
| G = 3, 4 rows (8x512, 7x512, 1x3904, 1x4000) | 4 | 8 | 230,400 (225 KB) | 230,400 (225 KB) |
| worst plannable (r = 4, W = 12) | 4 | 12 | 312,320 (305 KB) | 328,704 (321 KB) |

All under the 384 KB CB ceiling. The R2 column is what the default realization allocates (`4096·r·W + 8448·r + 8192·W`;
the absent `cb_weights_bcast` is not counted). `circular_buffer_bytes(num_tokens, batch, memory_config)` and validate()'s
budget check sum `circular_buffer_specs(...)` in the program-descriptor module — the same list the descriptor allocates
from, so the reported figure cannot drift from the allocation (verifier fix: it was a parallel formula before).

## Data-movement budget (chosen split: `owner_computes` + `group_shared_index_weights`, Refinement 2)

At 8x512 (T = 4096, R = 8416, G = 3, 96 cores); 2x512 / 4x512 in brackets.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `expert_rows` (L1, 12.9 MB allocated) | 0 (L1-resident input) | Phase 0 regime is rows in L1 | **12.6 MB** [3.1 / 6.3 MB]: every used row's 1536 B read exactly once, as G pieces of W·64 B (512 B; 768 B at G = 2) — the minimum; unused rows never read |
| `combine_index` (32 KB) | 1x (32 KB) [8 KB; 16 KB] | `SHARE_INDEX`: only the group leader's reader k reads its slice (one read per group per choice); was 3x (96 KB) when every owner read its own | (G-1)·`core_row_tiles`·128 B a group per choice: 64 KB [8 KB; 32 KB] of L1→L1 unicast writes + one semaphore increment per peer |
| `combine_weights` (1 MB) | 0.5x (0.5 MB) [128 KB; 256 KB] | `SHARE_WEIGHTS`: faces 0, 2 (2 KB of 4 KB) of each tile, read once per row group by its leader; was 1.5x (1.5 MB) when each of the G owners read them | (G-1)·2 KB per tile row per choice: 1.0 MB [128 KB; 512 KB] of L1→L1 unicast writes (two 1 KB faces per peer), semaphore bumped once the writes are acked |
| prepared bias (24 KB) | 96 KB [96 KB; 96 KB]: W·128 B per core | one DRAM read per core (staged in bias tile 0), spread over the 8 bank copies; placed by 2W local 64 B NoC reads (self, no DRAM). 2W 64 B DRAM reads a core measured ~3 us slower (1536 small requests back up at DRAM and stall the row reads) | 2W·64 B local (self) |
| output (6.3 MB) [1.6 / 3.1 MB] | 0 | packed into the core's own shard | **0** (no NoC write at all) |

Totals at 8x512: DRAM ≈ 0.65 MB read (was 1.7 MB before Refinement 2); NoC ≈ 12.6 MB of row reads (split across NOC0
(choice 0) and NOC1 (choice 1)) + ≈ 1.06 MB of index/weight unicast writes from the group leaders to their peers. No
output bytes cross the NoC. No CB was added: the forwarded bytes land in the peers' existing `cb_index_choice*` /
`cb_weights_raw_choice*` at the same L1 offsets; two semaphores (one per choice) were added.

> **Update (Refinement 2): `group_shared_index_weights` is now built** (the deferral reason below was falsified by
> measurement, see the verifier audit). 8x512 14.9 → 12.9–13.5 us.
>
> Cheapest-traffic split considered: `owner_computes` (token rows x hidden slice on the shard owner) — it is also the
> implemented one: 12.6 MB NoC, no output traffic. The nearest alternatives add traffic: `whole_row_remote_write`
> +4.2 MB of cross-core tile writes (for 3x fewer, larger reads); `group_shared_index_weights` trades 1.0 MB of DRAM
> reads for 1.0 MB of cross-core writes plus a rendezvous. Implemented: `owner_computes`. Deferred (regime table):
> `group_shared_index_weights`, because every DRAM byte it saves is prefetched at kernel start and consumed only by
> compute, which lags the row reads — off the critical path — while its rendezvous would sit ahead of each peer's first
> row read; the structure keeps it reachable (CBs at identical addresses on every core, a semaphore wait replaces the
> DRAM wait).

## Implementation notes (measured, ttnn-implementer)

- The weight faces (1.5 MB of DRAM at 8x512) are **on** the critical path, contrary to the regime table's premise for
  deferring `group_shared_index_weights`: ablating the weight reads takes 8x512 from 14.7 to 12.8 us (readers 11.2 ->
  8.7 us) because the grid's DRAM traffic slows every core's first tile row of L1 row reads (3.3-6 us instead of
  ~1.9). Spreading the weight reads over the gather, a separate request VC, and one 3 KB read per tile were all
  neutral or worse (op file knob comments); cutting the bytes (group sharing, G-fold) is the remaining lever.

## Verifier audit (2026-11-04)

- **Deferral reason for `group_shared_index_weights` no longer holds.** The regime table defers it as "off the critical
  path"; the implementation note above and the verifier's re-measurement both say otherwise (8x512: 14.86 us with
  the weight reads, 12.03 us without them; 4x512 8.77 -> 7.88). The cheapest-DRAM-traffic decomposition therefore no
  longer has a positive reason to stay deferred. It is filed as a perf refinement in `op_requirements.md`
  (Refinement 2), not left as a deferred row.
- Ledger currency fixed: `batch_reads x batch_depth` is 8 x 2 (it said 4 x 2); the transaction-id bound is
  `batch_depth + 3 + r <= 15` (one id per tile row's weight tile; it said `batch_depth + 3 <= 15`) and is now enforced
  in validate(); the R2 totals table had counted the absent `cb_weights_bcast` (2·r·4096 too high in every row).
