# L1 Ledger: multiply

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target: `quasar` (ttsim), budget
`ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 B per cluster on this
target. Thread plan (1, 1, 1): every capacity is per NEO × C with C = 1. Block axes (op_design.md
Blocking Model): `n`, `h` (`block_rows`), `w` (`block_cols`), and `flat` (the `flat_stream` 1-D run,
`core_tiles`). A program holds one regime, so `cb_in_b_tiles` and `cb_in_b_window` never coexist (same
slot, chosen by the host).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_in_a_tiles` | `stream_depth · C` = 4 | 1 tile (compute's front tile) | `{n: streams, h: streams, w: streams, flat: streams → window stream_depth}` | bf16 (a's dtype: dataflow payload unpacked to SrcA, never through DEST) | reader | compute | whole kernel, both regimes | capacity 4 > live set 1 for in-flight reads: implicit sync splits the ring into 2 txn-id halves (DFB.md C4) so one half fills while the other is credited — depth is the mechanism reason. Cannot share with `cb_in_b_*` or `cb_out_tiles`: all three are live concurrently (streaming pipeline) |
| `cb_in_b_tiles` (`flat_stream` only) | `stream_depth · C` = 4 | 1 tile | `{n: streams, h: streams, w: streams, flat: streams → window stream_depth}` | bf16 | reader | compute | whole kernel | same depth reason as a; concurrent with a and out, so no aliasing; replaced (not added) by `cb_in_b_window` in the other regime |
| `cb_in_b_window` (`row_bcast_strips` only) | `b_window_depth · block_cols · C` = 2 · block_cols | `block_cols` tiles (the held window of the current block; `cols ≤ block_cols` of them filled) | `{n: streams (one window per image row-segment), h: streams (window reused across block_rows — the reuse this buffer exists for), w: spans → block_cols, flat: n/a in this regime (not a block axis of row_bcast_strips)}` | bf16 | reader | compute | one window per block; two windows resident (current + prefetched) | capacity = 2 × live set: the second window is the reader's prefetch of block k+1 AND a liveness requirement (Mechanism caps: with one window the reader would block before completing block k's last implicit-sync a batch). Nominal `block_cols` windows (not `cols`) keep windows aligned to the ring so indexed reads never straddle its end. Cannot alias a or out: concurrent |
| `cb_out_tiles` | `stream_depth · C` = 4 | 1 tile (the packer's current tile) | `{n: streams, h: streams, w: streams, flat: streams → window stream_depth}` | bf16 = output dtype. DEST is fp32 (`fp32_dest_acc_en=True`); a 16-bit page under 32-bit DEST is audit 2's *under* case, answered by the output ABI: the output tensor is bf16 and the product is rounded once at pack, which is what R6 asks for. A float32 output (`dtype=float32`, deferred) makes this page Float32 | compute | writer | whole kernel | depth 4 for write overlap (2 txn-id halves); concurrent with the inputs, so no aliasing; no intermediate CB exists to fold it into — compute packs straight into the destination ring (Rule 3, pattern 1) |

Inventory check (Rule 1): one ring per operand and one for the output is the floor for a streaming
binary op whose operands come from separate tensors; there is no intermediate phase, so no scratch
buffer, no in-place candidate and no alias candidate. No budget solve exists: the only knob the budget
bounds is `block_cols`, through a closed-form clamp.

## Symbol table

| Symbol | Bound | Predicate / source |
|--------|-------|--------------------|
| `C` | 1 | `THREAD_PLAN = (1, 1, 1)`; host asserts C = 1 for `row_bcast_strips` |
| `stream_depth` | 4 (constant) | host constant |
| `b_window_depth` | 2 (constant, ≥ 2 required) | host constant + assert |
| `block_cols` | `1 ≤ block_cols ≤ min(Wt, max_units_per_core, block_cols_cap)` | host derivation, single source; `block_cols_cap = floor(min(DFB_RING_MAX_BYTES, budget − stream_depth·C·(a_tile_bytes + out_tile_bytes)) / (b_window_depth · C · b_tile_bytes))` (each ring with its own page size; verifier fix — it used the output's page size for all three); `DFB_RING_MAX_BYTES` = 1,048,560 (target page) → 255 bf16 tiles here |
| `block_rows` | ≤ `ceil(Ht / G)` | derived from `G`; no buffer scales with it (a streams) |
| `G` | `1 ≤ G ≤ Ht` | `clamp(ceil(num_cores / (N_out · Wt)), 1, Ht)`, `num_cores` from `device.compute_with_storage_grid_size()` |
| `tile_bytes` (`a_`, `b_`, `out_`) | 2,048 each (bf16) in Phase 0; 4,096 for a float32 tensor | `tensor.buffer_page_size()` of that ring's tensor |
| `budget` | 3,963,328 B on this target | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`, `l1_reserve_bytes = 0` |

## Footprint

- `flat_stream`: `3 · stream_depth · C · tile_bytes` = 24,576 B — constant; scales with `stream_depth`
  (and C) only.
- `row_bcast_strips`: `2 · stream_depth · C · tile_bytes + b_window_depth · block_cols · C · tile_bytes`
  = 16,384 + 4,096 · block_cols B; scales with `block_cols` (w) and `b_window_depth`; never with
  `block_rows`, `n` or any op dimension. Worked: `[1,4,33600]×[1,1,33600]` (block_cols 33) → 151,552 B;
  at the cap (255) → 1,060,864 B, 27% of the budget.

## Data-movement budget

Counted in whole-tensor crossings of the tensor (DRAM or L1 interleaved) ↔ cluster boundary.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| a | 1 | every regime streams each a page exactly once | none |
| b (`flat_stream`) | 1 | read alongside a, same page ids | none |
| b (`row_bcast_strips`) | `G` (1 on every INPUTS over_h entry except `[128,512]×[1,512]`, G = 2) | the window is held across the block's `block_rows`; the split cuts the axes b varies along (n, w) first, and cuts h into `G` groups only to fill the grid | none |
| out | 1 | written once | none |

Totals per tier (bf16): DRAM-or-L1 tensor tier = `(|a| + G·|b| + |out|) · 2,048 B`; cross-core = 0;
core-local re-reads: b window tiles re-read from L1 `block_rows` times per block (SrcB unpack), a and
out once. Examples: `[1,4,8400]×[1,1,8400]` = 3 × 263 tiles = 1,615,872 B (minimum);
`[2,4,128,512]×[2,4,1,512]` = (512 + 128 + 512) tiles = 2,359,296 B (minimum; the rejected flat split
would move 3,145,728 B).

> Cheapest-traffic split considered: cut (n, w) across clusters and keep h whole (b read once), plus a
> multicast of b to the clusters of one row group when `G > 1` — delta vs. implemented: `(G−1)·|b|`
> tensor-tier reads replaced by `(G−1)·|b|` NoC multicast bytes; for `[128,512]×[1,512]` that is
> 16 tiles (32 KB). Implemented: `row_bcast_strips` with G row groups (b read G times) and
> `flat_stream` (already the minimum). Deferred because `G > 1` only occurs when `N_out·Wt < num_cores`,
> so the extra b reads are fewer than `num_cores` tiles per call — less than one tile per cluster —
> while the multicast would add a sender/receiver handshake per window; the deferred `leading_reuse`
> row in op_design.md carries the same multicast contract, and `load_b_window` on an explicit-sync
> ring is the one operation it would replace.
