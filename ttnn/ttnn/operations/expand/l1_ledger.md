# L1 Ledger: expand (Phase 0, regime R-H)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target: quasar (4 MiB L1 per
cluster, ring ≤ 1,048,560 B). One buffer; no compute kernel, so no DEST and no page-format
conversion anywhere.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_row_bcast_tiles` | `depth · block_src_tiles`, with `block_src_tiles = min(max_src_tiles_per_core, block_src_tiles_cap)` and `depth = min(buffer_depth, max_blocks_per_core)` (verifier, Phase 0: the depth is clamped to the blocks the busiest cluster runs) | `block_src_tiles` while one block is being fanned out; `2 · block_src_tiles` when the reader builds block `b+1` during the writer's fan-out of block `b` (only for cores with ≥ 2 blocks, which is exactly when `depth` is 2) | `{source tiles (j, p): spans → block_src_tiles; output tile rows r (expanded): streams → 0 extra pages (every row of a segment is a write from the same resident entry); rows inside a tile: spans → whole page (in-place build)}` | the input dtype's page (bf16 tile 2,048 B; float32 4,096 B later). Dataflow-only payload: bytes are never unpacked, so audit 2's DEST-width question does not apply | reader | writer | whole program (every block) | nothing to share with: it is the op's only buffer. The input page is read straight into the entry and built in place (no separate input buffer, no separate built-tile buffer); a separate output buffer is unnecessary because the writer fans out of the same entry. Capacity exceeds the one-block live set by the depth factor only when a cluster runs ≥ 2 blocks: an explicit pipelining decision (double buffering, depth knob `buffer_depth`). With one block per cluster (the default whenever the busiest run fits the cap, e.g. every Phase 0 INPUTS entry) `depth` clamps to 1, since a second half would never be live |

## Symbol table

| Symbol | Meaning | Bound | Predicate / source of the bound |
|--------|---------|-------|---------------------------------|
| `page_size` | bytes per tile page | 2,048 (bf16), 4,096 (float32) | output `buffer_page_size()`; SUPPORTED dtype |
| `buffer_depth` | blocks in flight (the knob) | 2 (Phase 0) | host constant |
| `max_blocks_per_core` | blocks the busiest cluster runs | `ceil(max_src_tiles_per_core / block_src_tiles)` | computed on host |
| `depth` | blocks the ring holds | `min(buffer_depth, max_blocks_per_core)` ≤ `buffer_depth` | computed on host |
| `max_src_tiles_per_core` | most source tiles any engaged core's run touches | `≤ ceil(max_run_tiles / fan_out_tiles) + 1` and `≤ num_sources` | computed on host from the `split_work_to_cores` runs |
| `block_src_tiles_cap` | largest block the ring may hold | `floor(min(DFB_RING_MAX_BYTES, l1_budget) / (buffer_depth · page_size))` = 255 (bf16), 127 (float32) on quasar | `DFB_RING_MAX_BYTES = 1,048,560` (target page cluster model; no Python query); `l1_budget = ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` (query; value on quasar ≤ 4,194,304 B, exact figure not on the page) |
| `l1_reserve_bytes` | L1 held back from the budget | 0 (Phase 0) | host constant |
| `block_src_tiles` | source tiles per block | `≤ block_src_tiles_cap` | `min(...)` above — the clamp is the bound; `W`, `H_out` and the plane count are therefore unbounded in the input without affecting L1 |
| `fan_out_tiles` | `output_h_tiles = ceil(H_out / 32)` | unbounded — appears in no capacity expression (streamed) | — |

## Total per-core footprint

`footprint = depth · block_src_tiles · page_size`, `depth = min(buffer_depth, max_blocks_per_core)`

- scales with `block_src_tiles` (source-axis block knob) and `buffer_depth` (depth knob, clamped to the blocks run), and with
  dtype through `page_size`; independent of `H_out`, of `fan_out_tiles` and of the total output size;
- bounded by `buffer_depth · block_src_tiles_cap · page_size ≤ min(1,048,560 B, l1_budget)`;
- worked (quasar, bf16): traced call `1 · 3 · 2,048 = 6,144 B` (one block per cluster); largest Phase 0
  INPUTS entry `(1,8192)→(17,-1)`: `1 · 8 · 2,048 = 16,384 B` (float32 later: 32,768 B). With the
  block forced smaller (≥ 2 blocks per cluster) the depth is 2: e.g. traced call with
  `block_src_tiles = 1`: `2 · 1 · 2,048 = 4,096 B`. Under 1 % of L1 in every case.

## Data-movement budget (chosen split: R-H, source-major contiguous runs)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input `[1, W]` | `≤ 1 + (engaged_cores − 1) / num_sources` whole-tensor equivalents: each source page read once per core whose run touches it (`≤ num_sources + engaged_cores − 1` page reads). Traced call: 96 page reads of 80 = 1.2 crossings | each core keeps the sources of its block resident while it writes every output tile they feed, so no source is re-read within a core; only a run boundary inside a source's rows reads it on two cores | 0 |
| output `[H_out, W]` | 1 | every output page is written exactly once, from the resident built tile | 0 |
| (local) row replication | not DRAM | ~1 page of self-aimed L1→L1 copies per source read (`(1+2+4+8)·2·face_row_bytes + 2·face_bytes` = 1,984 B per bf16 tile) | 0 |

Totals, traced call `(1,2560)→(1330,-1)` (quasar, bf16): DRAM read 96 · 2,048 = 196,608 B; DRAM
write 3,360 · 2,048 = 6,881,280 B; cross-core 0 B; local L1 copies ≈ 96 · 1,984 = 190,464 B.

> Cheapest-traffic split considered: column-aligned source-major runs (R-H-colalign) — DRAM reads
> −16 pages (−32,768 B, 0.5 % of the 7,077,888 B moved on the traced call); DRAM writes and
> cross-core unchanged. Implemented: source-major contiguous runs balanced to ±1 output tile
> (R-H). Deferred because column alignment costs up to one source column of imbalance per core
> (traced: busiest core 126 instead of 105 output tiles, +20 % on the critical path, against a
> 0.5 % byte saving) and caps engaged cores at `num_sources` (W = 17 → 1 core); it stays
> reachable as a host-split change alone, since the kernels already take arbitrary runs.
