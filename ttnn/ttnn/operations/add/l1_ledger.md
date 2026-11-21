# L1 Ledger: add

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Design: `op_design.md` (same directory).
Worked values for `quasar` (bf16 tiles of 2,048 B, Phase 0 knobs: `block_tiles` = 4, `RING_DEPTH_BLOCKS` = 2, C = 1).
Block axes of the Blocking Model: `tiles` (flat page axis, extent `block_tiles`), and the folded axes `leading`, `tile_rows`, `tile_cols`.

## Ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_a_tiles` | `RING_DEPTH_BLOCKS × block_tiles × C` (8) | `block_tiles × C` (4): the block compute is adding; the second block is the one the reader fills meanwhile | `{tiles: spans → block_tiles (× depth), leading: streams (folded), tile_rows: streams (folded), tile_cols: streams (folded)}` | a.dtype = Float16_b. Dataflow-only payload: tensor pages copied unchanged; the unpacker feeds srcA, so the page does not follow DEST width | reader | compute | whole program, every block | Not shared with `cb_input_b_tiles`: both are live for every block (concurrent). Not aliased with `cb_output_tiles`: output block k drains while input block k+1 fills (pipelining decision). Not packed in place: the ring has a dataflow producer and the writer would become a second consumer. Capacity − live set = one block of double buffering (overlap of read with add) |
| `cb_input_b_tiles` | `RING_DEPTH_BLOCKS × block_tiles × C` (8) | `block_tiles × C` (4) | `{tiles: spans → block_tiles (× depth), leading: streams (folded), tile_rows: streams (folded), tile_cols: streams (folded)}` | b.dtype = Float16_b; same mechanism reason as a | reader | compute | whole program, every block | Not shared with `cb_input_a_tiles` (concurrent, both operands of every add); not aliased with `cb_output_tiles` (pipelined). Gap = double buffering |
| `cb_output_tiles` | `RING_DEPTH_BLOCKS × block_tiles × C` (8) | `block_tiles × C` (4): the block being packed; the other is draining to the writer | `{tiles: spans → block_tiles (× depth), leading: streams (folded), tile_rows: streams (folded), tile_cols: streams (folded)}` | out.dtype = Float16_b under fp32 DEST. Output ABI: the output tensor is bfloat16 (R8: the fp32 DEST sum rounded once into the output dtype); no later phase reads this ring, so no accumulation is truncated | compute | writer | whole program, every block | Cannot pack into an input ring (those have the reader as producer and compute as consumer; a writer consumer would be a second consumer). Not aliased with the inputs (pipelined, concurrent). Gap = double buffering (pack of block k overlaps the write of block k−1) |

No intermediate ring: the sum lives in DEST for one block and is packed straight into the destination ring (Rule 3, pattern 1). No scalar, mask or constant ring.

## Symbol table

| Symbol | Bound | Predicate / source that establishes it |
|--------|-------|----------------------------------------|
| `block_tiles` | 1 ≤ `block_tiles` ≤ `DEST_AUTO_LIMIT` (4 at fp32 half-sync, 8 at 16-bit half-sync or fp32 full-sync, 16 at 16-bit full-sync) | host `max(1, min(BLOCK_TILES, dest_section_tiles(config), ceil(max_core_tiles / C)))` (`add_program_artifacts.py`; `dest_section_tiles` mirrors `get_dest_limit`); kernel `static_assert(block_tiles <= DEST_AUTO_LIMIT)` (`dest_helpers.hpp:89-103`) |
| `RING_DEPTH_BLOCKS` | ≥ 2; 2–4 is the perf-lamp range | host check `ring_depth >= 2` in `create_program_artifacts`; no explicit upper bound — the per-ring and footprint asserts below bound it |
| `C` | {1, 2, 4} | `validate_thread_plan` (target page *Thread plan*) |
| `tile_bytes` | 2,048 (bf16) in Phase 0; ≤ 4,096 (float32) in TARGET | `buffer_page_size()` of each tensor; bfloat8_b unbuildable on quasar |
| one ring | `RING_DEPTH_BLOCKS × block_tiles × C × tile_bytes` ≤ 1,048,560 B | host assert (target page: ring cap) |
| budget | `ttnn.get_max_worker_l1_unreserved_size()` (3,963,328 B on quasar, probed 2026-11-21); no reserve knob (no solve) | host assert `footprint ≤ budget` |

## Footprint

`footprint = 3 × RING_DEPTH_BLOCKS × block_tiles × C × tile_bytes`

- Every term scales linearly with `RING_DEPTH_BLOCKS`, `block_tiles` and `C`; with `tile_bytes` per ring's own tensor.
- Phase 0 on quasar: 3 × 2 × 4 × 1 × 2,048 = **49,152 B** (1.2% of the budget).
- Largest over the bounded knobs (depth 4, block 16, C 4, float32): 3 × 4 × 16 × 4 × 4,096 = 3,145,728 B: still under the queried budget, so no knob needs a budget solve; the host assert stays.
- No term depends on an op dimension.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| a | 1 | streamed page by page, consumed once; nothing is re-read | 0 |
| b | 1 | same (no broadcast in Phase 0, so no reuse to exploit) | 0 |
| out | 1 | each page packed once and written once | 0 |

For L1 operands the tier is the L1 banks over the NoC instead of DRAM, with the same counts.

Height-sharded operands (Refinement 0): the tier is the shard-holding cores' L1, still crossed over the NoC
through each tensor's `TensorAccessor` (a page is fetched from whichever core holds its shard, which is in general
not the core that adds it), with the same counts: a, b and out once each, 0 cross-core traffic beyond those
crossings. No ring is added, resized or backed on a shard, so the ledger above is unchanged. Consuming each shard
in place on the core that holds it (a zero-copy ring over the shard) would remove those NoC crossings; the
refinement left that out on purpose, and it is a separate performance refinement.

Totals per tier (tile-padded pages, bf16):

| Shape | Pages per tensor | Read (a + b) | Written (out) | Cross-core |
|-------|------------------|--------------|---------------|------------|
| `[1,2,33600]` | 1,050 | 4,300,800 B | 2,150,400 B | 0 |
| `[1,2,8400]` | 263 | 1,077,248 B | 538,624 B | 0 |
| `[1,1,1600,128]` | 200 | 819,200 B | 409,600 B | 0 |
| general | `N = prod(leading) × ceil(H/32) × ceil(W/32)` | `2 × N × tile_bytes` | `N × tile_bytes` | 0 |

> Cheapest-traffic split considered: flat page ranges over the full grid — every candidate split (flat by tile, flat by block, by tile-row, by tile-column) moves exactly these bytes, since each page of a, b and out crosses its tier once and no cluster needs another's data. Implemented: flat page ranges at tile granularity, `row_wise=True`. The implemented split is also the cheapest; occupancy and balance chose it among equals (tile-row splits strand the one-tile-row YOLOv8 shapes on a single cluster).
