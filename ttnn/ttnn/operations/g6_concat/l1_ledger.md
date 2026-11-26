# L1 Ledger: g6_concat

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md`
→ Blocking Model: `out_page` (extent `block_pages`), `slice`, `concat`, `inner` (all three folded
into `out_page`), `tile_row` (whole tile, 32 rows).

## Inventory (Rule 1, Rule 3 applied before any sizing)

- No compute kernel, so no intermediate CB between phases: the reader **packs into the destination**
  (`cb_out_tiles`) directly — the page_relocation regime has exactly one CB.
- row_gather needs one extra buffer, `cb_row_stage`, only because a 32 B face row cannot be read from
  DRAM to a 32 B-misaligned L1 offset (64 B DRAM read alignment, Blackhole). It cannot be removed
  by transforming in place (the misaligned piece must land somewhere with congruent alignment
  first), cannot alias `cb_out_tiles` (both live during `place_rows_block`), and is not allocated
  at all in the page_relocation regime.
- No budget solve, safety fraction or search exists: `block_pages = min(core_pages, BLOCK_PAGES_MAX)`
  with a declared constant, and the footprint below is closed-form and far under the budget.

## Ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_out_tiles` | `out_buffer_depth · block_pages` pages of `tile_bytes` | `out_buffer_depth · block_pages` (one block being filled by the reader while one drains to the writer) | `{out_page: spans → block_pages · out_buffer_depth, slice: streams (folded), concat: streams (folded), inner: streams (folded), tile_row: spans → 32 (whole tile)}` | the inputs' dtype as raw bytes (bf16 Phase 0; f32 / bf8b pages are byte-identical copies). Never passes through DEST, so audit 2 does not apply: a dataflow-only payload | reader | writer | whole program | Shares with nothing: it is the only CB of page_relocation; in row_gather it is live concurrently with `cb_row_stage` (placement reads stage → slot). Capacity = live set; the ×2 is the double-buffering mechanism (read/write overlap) |
| `cb_row_stage` (row_gather only) | `block_pages` pages of `STAGE_PAGES_PER_TILE · tile_bytes` (`STAGE_PAGES_PER_TILE = 2`) | `block_pages` tile windows (all windows of a block are staged before the single stage barrier) | `{out_page: spans → block_pages, slice: streams (folded), concat: streams (folded), inner: streams (folded), tile_row: spans → ≤ 2 × a tile's live bytes}` | inputs' dtype as raw bytes (dataflow-only) | reader | reader (same thread; scratch, no CB handshake) | `stage_rows_block` → `place_rows_block` of one block | Cannot alias `cb_out_tiles` (concurrent within the block). Capacity exceeds live bytes by ≤ 2× because of the DRAM-read alignment overhang (mechanism cap) — a 32 B bf16 row arrives inside a 64 B window. Not double-buffered: same-thread producer/consumer, nothing to overlap (a trid-pipelined variant is a perf lamp) |

## Symbol table

| Symbol | Bound | Predicate / source |
|--------|-------|--------------------|
| `block_pages` | `1 ≤ block_pages ≤ BLOCK_PAGES_MAX = 32` | `min(core_pages, BLOCK_PAGES_MAX)`, host |
| `out_buffer_depth` | `2` | host constant (Phase 0) |
| `tile_bytes` | 2048 (bf16), 4096 (f32), 1088 (bf8b) | output `buffer_page_size()`; TARGET dtypes only |
| `STAGE_PAGES_PER_TILE` | 2 | mechanism cap: holds iff `2 · face_row_bytes ≥ NOC_DRAM_READ_ALIGNMENT_BYTES` — true for bf16 / f32 on WH and BH; host asserts it when row_gather is selected |
| `core_pages` | `≤ ceil(tensor_out_pages / num_cores)` | `split_work_to_cores`; never enters a capacity except through `min(…, BLOCK_PAGES_MAX)` |
| `K` | `≤ K_MAX = 128` | host `ValueError`; enters no CB (args only) |

## Footprint

```
L1_op = out_buffer_depth · block_pages · tile_bytes
      + [row_gather] block_pages · STAGE_PAGES_PER_TILE · tile_bytes
```

Scales with `block_pages` (both terms) and `out_buffer_depth` (first term); with `tile_bytes`
(dtype). Nothing scales with an op dimension, K, or the grid.

| Case | page_relocation | row_gather | Budget |
|------|-----------------|------------|--------|
| bf16, `block_pages = 32` (Phase 0 max) | 2·32·2048 = 131072 B | 131072 + 32·2·2048 = 262144 B | 1347776 B |
| f32, `block_pages = 32` (refinement) | 262144 B | 524288 B | 1347776 B |
| bf8b, `block_pages = 32` (refinement) | 69632 B | — (deferred) | 1347776 B |

Every golden INPUT fits with ≥ 2.5× margin at the maximum block; no shape-dependent refusal exists.

## Data-movement budget

Chosen split: output-stationary contiguous output-page ranges over the full grid.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| each `x_k`, page_relocation | 1 (every stored page, padding rows included in the page) | each output page has one source page; each input page is the source of exactly one output page | none |
| each `x_k`, row_gather | 1 per logical row, in aligned windows ≤ 2× the live bytes (bf16 on BH) | each logical row belongs to exactly one output tile, owned by one core; the overhang is the 64 B DRAM-read alignment | none (the stage → slot move is core-local L1) |
| `out` | 1 | each output page written once, whole, by its owner core | none |

Totals per tier (examples):

| Case | DRAM read | DRAM write | Cross-core | Core-local L1 extra |
|------|-----------|------------|------------|---------------------|
| p2 (16 × (1,6,4096,256) bf16, dim 0) | 201326592 B | 201326592 B | 0 | 0 |
| p1 (f32, (108+84,4096,128), dim 0, refinement) | 402653184 B | 402653184 B | 0 | 0 |
| Qwen 79 (16 × (1,1,1,62080) bf16, dim 2) | 3973120 B (62080 windows of 64 B; 1986560 B live) | 3973120 B | 0 | 1986560 B (placement reads) |

> Cheapest-traffic split considered: output-stationary over the linear output-page axis — it is
> the minimum for page_relocation (every input and output crosses DRAM once, no cross-core traffic)
> and the cheapest reachable for row_gather (the only excess is the alignment overhang, ≤ 1× the
> live input bytes, versus 31× for whole-tile reads). Implemented: the same split. The implemented
> split is the cheapest-traffic split.
