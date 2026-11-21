# L1 Ledger: concat

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target: `quasar` (worked values: bf16, `tile_bytes` = 2,048 B).

Block axes from the Blocking Model:

- `out_tile`: extent `block_tiles`.
- `piece`: whole extent, not split.
- `in_tile`: whole tile, not split.

`image`, `concat` and `cross` are folded into `out_tile`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_output_tiles` | `OUTPUT_DEPTH · block_tiles` (Phase 0: 2 · 8 = 16 pages = 32,768 B) | `2 · block_tiles` pages: one block being filled by the reader and one being drained by the writer | `{out_tile: spans → block_tiles × OUTPUT_DEPTH, piece: streams → the runs of up to count+1 pieces land in one entry and none is held, in_tile: spans → one whole tile per entry (runs fill sub-ranges of it), image/concat/cross: folded into out_tile}` | the output dtype (bfloat16). The payload is dataflow only and never goes through DEST, so audit 2 does not apply | reader | writer | whole program (both kernels, every block) | **No other buffer exists to share with.** It is the destination itself: runs are NoC-read straight into the entry (Rule 3, pattern 1). Capacity exceeds one block by exactly one block, for double buffering (reader fills block i+1 while writer drains block i), and the capacity is a whole multiple of `block_tiles` so no reservation wraps |

## Symbol table

| Symbol | Bound | Predicate / source establishing it |
|--------|-------|-----------------------------------|
| `block_tiles` | `1 ≤ block_tiles ≤ min(BLOCK_TILES, max_core_tiles, ring_cap_tiles, budget_tiles)` | host: `block_tiles = max(1, min(BLOCK_TILES, max_core_tiles, ring_cap_tiles, budget_tiles))`, single source `BLOCK_TILES` (Phase 0 = 8) |
| `OUTPUT_DEPTH` | 2 (constant) | host constant |
| `tile_bytes` | 2,048 for bf16 (4,096 for 32-bit dtypes later; 1,024 for uint8) | `out.buffer_page_size()` |
| `ring_cap_tiles` | `floor(RING_MAX_BYTES / (OUTPUT_DEPTH · tile_bytes))` = 255 (bf16) | `RING_MAX_BYTES = 1,048,560` (target page, cluster model; a named host constant, since Python has no query) |
| `budget_tiles` | `floor((ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES) / (OUTPUT_DEPTH · tile_bytes))` | run-time query. Its value on `quasar` was not probed (≤ 4,194,304 B per cluster) |
| `max_core_tiles` | `ceil(total_out_tiles / num_cores)` | `ttnn.split_work_to_cores` |
| `count` | `2 ≤ count ≤ 3` in Phase 0 (`≤ MAX_INPUTS = 5` in TARGET). Sets no buffer size | `validate()` |

**Total per-core footprint:** `OUTPUT_DEPTH · block_tiles · tile_bytes` = 32,768 B at Phase 0 on `quasar`. It scales linearly with `block_tiles` (knob `BLOCK_TILES`) and `OUTPUT_DEPTH`, and with `tile_bytes` (dtype). It is independent of every op dimension, of `count`, and of the regime. No budget solve is needed: the inventory is one buffer, and at the Phase 0 knobs its footprint is under 1% of L1.

## Data-movement budget

Chosen split: contiguous ranges of flat output tiles over the full grid (`split_work_to_cores(..., row_wise=True)`).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| each input `in_k` (`page_copy`) | 1: every page read once, whole | each output page is exactly one input page, and the split assigns each output page to one core | none |
| each input `in_k` (`run_assembly`) | ≤ 1: only bytes that land in the output are read, each once. Input padding along `dim` is never read | runs are NoC-read straight into the destination entry, and a source page straddling two output tiles is read in two disjoint parts (possibly by two cores) | none |
| `out` | 1: every page written once, whole | one writer per page | none |

**Totals per tier.**

- DRAM read is `Σ_k pages(in_k) · tile_bytes` in `page_copy`, and `≤ total_out_tiles · tile_bytes` in `run_assembly`.
- DRAM write is `total_out_tiles · tile_bytes`.
- Cross-core traffic is 0.
- Core-local L1 traffic is the ring only: each byte is written once by the NoC and read once by the NoC.

Transactions: 1 read and 1 write per output tile in `page_copy`. In `run_assembly` reads are 2–6 per tile (row-granular `dim=-2`), ≤ 8 per tile (face-granular `dim=-1`) or ~64–160 per tile (element-granular `dim=-1`), and writes stay 1 per tile.

> Cheapest-traffic split considered: contiguous flat output-tile ranges with direct sub-page reads, which moves exactly the minimum bytes per tier (each input byte that lands in the output crosses DRAM once, each output page once, no cross-core traffic). Implemented: that same split. It is also the cheapest-traffic split. The only alternative that differs is `staged_shift` (deferred). It moves **more** DRAM bytes: whole source pages plus a halo page per piece per segment, versus `run_assembly`'s exact runs. It trades those bytes for fewer, larger transactions on element-granular `dim=-1` seams. It is deferred because `run_assembly` already serves every such cell at minimum bytes, and no traced caller reaches that predicate. It stays reachable: the split, the writer and `cb_output_tiles` are untouched by it.
