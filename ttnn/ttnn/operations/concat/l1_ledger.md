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
| `tile_bytes` / `page_bytes` | TILE: 2,048 for bf16 (4,096 for 32-bit dtypes later; 1,024 for uint8). ROW_MAJOR (Refinement 1): `align_up(W_out · elem_bytes, l1_alignment)`, ≤ every input's and the output's `buffer_aligned_page_size()` (asserted); 16,384 B for `[1,32,4096]×2` along W | `page_geometry(out, layout)`: `out.buffer_page_size()` for TILE |
| `block_tiles` default (ROW_MAJOR) | `max(1, BLOCK_TILES · 1024 · elem_bytes // page_bytes)` sticks: the bytes of `BLOCK_TILES` tiles (128 sticks of 128 B; 1 stick of 16 KiB), then the same clamp as TILE | `default_block_pages` — derived from the single source `BLOCK_TILES` |
| `ring_cap_tiles` | `floor(RING_MAX_BYTES / (OUTPUT_DEPTH · tile_bytes))` = 255 (bf16) | `RING_MAX_BYTES = 1,048,560` (target page, cluster model; a named host constant, since Python has no query) |
| `budget_tiles` | `floor((ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES) / (OUTPUT_DEPTH · tile_bytes))` | run-time query. Its value on `quasar` was not probed (≤ 4,194,304 B per cluster) |
| `L1_RESERVE_BYTES` | 262,144 B (constant) | host constant in `concat_program_artifacts.py`: L1 left free beside the ring, because L1 input tensors share the cluster's L1 (verifier: row added, it was used in `budget_tiles` with no row) |
| `max_core_tiles` | `ceil(total_out_tiles / num_cores)` | `ttnn.split_work_to_cores` |
| `count` | `count ∈ {2, 3, 4, 9}` after Refinement 20 (`≤ MAX_INPUTS = 9`, TARGET's largest count; the reader binds `in_0 … in_8`). Sets no buffer size | `validate()`; host `MAX_INPUTS`, asserted equal to the reader's binding list |

**Total per-core footprint:** `OUTPUT_DEPTH · block_tiles · tile_bytes` = 32,768 B at Phase 0 on `quasar`. It scales linearly with `block_tiles` (knob `BLOCK_TILES`) and `OUTPUT_DEPTH`, and with `tile_bytes` (dtype). It is independent of every op dimension, of `count`, and of the regime. No budget solve is needed: the inventory is one buffer, and at the Phase 0 knobs its footprint is under 1% of L1.

## Data-movement budget

Chosen split: contiguous ranges of flat output tiles over the full grid (`split_work_to_cores(..., row_wise=True)`).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| each input `in_k` (`page_copy`) | 1: every page read once, whole | each output page is exactly one input page, and the split assigns each output page to one core | none |
| each input `in_k` (`row_major_copy`, `row_major_runs`) | 1: every stick read once (`row_major_runs`: exactly the `ext_k · elem_bytes` logical bytes; `row_major_copy`: the stick rounded up to 16 B, within its aligned page) | each output stick is one input stick (copy) or one run of each piece's same-index stick (runs) | none |
| each input `in_k` (`run_assembly`) | ≤ 1: only bytes that land in the output are read, each once. Input padding along `dim` is never read | runs are NoC-read straight into the destination entry, and a source page straddling two output tiles is read in two disjoint parts (possibly by two cores) | none |
| `out` | 1: every page written once, whole | one writer per page | none |

**Totals per tier.**

- DRAM read is `Σ_k pages(in_k) · tile_bytes` in `page_copy`, and `≤ total_out_tiles · tile_bytes` in `run_assembly`.
- DRAM write is `total_out_tiles · tile_bytes`.
- Cross-core traffic is 0.
- Core-local L1 traffic is the ring only: each byte is written once by the NoC and read once by the NoC.
- Placement (Refinement 10): an input in L1 — interleaved, or height-sharded with a shard grid of its own — is read
  through the same accessor, one NoC crossing per byte from the cluster that holds the page to the cluster that
  assembles the output page, exactly as a DRAM page is. The counts above hold for every placement; for an L1 input
  the "DRAM crossing" of its row is an L1-to-L1 NoC crossing instead. A sharded input is not consumed in place: the
  output split (contiguous output pages over the whole grid) does not follow any input's shard grid, and a `dim=-1`
  output page draws from every input's shard. Reading each shard on its own cluster is a perf refinement of its
  own (op_requirements.md, Refinement 10, "Keep the data path"). The ring is unchanged, no CB is added.

Transactions: 1 read and 1 write per output tile in `page_copy`; 1 read and 1 write per output stick in `row_major_copy`; `count` reads and 1 write per output stick in `row_major_runs` (one barrier per block of sticks). In `run_assembly` reads are 2–6 per tile (row-granular `dim=-2`), ≤ 8 per tile (face-granular `dim=-1`) or ~64–160 per tile (element-granular `dim=-1`), and writes stay 1 per tile.

> Cheapest-traffic split considered: contiguous flat output-tile ranges with direct sub-page reads, which moves exactly the minimum bytes per tier (each input byte that lands in the output crosses DRAM once, each output page once, no cross-core traffic). Implemented: that same split. It is also the cheapest-traffic split. The only alternative that differs is `staged_shift` (deferred). It moves **more** DRAM bytes: whole source pages plus a halo page per piece per segment, versus `run_assembly`'s exact runs. It trades those bytes for fewer, larger transactions on element-granular `dim=-1` seams. It is deferred because `run_assembly` already serves every such cell at minimum bytes, and no traced caller reaches that predicate. It stays reachable: the split, the writer and `cb_output_tiles` are untouched by it.

## Implementation notes (ttnn-implementer, phase 0)

- Inventory as planned: one ring, `cb_output_tiles`, `OUTPUT_DEPTH · block_tiles` entries. No buffer added, resized or removed.
- `run_assembly` reads **less** than the budget above: cross-axis tile padding is not read either (the reader limits rows,
  for `dim=-1`, or column halves, for `dim=-2`, to the valid extent `cross_extent - 32·x`). The design allowed reading it
  into output padding; skipping it removes those transactions. DRAM read in `run_assembly` is exactly the bytes that land
  in the output's data, and `≤ total_out_tiles · tile_bytes` still bounds it.

## Implementation notes (ttnn-implementer, Refinement 1 — ROW_MAJOR)

- No buffer added or removed: the same ring `cb_output_tiles`, its entry now one output **page** (tile or 16-B-aligned
  stick, `page_geometry`). Capacity `OUTPUT_DEPTH · block_tiles` entries with `block_tiles` counted in sticks; at the
  default it holds `2 · BLOCK_TILES` tiles' worth of bytes (32,768 B for bf16) whatever the stick width, and a stick wider
  than `BLOCK_TILES` tiles' bytes gives a block of one stick (e.g. 2 · 16,384 B for `[1,32,4096]×2` along W).

## Implementation notes (ttnn-implementer, Refinement 20 — `memory_config`, four and nine tensors)

- No buffer added, resized or removed. `count` 4 and 9 set no buffer size (runtime array args of `count` words only).
- `memory_config=ttnn.L1_MEMORY_CONFIG` places the **output** L1 interleaved; the ring is unchanged. The output pages now
  share each cluster's L1 with the ring and with any L1 input, which is what `L1_RESERVE_BYTES` (256 KiB) was already
  holding back from `budget_tiles`.
- Bounds on the largest join in INPUTS, Qwen's logits `4 × [1,1,32,37984]` → `[1,1,32,151936]` bf16 (9,723,904 B out):
  - TILE: `page_copy`, 4,748 output tiles over 32 clusters (149 per cluster at most); ring 2 · 8 · 2,048 = 32,768 B.
    L1 output: ≤ 149 tiles = 305,152 B per bank. L1 interleaved inputs: the same again.
  - ROW_MAJOR: `row_major_runs`, entry = one output stick = 151,936 · 2 = 303,872 B (16-B aligned); default block
    `max(1, 8 · 1024 · 2 // 303,872)` = 1 stick; ring 2 · 303,872 = 607,744 B, under `RING_MAX_BYTES` (1,048,560) and
    under `budget_tiles` (5 sticks at the unreserved size the allocator reports, 3,860,928 B). 32 sticks over 32
    clusters, one stick each. L1 output: one stick per bank (303,872 B); L1 interleaved inputs: one 75,968 B stick
    per tensor per bank. Worst cluster (RM, L1 inputs, L1 output): 607,744 + 303,872 + 4 · 75,968 ≈ 1.2 MB of 3.86 MB.
  - Height-sharded: the runner gives a TILE Qwen tensor one core (one tile row, 2,430,976 B per shard) and skips the
    cell (four such shards on one core do not fit); the ROW_MAJOR tensors spread one stick per core and pass.
  - The ROW_MAJOR ring has one hard bound the clamp cannot lower: one entry per output stick, two entries, so an output
    stick wider than `RING_MAX_BYTES / OUTPUT_DEPTH` = 524,280 B (262,140 bf16 columns) would exceed the ring cap at
    block 1. No INPUTS entry comes near it (Qwen: 303,872 B); a wider join would need a stick split into sub-entries.
