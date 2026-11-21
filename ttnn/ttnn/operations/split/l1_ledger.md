# L1 ledger: split

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Design: `op_design.md` (same directory).
Target: `quasar` (4 MiB L1 per cluster). Thread plan (R, C, W) = (1, 0, 1): no compute kernel, so no
capacity is multiplied by a compute-thread count; ring capacities are per DM thread × 1.

Block axes of the Blocking Model: `outer`, `cut`, `width`, `within-page`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_output_pages` (both regimes) | `RING_DEPTH × reader_threads` = 2 × 1 = **2** entries of `page_bytes` | 1 entry on the producer side + 1 on the consumer side = 2 | `{outer: streams → 1 page, cut: streams → 1 page, width: streams → 1 page, within-page: spans → page_bytes}` | input dtype as stored (bf16; float32 later). Dataflow-only payload, never passes DEST — no DEST-width rule applies | reader | writer | whole program | Shares with nothing: it is the only ring. Capacity = live set (2) — the second entry is the double buffer that lets the reader fill while the writer drains (`RING_DEPTH`). Cannot alias `scratch_input_window`: both live at the same instant during `assemble_block` (read window, write entry). Not removable: there is no DRAM→DRAM NoC transfer, every page must land in L1 once |
| `scratch_input_window` (`tile_row_shift` only; a `ScratchpadSpec`, not a ring) | `WINDOW_TILES = 2 × BLOCK_H_TILES` = **16** tiles of `page_bytes` | up to `2 × block_h_tiles` = 16: a block of `block_h_tiles` output tiles may span several tile columns; a column segment of m output tiles reads ≤ m + 1 input tiles, so ≤ 2 × block_h_tiles per block (the tile shared with the previous block included) | `{outer: streams → (folded into the walk), cut: spans → ≤ block_h_tiles + 1 per column segment, width: spans → ≤ block_h_tiles columns (short columns), within-page: spans → page_bytes}` — capacity scales with `BLOCK_H_TILES`, the one knob both spanned axes draw on | input dtype as stored; scratchpad metadata omitted (DM-only) | reader | reader (private) | whole program | Not shareable with `cb_output_pages` (concurrent, see above). Not removable by **pack into destination**: the destination is the chunk tensor in DRAM/L1 banks, and assembling there means 32-B sub-page writes below the 64-B DRAM alignment (`dram_subtile_direct`, rejected). Not removable by **transform in place**: an output tile mixes rows of two input tiles, and a scratchpad cannot be drained by the writer (private to one kernel), so in place would make the reader also the writer and lose read/write overlap across harts. Capacity = live set |

## Symbol table

| Symbol | Meaning | Bound | Predicate / source establishing it |
|---|---|---|---|
| `page_bytes` | aligned page size of the input (= every chunk's) | ≤ 65,536 B; tile: 2,048 (bf16) / 4,096 (fp32); RM: `align_up(W*elem, alignment)`, ≤ 32,768 B in TARGET (W ≤ 8192: 16,384 B bf16, 32,768 B fp32) | host: `input.buffer_aligned_page_size()`; `NotImplementedError` if `> NOC_MAX_BURST_SIZE` (64 KiB, implicit sync = one transaction per entry); host asserts each chunk's `buffer_aligned_page_size()` equals it |
| `RING_DEPTH` | entries per DM thread in `cb_output_pages` | 2 (Phase 0 knob) | host constant |
| `BLOCK_H_TILES` | output tiles per DRAM-read barrier in `tile_row_shift` | 8 (Phase 0 knob); per block clamped to the tiles left in the column segment | host constant → CT arg `block_h_tiles` |
| `WINDOW_TILES` | window slots | `2 × BLOCK_H_TILES` (derived, never restated; the reader `static_assert`s `window_tiles >= 2 * block_h_tiles`) | host derivation |
| `reader_threads`, `writer_threads` | DM threads | 1, 1 (`THREAD_PLAN = (1, 0, 1)`) | host constant |
| per-core budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` | value on quasar not recorded in the profile (L1 = 4 MiB per cluster) | **not evaluated by the op**: the footprint below is bounded by constants and the 64 KiB page cap, so no budget predicate, safety fraction or blocking search exists |

## Total per-core footprint

- `page_remap`: `RING_DEPTH × page_bytes` → scales with `RING_DEPTH` and the page size only; ≤ 2 × 64 KiB = 128 KiB (cap); worked: `(1,4,8400)` RM DRAM = 2 × 16,832 = 33,664 B; TARGET max (RM fp32 W = 8192) = 65,536 B; tile bf16 = 4,096 B.
- `tile_row_shift`: `(RING_DEPTH + 2 × BLOCK_H_TILES) × page_bytes` → scales with `RING_DEPTH` and `BLOCK_H_TILES`; bf16 = 18 × 2,048 = 36,864 B; fp32 = 73,728 B.

No term scales with an op dimension (H, W, B, chunk count).

## Data-movement budget

Tier: "input tier" = DRAM for a DRAM tensor, the interleaved L1 banks for an L1 tensor (reached over the NoC either way). Core-local = the core's own L1.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input` (`page_remap`) | 1 | each input page is exactly one output page; read once into the ring | none |
| `input` (`tile_row_shift`) | 1 + ≤ (cores engaged − 1) tiles | `scratch_input_window` keeps the tile shared between consecutive output tiles of one column (cut inside a tile, chunk boundaries); only a range starting mid-column re-reads one tile (halo) | none |
| `chunk_0 … chunk_{n-1}` | 1 each | written once from the ring | none |
| assembled tiles (`tile_row_shift`) | 0 | core-local: each output tile's valid rows copied window → ring once (≤ 8 self-aimed copies) | none |

Totals per tier: input tier = `input_bytes` (+ ≤ 31 × `page_bytes` halo in `tile_row_shift`, ≤ 62 KiB bf16) read, `input_logical_pages × page_bytes` written; core-local = `output_bytes` (tile_row_shift only); cross-core = 0.

> Cheapest-traffic split considered: whole tile columns dealt to cores (`tile_row_shift`) — saves the ≤ (cores − 1)-tile halo (≤ 62 KiB bf16 at 32 cores; 13 tiles = 26 KiB on the traced `[1,4,400]`), adds nothing, but engages only `outer × tensor_w_tiles` cores (13 of 32 on `[1,4,400]`, 2 on `[1,32,50]`). Implemented: contiguous ranges of the column-major output-tile walk, which equals whole columns whenever ranges are column-multiples (`(2,6,512)`, `(1,34,4096)`). Deferred because the halo is bounded by the core count, not the tensor, and buys full occupancy on narrow tensors; the grid-synchronization perf lamp measures the two (`op_design.md` Perf lamps). For `page_remap` the implemented split is also the cheapest (each tensor crosses once).

## Implementation notes (Phase 0, ttnn-implementer)

Implemented as inventoried: one ring `cb_output_pages` (`RING_DEPTH × reader_threads` entries of
`input.buffer_aligned_page_size()`) in both regimes, and the `scratch_input_window` scratchpad
(`WINDOW_TILES × page_bytes`, `WINDOW_TILES = BLOCK_H_TILES + 1`) in `tile_row_shift` only. No CB added,
resized, merged or removed; the data-movement budget above holds as written (each input page read once in
`page_remap`; once per column segment plus ≤ 1 halo tile per core in `tile_row_shift`; each chunk page written
once). The self-aimed window → ring copies take the Scratchpad itself as the NoC source (its typed loopback
endpoint) rather than `dataflow_kernel_lib::local_addr(raw address)`: same transfer, bounds-checked.

## Verifier amendment (Phase 0 verification)

`scratch_input_window` grew from `BLOCK_H_TILES + 1` to `2 × BLOCK_H_TILES` slots. The Phase 0 reader clamped
every block to one tile column, so on the short-H shapes the op serves (the traced `[1,4,400]` / `[1,4,8400]`
TILE calls, every Phase 0 `tile_row_shift` entry) a block was 1–2 output tiles over **one** input tile: one DRAM
read per barrier, the `BLOCK_H_TILES` knob collapsed to 1 in effect. A block now runs over `BLOCK_H_TILES` output
tiles of the walk across column boundaries, and the window is indexed by read sequence number (slot = seq %
`WINDOW_TILES`). Structural counts (Python port of the reader's window logic, 32 cores): `(1,4,8400) s2` TILE
DRAM read barriers 270 → 71 (1.0 → 3.8 reads per barrier); `(1,34,4096) s17` 128 → 32 (2 → 8); `(2,40,8192) s20`
8 per barrier. Input reads unchanged (still minimum + halo). The data-movement budget above is unchanged.

## Refinement 1 (cut along W: `tile_col_shift`, `rm_substick`)

No CB added or removed. Both new regimes reuse the two buffers above with the shift-regime roles of
`tile_row_shift`:

- `cb_output_pages`: entry = chunk 0's aligned page (`chunk.buffer_aligned_page_size()`). Equal to the input page
  in every regime but `rm_substick`, where an entry is one output stick (`align(split_size · elem)`), shorter than
  the input stick. Capacity still `RING_DEPTH × reader_threads` = 2. Producer explicit in all three shift regimes;
  consumer implicit, except `rm_substick` with a shorter last-chunk page (explicit, sized writes; `SPLIT_SIZED_WRITES`).
- `scratch_input_window`: `WINDOW_TILES = 2 × BLOCK_H_TILES` slots of the **input** page (`in_page_bytes`), in every
  shift regime. For `rm_substick` a slot is one input stick: ≤ 32,768 B in TARGET (fp32, W = 8192), so the window is
  ≤ 16 × 32 KiB = 512 KiB. Shares with nothing, for the reasons in the table above (concurrent with the ring during
  `assemble_block`; the destination is in DRAM/L1 banks).

Total per-core footprint, shift regimes: `RING_DEPTH × entry_bytes + 2 × BLOCK_H_TILES × in_page_bytes`; worst case in
TARGET (`rm_substick`, fp32, W = 8192, one chunk) = 2 × 32 KiB + 512 KiB = 576 KiB of 4 MiB. Still no term scales
with an op dimension beyond the page size (capped at 64 KiB).

Data-movement budget, new rows: `input` (`tile_col_shift`, `rm_substick`) crosses its tier once + ≤ (cores − 1)
pages of halo (a range starting mid-line); each chunk once; assembly is core-local (RISC-V copies or one self-aimed
NoC copy per entry).
