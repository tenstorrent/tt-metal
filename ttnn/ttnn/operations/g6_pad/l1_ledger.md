# L1 ledger: g6_pad

Schema: `.claude/references/l1-footprint-discipline.md`. Block axis: the folded work-tile axis `w`
(extent `block_tiles`); its constituents (slice, tile row, tile column) are folded and accounted
through `w`; the intra-tile 32×32 is always one whole page.

Inventory: regime E holds **one** CB (the fill bytes it writes from — it reads nothing, so there is
no data buffer at all). Regime N holds two: the slot the reader lands a tile in is the slot the writer
sends it from, and boundary pages are assembled in place in that slot, so the read → assemble →
store phase boundaries are not storage.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_fill_tile` (E) | `1` — constant, independent of every block axis | ≤3 segment ranges of one tile (≤ `1024e - 16e` bytes); the page is the allocation quantum | `{w: streams (every touched tile's writes source the same bytes); slice/tile row/tile col: folded into w; intra-tile: one page}` | x's dtype, raw (DM only, never through DEST; audit 2 n/a: no compute, no `fp32_dest_acc_en`) | writer | writer (self-owned scratch) | whole kernel | nothing to share with: the only CB of the regime. A smaller-than-a-page buffer would need a source offset different from the destination offset; one page keeps `src_off == dst_off` (alignment congruence for free) at ≤ 4 KB |
| `cb_work_tiles` (N) | `block_tiles * buffer_depth` | `block_tiles` (one block reserved by the reader or held by the writer); `2 * block_tiles` when `buffer_depth = 2` | `{w: spans → block_tiles; slice/tile row/tile col: folded into w; intra-tile: one tile per page}` | x's dtype, raw (DM only; audit 2 n/a) | reader | writer | whole kernel | cannot share with `cb_fill_tile`: concurrent lifetime (the fill page must stay intact while slots cycle every block) and different producer (ownership invariant). Capacity = live set; slots of fill-only tiles stay unused because slot `i` ↔ block tile `i` keeps the push/pop quantum nominal (ring-wrap invariant) — at most `block_tiles` idle pages, ≤ 48 KB in INPUTS |
| `cb_fill_tile` (N) | `1` — constant | 1 | `{w: streams (read once per fill-only tile, never rewritten); slice/tile row/tile col: folded; intra-tile: one page}` | x's dtype, raw | writer | writer (self-owned scratch) | whole kernel; allocated only when the call has fill-only tiles (`has_fill_only_tiles`) | cannot alias into `cb_work_tiles`: concurrent lifetime, and a slot would need a copy per fill-only tile; the reader masks with `fill_l1_range` directly and needs no fill page |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `tile_bytes` | 2048 (bf16), 4096 (fp32/int32/uint32) | `ttnn.tile_size(dtype)`, dtype in TARGET |
| `e` (`elem_bytes`) | 2 or 4 | dtype in TARGET |
| `block_tiles` (N) | `<= block_tiles_cap = floor((1347776 - tile_bytes) / (2 * tile_bytes))` = 328 (bf16) / 164 (32-bit) | host clamp; never binds in INPUTS (largest N run: p1, 1536 tiles over the grid → ≤ 14 per core at 110 cores) |
| `buffer_depth` (N) | ∈ {1, 2} | `2 iff ceil(max_core_work_tiles / block_tiles) > 1` |
| `has_fill_only_tiles` (N) | ∈ {0, 1} | any output tile with `!has_input` |

## Total per-core footprint

| Regime | Expression | Scales with | Bound |
|---|---|---|---|
| E | `tile_bytes` | nothing (constant) | 2048 / 4096 B for every shape |
| N | `tile_bytes * (block_tiles * buffer_depth + has_fill_only_tiles)` | `block_tiles` (block knob), `buffer_depth` (depth knob); fill term constant | `≤ tile_bytes * (2 * block_tiles_cap + 1) ≤ 1347776` B for every shape — BB-05 met by construction, no host refusal needed |

Phase 0 examples (110-core grid of the probed device, illustration): p2–p4 (E) 2048 B; case 247 (E) 2048 B; case 65 (N,
1 tile per core + fill) 4096 B; case 64 (N) 4096 B. Refinement p1 (N, fp32, 14 tiles per core +
fill) 61,440 B.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x, regime E | 0 — nothing is read | E writes `c` from L1; the logical rows of the last tile row stay in place, so no assembly needs them | none |
| out (= x's buffer), regime E | only rows `[in_row_rem, out_row_end)` of the last tile row of each slice, written once; every other tile row 0 | in place + write-only segments | none |
| x, regime N | each input tile read once (every input tile lies inside the output: padding is high-only) | read into the slot it is written from; no re-read | none |
| out, regime N | every output tile written once | copy/boundary pages from the slot, fill-only pages from `cb_fill_tile` (never read from DRAM) | none |

Totals (bf16 unless stated):

| Case | Read B | Written B | Read-mask-write (rejected A) would move |
|---|---|---|---|
| p2 258 | 0 | 3,724,800 | 7,946,240 |
| p3 253 | 0 | 993,280 | 7,946,240 |
| p4 291 | 0 | 2,979,840 | 7,946,240 |
| 247 | 0 | 86,016 | 196,608 |
| 65 (N) | 100,352 | 131,072 | — |
| 64 (N) | 57,344 | 65,536 | — |
| p1 266 (N, fp32, refinement) | 3,932,160 | 6,291,456 (= the minimum) | — |

Cross-core traffic: 0 B in every regime and case.

> Cheapest-traffic split considered: every split of the touched tile set moves identical bytes (no
> shared operand moves; `c` is made per core) — a tie, broken by occupancy: the folded work run over
> the full grid. Implemented: that split, which is therefore also the cheapest. Cheapest-traffic
> regime: E (extension write-only) — implemented; it is the minimum (zero reads, only the new rows
> written). No cheaper regime is deferred.
