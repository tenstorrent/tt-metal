# L1 ledger: g6_pad

Schema: `.claude/references/l1-footprint-discipline.md`. Block axis: the folded work-tile axis `w`
(extent `block_tiles`); its constituents (slice, tile row, tile column) are folded and accounted
through `w`. Inventory: two CBs; no intermediate (reads land in the slot the writer sends from,
boundary pages are assembled in place in that slot — the read→assemble phase boundary is not
storage).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_work_tiles` | `block_tiles * buffer_depth` | `block_tiles` (one block reserved by the reader or held by the writer); `2*block_tiles` when `buffer_depth = 2` | `{w: spans → block_tiles; slice/tile row/tile col: folded into w; intra-tile: whole tile per page}` | x's dtype, raw (DM only, never through DEST; audit 2 n/a: no fp32_dest_acc) | reader | writer | whole kernel | cannot share with `cb_fill_tile`: concurrent lifetime (the fill page must stay intact while slots are rewritten every block). Capacity = live set (depth 2 only when a core has > 1 block, i.e. when the second slot overlaps the next block's read) |
| `cb_fill_tile` | `1` (constant: one fill page, independent of every block axis) | 1 | `{w: streams (read once per fill-only tile, never rewritten); slice/tile row/tile col: folded; intra-tile: whole tile}` | x's dtype, raw | writer | writer | whole kernel; allocated only when `has_fill_only_tiles` | cannot alias into `cb_work_tiles`: concurrent lifetime and a slot would need a copy per fill-only tile; not allocated at all for extension calls (no fill-only tiles) |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `tile_bytes` | 2048 (bf16), 4096 (fp32/int32/uint32) | `ttnn.tile_size(dtype)`, dtype in TARGET |
| `block_tiles` | `<= block_tiles_cap = floor((1347776 - tile_bytes) / (2*tile_bytes))` = 328 (bf16) / 164 (32-bit) | host clamp; in INPUTS the clamp never binds (largest per-core assignment ≈ 1940 / num_cores) |
| `buffer_depth` | ∈ {1, 2} | `2 iff ceil(max_tiles_per_core / block_tiles) > 1` |
| `has_fill_only_tiles` | ∈ {0, 1} | any output tile with `!has_input`; always 0 in regime A |

## Total per-core footprint

`L1 = tile_bytes * (block_tiles * buffer_depth + has_fill_only_tiles)`

- scales with `block_tiles` (block knob) and `buffer_depth` (depth knob); the fill term is constant.
- bound: `≤ tile_bytes * (2 * block_tiles_cap + 1) ≤ 1347776` B for every shape (BB-05 met by
  construction; no host refusal needed).
- Phase 0 examples (grid 130, illustration): p2–p4 15 tiles → 30720 B; case 254 6 tiles → 12288 B;
  case 65 1 tile + fill → 4096 B.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x, regime A (extension) | last tile row of each slice read once (`S*Wt` tiles); other tile rows 0 | whole-tile RMW of the touched row (assembly needs the logical rows above `r0` in the written tile) | none |
| out, regime A | same `S*Wt` tiles written once (in x's buffer) | in place; whole-tile writes | none |
| x, regime B (new storage) | each input tile read once if it overlaps the output (always, padding is high-only) | read into the slot it is written from; no re-read | none |
| out, regime B | every output tile written once | copy/boundary from the slot, fill-only from `cb_fill_tile` (never read from DRAM) | none |

Totals, Phase 0 perf cases (bf16): p2/p3/p4: read 3,973,120 B, write 3,973,120 B. Case 247:
read 98,304 B, write 98,304 B (of 11.9 MB tensor). Case 65: read 100,352 B, write 131,072 B.
Refinement p1 (fp32, B): read 3,932,160 B, write 6,291,456 B (= the minimum). Cross-core: 0 B
in every case.

> Cheapest-traffic split considered: every split of the work-tile space moves identical bytes
> (no shared operand moves; `c` is made per core) — the split ranking is a tie and the folded
> run over the full grid is chosen on occupancy. Cheapest-traffic *regime*: C (extension
> write-only) — delta on p2: −3,973,120 B read, −248,320 B write (writes 3,724,800 B instead of
> 3,973,120 B); on p3: −3.97 MB read, −2.98 MB write. Implemented: A. Deferred because C's
> sub-tile writes start at face-row offsets that violate DRAM write alignment for odd `d % 32`
> on Blackhole and multiply transfers per tile 2–3×; it layers on A's work set and per-tile walk
> unchanged, so it stays reachable as a per-tile action switch behind its alignment predicate.
