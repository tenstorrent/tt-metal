# L1 Ledger: permute (Phase 0 — `whole_tile_relocation`)

Schema owner: `.claude/references/l1-footprint-discipline.md`.

Named block axes (from `op_design.md` §Axes): `n`, `c`, `ht`, `wt`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_tiles` | `BUFFER_DEPTH * BLOCK_TILES` = `2 * 8` = 16 pages of `tile_size(fp32)` = 4096 B → 65536 B | `BLOCK_TILES` = 8 pages (32768 B) — one block is simultaneously resident; the second depth slot is a *different* block in flight, not part of one block's live set | `{n: streams → 1 plane at a time (block_n = 1); c: streams → 1 plane at a time (block_c = 1); ht: spans → jointly with wt as one linear run of BLOCK_TILES tiles; wt: spans → jointly with ht, same run}` | `Float32` — the tensor dtype. No DEST, no compute kernel, so `fp32_dest_acc_en` is not applicable; the page format must equal the tensor format exactly or the copy is not bit-preserving | reader | writer | whole program (single phase) | **Nothing to share with — it is the only CB.** Capacity exceeds the live set by exactly the depth factor `BUFFER_DEPTH = 2`; that gap is the deliberate double-buffering mechanism (reader fetches block `i+1` while writer drains block `i`), not slack. |

**Inventory justification (Rule 1 / Rule 3).** One buffer. The read→write phase boundary *is* this
CB, so "pack into the destination" and "transform in place" are vacuous (no second phase, no format
change), aliasing has nothing to alias against, and DEST folding does not apply (no accumulation).
No L1 budget predicate, safety fraction, or blocking search is introduced.

## Symbol table

| Symbol | Kind | Bound | Predicate establishing the bound |
|--------|------|-------|----------------------------------|
| `BLOCK_TILES` | block extent knob (joint `ht`x`wt` run) | `1 <= BLOCK_TILES <= tiles_per_plane`, and `BLOCK_TILES <= l1_budget / (BUFFER_DEPTH * tile_bytes)` | Mechanism caps in `op_design.md`: plane contiguity + L1 residency; runtime extent additionally clamped by `tiles_this_core` |
| `BUFFER_DEPTH` | depth knob | `2 <= BUFFER_DEPTH <= 4` (lamp range) | double-buffering requires >= 2; upper end is the overlap perf lamp, bounded by the same L1 expression |
| `tile_bytes` | constant per dtype | `ttnn.tile_size(dtype)`: 4096 (fp32), 2048 (bf16), 1088 (bf8b) | dtype is a SUPPORTED axis value (Phase 0: fp32 only) |
| `tiles_per_plane` | derived tensor quantity | `ceil_div(H,32) * ceil_div(W,32)`; used only as a **clamp**, never as a capacity | appears in no capacity expression, so it needs no residency bound |
| `tiles_this_core` | derived per-core quantity | `<= ceil_div(tensor_tiles, num_cores)` | `ttnn.split_work_to_cores` contract; used only as a clamp |

No whole-op dimension appears in any capacity expression, so no regime predicate is needed to
bound one.

## Total per-core footprint

```
L1_per_core = BUFFER_DEPTH * BLOCK_TILES * tile_bytes
            = 2 * 8 * 4096 = 65536 B  (64 KB, ~4% of 1.5 MB L1)
```

Scaling: linear in `BLOCK_TILES` (extent knob), linear in `BUFFER_DEPTH` (depth knob), linear in
`tile_bytes` (dtype, R4). **Independent of every tensor dimension** — no term grows with `N`, `C`,
`H` or `W`. Headroom at Phase 0 values permits `BLOCK_TILES` up to 32 at depth 4 (512 KB) without a
solve, which is why the extent and overlap lamps can be swept without touching the inventory.

# Data-movement budget — chosen split: linear output-tile range, `row_wise=True`

Let `B = tensor_tiles * tile_bytes` (total tensor bytes; for `(2,4,512,512)` fp32:
`2*4*16*16 = 2048` tiles x 4096 B = 8 MB).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input` | 1 (`B` bytes read) | every output tile reads exactly one input tile, and each core's range is disjoint — no tile is read by two cores and none is re-read | none |
| `output` | 1 (`B` bytes written) | each output tile is written exactly once by its owning core; no read-modify-write, no partials | none |

Totals per tier: **DRAM `2B`** (8 MB read + 8 MB written = 16 MB for `(2,4,512,512)` fp32);
**cross-core NoC 0 B**; **core-local L1** `2B` (one CB write by the reader, one CB read by the
writer) — unavoidable for a DRAM→DRAM move. Transaction count: `tensor_tiles` reads +
`tensor_tiles` writes, batched `BLOCK_TILES` per barrier, all whole-tile pages (no sub-tile faces).

> Cheapest-traffic split considered: **linear output-tile range (the one implemented)** — it is the
> minimum for an interleaved-DRAM→interleaved-DRAM move (`2B`, delta 0 vs. minimum), and every
> alternative interleaved split moves the same bytes with equal or worse transaction shape
> (plane-per-core strands cores; `wt`-only loses run contiguity). Implemented: linear output-tile
> range. Nothing deferred on interleaved-traffic grounds.
>
> The one strictly cheaper **scheme** is the `sharded_output` regime: writing into an L1-sharded
> output removes the output crossing entirely — **DRAM `2B` → `B`, cross-core 0 B unchanged** (a
> `B`-byte saving, i.e. up to 2x on a DRAM-bandwidth-bound op). Deferred (R2) because Phase 0's
> SUPPORTED rectangle is `dram_interleaved` and the built regime serves every interleaved shape in
> `INPUTS`; not foreclosed — the writer's only coupling to placement is its output CB/accessor, so
> the swap to `ttnn.cb_descriptor_from_sharded_tensor(output)` is a placement change, not a new
> algorithm.
