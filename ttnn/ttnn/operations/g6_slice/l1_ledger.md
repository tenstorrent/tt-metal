# L1 Ledger: g6_slice (Phase 0)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes and knobs: `op_design.md` → *Blocking Model*. Budget: 1,347,776 B per core (BB-05), shared with any L1-resident share of out (none in Phase 0: out is in DRAM).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_src_stage` | `stage_depth_blocks · block_tiles · src_slots` | `stage_depth_blocks · block_tiles · src_slots` — at peak the writer still holds block k (waiting on its flush) while the reader has block k+1 in flight; capacity = live set | `{lead0: streams, lead1: streams, tile_row: streams, tile_col: streams, out_page: spans → block_tiles, src_slot: spans → src_slots}` | x's storage format, page = `x.buffer_page_size()` (bfloat16 tile, 2048 B). Dataflow-only payload: never unpacked, never through DEST, so audit 2 does not apply (no `fp32_dest_acc_en` — there is no compute kernel) | reader | writer | whole program (every block) | No other buffer exists to share with. Considered and taken: **pack into destination** — in R1 the staged source page is the output page, so there is no separate output CB; **no intermediate** — R2 writes the selected row runs straight from the staged source page to out in DRAM (L1→DRAM writes only need offsets equal mod 16, which 32-B face rows satisfy), so there is no assembly buffer; **transform in place** — R3 assembles each output page in slot 0 of its own staged source group in raster order (the source position is always later in raster order than the destination, so nothing is overwritten before it is read) and writes slot 0, so there is no scratch tile. The depth-2 gap over one block is the read/write overlap (double buffering), not idle space |

## Symbol table

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `page_bytes` | 2048 (bfloat16, Phase 0); ≤ 4096 in TARGET (float32 / int32 / uint32) | SUPPORTED `dtype`; `x.buffer_page_size()` |
| `src_slots` | R1: 1; R2: 1 or 2; R3: 2 or 4 | `select_regime` and the slot predicates in `op_design.md` (*Axes*, `src_slot` row) |
| `block_tiles` | `1 ≤ block_tiles ≤ max(1, min(core_pages_max, BLOCK_BYTES_CAP // (page_bytes · src_slots)))` | host derivation from `BLOCK_BYTES_CAP = 64 KiB` and the work split |
| `stage_depth_blocks` | 2 | host constant (single source) |
| `core_pages_max` | `ceil(tensor_out_pages / num_cores)` | `ttnn.split_work_to_cores` |

No whole-op dimension appears in any capacity: `tensor_out_pages`, `M`, `N` enter only through `core_pages_max`, which `block_tiles` is clamped *below*, never sized *to*.

## Total per-core footprint

`L1_op = stage_depth_blocks · block_tiles · src_slots · page_bytes`

- scales with `block_tiles` (knob `BLOCK_BYTES_CAP`) and `stage_depth_blocks`; `src_slots` is fixed by the regime and `block_tiles` shrinks to compensate, so `block_tiles · src_slots · page_bytes ≤ max(BLOCK_BYTES_CAP, src_slots · page_bytes)`.
- Phase 0 bound: `2 · 64 KiB = 131,072 B` (R1, R2 and R3 alike, bfloat16) ≤ 1,347,776 B. Every golden INPUTS shape fits.
- Refinements add: `cb_begin_scratch` (one aligned starts page, ≤ 64 B, `R1-dev`); for `l1_interleaved` out, the core's share `ceil(out_pages / num_l1_banks) · page_bytes`; for `l1_width_sharded`, the output shard itself (backed in place; x's shard is the model's and is not counted).

## Data-movement budget

Chosen split: contiguous `out_page` ranges over the full grid (output-owner). Phase 0 has no cross-core traffic.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (R1) | each page holding a selected element: once. Unselected pages: never | output-owner split; each output page has exactly one source page | 0 |
| x (R2) | each selected face-row band: once (64 B envelope on Blackhole) | bands of neighbouring output tile rows are disjoint; only the selected rows are read | 0 |
| x (R3) | each needed source page: ≤ 2 (≤ 4 with a row shift) | neighbouring output tiles share a source column; no residency across pages (deferred `R3_reuse`; no production case) | 0 |
| out | each page once (R1, R3 whole pages); R2 only the selected rows | writer owns its contiguous output range | 0 |
| starts tensor (refinement) | one aligned page per core | every core needs the same begin word (deferred `R1-dev-bcast`) | 0 |

Totals, by case:

| Case | DRAM read | DRAM written | Cross-core |
|------|-----------|--------------|------------|
| p1 (703): `(16, 4096, 8704)` cols 4352..8704, R1, 278,528 pages | 570,425,344 B | 570,425,344 B | 0 |
| 587: `(16, 4096, 1280)` row 4095, R2, 640 pages | 81,920 B | 40,960 B | 0 |
| 1183: `(1, 1, 16, 62080)` row 7, R2, 1940 pages | 248,320 B | 124,160 B | 0 |

> Cheapest-traffic split considered: output-owner (`out_page` ranges) — the minimum for R1 and R2 at DRAM read granularity, 0 B cross-core. Implemented: output-owner. It is the cheapest split; the only traffic above the minimum is R3's source-column re-read (≤ 2×/4×), which serves no production case and is the deferred `R3_reuse` row.
