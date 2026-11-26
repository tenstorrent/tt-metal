# L1 Ledger: g6_typecast (Phase 0, TILE stream regime)

Schema: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`): the
**page axis** (extent `block_tiles`; leading dims, tile rows and tile cols are folded into it) and
the **intra-tile** axis (one whole tile, atomic).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `in_depth_blocks · block_tiles` (Phase 0: `2 · min(core_pages_max, 8)`) | `2 · block_tiles` pages: one block being converted + one being filled | `{page: spans → block_tiles ·in_depth_blocks, intra_tile: spans → 1 tile (page = whole tile)}` | input dtype (Phase 0 Float16_b, 2048 B) — a dataflow payload read from the tensor, its width is the tensor's | reader | compute | whole kernel, streaming | none — concurrent lifetime with `cb_output_tiles` (both live for the whole kernel) and differing page format/size; capacity − live set = 0 (depth 2 is the overlap mechanism) |
| `cb_output_tiles` | `out_depth_blocks · block_tiles` (Phase 0: `2 · min(core_pages_max, 8)`) | `2 · block_tiles` pages: one block being packed + one being written | `{page: spans → block_tiles ·out_depth_blocks, intra_tile: spans → 1 tile}` | output dtype: Float32 (4096 B), Bfp8_b (1088 B) or Int32 (4096 B) — the output ABI format; Float32/Int32 match `fp32_dest_acc_en = 1` for those pairs; Bfp8_b with 16-bit DEST | compute | writer | whole kernel, streaming | none — concurrent lifetime with `cb_input_tiles`; it is already the destination (no intermediate exists to fold into it) |

No intermediate CB: the conversion packs directly into the destination CB (Rule 3 pattern 1); an
in-place transform is impossible because input and output page formats differ.

### Symbol table

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `block_tiles` | `1 ≤ block_tiles ≤ 8` (Phase 0 `min(core_pages_max, BLOCK_TILES_DEFAULT=8)`); L1 bound ≈ `floor(1 347 776 / (2·(4096+4096))) = 82` at TARGET's widest pair | host constant `BLOCK_TILES_DEFAULT` |
| `in_depth_blocks`, `out_depth_blocks` | 2 | host constants |
| `in_page_bytes` | ≤ 4096 (tile of a 32-bit dtype) | TILE layout, TARGET dtypes |
| `out_page_bytes` | ≤ 4096 | TILE layout, TARGET dtypes |
| `core_pages` | unbounded (up to 8192/grid); **does not appear in any capacity** | — |

### Total per-core footprint

`L1_op = block_tiles · (in_depth_blocks · in_page_bytes + out_depth_blocks · out_page_bytes)`

- Phase 0 worst (bf16 → float32/int32): `8 · (2·2048 + 2·4096) = 98 304 B`; bf16 → bf8b: `8 · (2·2048 + 2·1088) = 50 176 B`.
- TARGET worst (32-bit → 32-bit): `8 · (2·4096 + 2·4096) = 131 072 B`.
- Every term scales linearly with `block_tiles` and with its CB's depth knob; nothing scales with the tensor size. Bound 1 347 776 B (BB-05) holds for every golden shape.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (DRAM placement) | 1 | every page is read exactly once by exactly one core; no re-read, no staging | none |
| `output_tensor` (DRAM placement) | 1 | every page written once | none |
| `input_tensor` (L1 placement) | 0 | lives in L1 banks | `(1 − 1/num_l1_banks)` of its bytes cross the NoC (pages are round-robin over all L1 banks; contiguous ranges do not align with banks) |
| `output_tensor` (L1 placement) | 0 | lives in L1 banks | same fraction |

Totals for the flagged Phase 0 cases: p3 Q2185 `[16,4096,12]` bf16→f32 DRAM: 4 194 304 B read + 8 388 608 B written from/to DRAM, 0 cross-core. p4 P39 `[1,128,256,256]` bf16→bf8b L1: 0 DRAM; ≈ 16 777 216 + 8 912 896 B over the NoC (× `(1 − 1/nb)` cross-core).

> Cheapest-traffic split considered: page axis, **bank-affine** strided sets (`page = bank_id + j·nb`) — saves all cross-core NoC traffic for L1 placements (P39: ≈ 25.7 MB × (1 − 1/nb)), identical for DRAM placements. Implemented: page axis, contiguous ranges (the cheapest split for every DRAM placement). Deferred because the contiguous split is already correct for every L1 shape and the bank-affine one needs the host L1 bank → logical core map verified per arch/harvest; the reader/writer carry `page_stride` RT args from Phase 0, so turning it on is a knob-turn (regime row "TILE bank-affine").
