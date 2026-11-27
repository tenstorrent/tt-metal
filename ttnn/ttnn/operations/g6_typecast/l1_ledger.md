# L1 Ledger: g6_typecast

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model: `tile_page` (knob `block_tiles`), `batch` / `tile_row` / `tile_col` (folded into `tile_page`), `tile_elements` (1 whole tile, fixed), `rm_row` / `rm_width` (ROW_MAJOR regimes, not built in Phase 0).

Built regime: `tile_convert`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `IN_DEPTH · block_tiles` (Phase 0: `2 · block_tiles`, `block_tiles ≤ 4`) | `IN_DEPTH · block_tiles`: block `k` held by compute for the whole `convert_block` (Upfront wait, AtEnd pop) while block `k+1` lands from the reader | `{tile_page: spans → block_tiles (× IN_DEPTH); batch: streams (folded into tile_page); tile_row: streams (folded); tile_col: streams (folded); tile_elements: spans → 1 tile per page (fixed); rm_row: n/a (TILE regime); rm_width: n/a (TILE regime)}` | input dtype. Phase 0: Float16_b (bf16 tile, 2048 B). Under `fp32_dest_acc_en=1` (→f32, →int32) this 16-bit page is the **input ABI** (the tensor arrives as bf16 from DRAM / L1), not a DEST-fed intermediate, so audit 2's "under" case does not apply: nothing accumulated is truncated | reader | compute | whole program | cannot share with `cb_output_tiles`: concurrent lifetime (depth-2 pipelining keeps blocks `k` and `k+1` of both CBs live) and a different page format/size (2048 B vs 4096 / 1088 B), so in-place transform is impossible. Capacity = live set; nothing to account for |
| `cb_output_tiles` | `OUT_DEPTH · block_tiles` (Phase 0: `2 · block_tiles`, `block_tiles ≤ 4`) | `OUT_DEPTH · block_tiles`: block `k` being written by the writer while compute packs block `k+1` (Upfront reserve, AtEnd push) | `{tile_page: spans → block_tiles (× OUT_DEPTH); batch: streams (folded); tile_row: streams (folded); tile_col: streams (folded); tile_elements: spans → 1 tile per page (fixed); rm_row: n/a; rm_width: n/a}` | output dtype: Float32 (4096 B) when `fp32_dest_acc_en=1` (→f32); Int32 (4096 B) when `fp32_dest_acc_en=1` (→int32); Bfp8_b (1088 B) when `fp32_dest_acc_en=0` (→bf8b). The Float32/Int32 pages are the **output ABI** and pair with 32-bit DEST, so audit 2's "over" case does not apply | compute | writer | whole program | cannot share with `cb_input_tiles` (same reason). No intermediate CB exists: the conversion packs straight into the output CB, which is the "pack into destination" pattern. Capacity = live set |

Deferred regimes (sizes stated so the refinement inherits the bound; not allocated in Phase 0):

| CB | Regime | Capacity (pages) | Note |
|----|--------|------------------|------|
| `cb_input_tiles` only | `tile_bitcopy` | `IN_DEPTH · block_tiles` | reader → writer directly; the compute kernel and `cb_output_tiles` are absent |
| `cb_input_slots`, `cb_output_slots` | `rm_convert` | `DEPTH · ceil(block_rows · S_in / tile_bytes_in)` and the output analogue | pages are tile-sized pseudo-tiles holding `tile_bytes_in / S_in` page slots; `block_rows` is bounded by `BLOCK_ROWS_CAP` (refinement host constant) |
| `cb_input_tiles` (RM page slots) | `rm_bitcopy` | `DEPTH · block_rows` pages of `aligned_page_size` | reader → writer |

## Symbol table

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `block_tiles` | `1 ≤ block_tiles = min(core_tiles_g1, BLOCK_TILES_CAP) ≤ 4` | host derivation from `BLOCK_TILES_CAP = 4` (single source). Measured: DRAM shapes flat over caps 1–32; P39 (L1) 17.4 µs at 4, 19.3 at 8, 36.6 at 16 — more remote reads in flight per core congest the mesh |
| `core_tiles_g1` | `ceil(tensor_pages / num_cores)`; unbounded in the op's contract, but it **never** sizes a CB (it is clamped through `block_tiles`) | `ttnn.split_work_to_cores` |
| `tensor_pages` | `input_tensor.buffer_num_pages()`; 1 … 8192 in Phase 0 INPUTS (P39) | the input tensor; it sizes no CB |
| `IN_DEPTH`, `OUT_DEPTH` | 2 | host constants |
| `in_page_bytes` | ∈ {2048 (bf16 / uint16 tile), 4096 (f32 / int32 / uint32 tile)} — Phase 0: 2048 | input dtype ∈ TARGET |
| `out_page_bytes` | ∈ {1088 (bf8b), 2048 (bf16), 4096 (f32 / int32 / uint32)} for TILE outputs — Phase 0: 4096 or 1088 | output dtype ∈ TARGET, TILE layout |
| `NB_L1` | number of L1 banks (≈ worker cores) | device allocator; used only for the op's own output allocation when `buffer == "l1"` |

## Per-core footprint

```
CB_bytes  = IN_DEPTH  · block_tiles · in_page_bytes        # scales with block_tiles and the input dtype
          + OUT_DEPTH · block_tiles · out_page_bytes       # scales with block_tiles and the output dtype
OUT_bytes = ceil(tensor_pages / NB_L1) · out_page_bytes    # only when buffer == "l1": the op's own output allocation
```

| Case | `block_tiles` | `CB_bytes` | `OUT_bytes` (L1 only) | Total vs. 1,347,776 B budget |
|------|---------------|------------|-----------------------|-------------------------------|
| Phase 0 worst CB (bf16 → f32 / int32) | 4 | 2·4·2048 + 2·4·4096 = 49,152 | — (DRAM) | 49,152 (3.6%) |
| P39 bf16 → bf8b, L1 (largest L1 output) | 4 | 2·4·2048 + 2·4·1088 = 25,088 | `ceil(8192/NB_L1)·1088` ≈ 81 KiB at `NB_L1 = 110` | ≈ 106 KiB (8%) |
| TARGET worst CB (32-bit in and out, refinement) | 4 | 2·4·4096 + 2·4·4096 = 65,536 | — | 65,536 (4.9%) |

No term scales with an op dimension: `tensor_pages` enters only the L1 output allocation, which is the tensor the op must return, divided by the bank count. A budget solve is unnecessary. The inventory is the minimum two CBs, and the block cap is set by the measured overlap lamp (mesh congestion on L1 placements), not by L1 pressure.

## Data-movement budget (chosen split: contiguous `tile_page` ranges, full grid)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 (DRAM placement) / 0 (L1 placement) | each page is read once, by exactly one core (disjoint contiguous ranges), and consumed from the CB; no second pass, nothing re-read | DRAM: none. L1 placement: each page crosses the NoC mesh once, from its bank's core to its worker core |
| output | 1 (DRAM placement) / 0 (L1 placement) | each page is packed once into `cb_output_tiles` and written once | DRAM: none. L1 placement: each page crosses the mesh once, from worker to bank core |

Totals per tier (examples):

| Case | DRAM bytes | NoC-mesh bytes (beyond the DRAM path) |
|------|------------|----------------------------------------|
| Q2185 `(16,4096,12)` bf16 → f32, DRAM | 4 MiB read + 8 MiB write = 12 MiB (12/32 of each tile row is logical; the padding is part of the TILE layout) | 0 |
| Q2187 `(1,1,3844,256)` bf16 → bf8b, DRAM | 968 · 2048 + 968 · 1088 = 2.9 MiB | 0 |
| P39 `(1,128,256,256)` bf16 → bf8b, L1 | 0 | 16 MiB + 8.5 MiB = 24.5 MiB |

> Cheapest-traffic split considered: `tile_convert_bank_local` (each core converts the pages of the L1 bank it hosts) — for L1 placements it removes all NoC-mesh crossings (P39: 24.5 MiB → 0, both tensors stay core-local). DRAM placements: every split moves the same bytes. Implemented: contiguous `tile_page` ranges over the full grid (`tile_convert`), which is also the cheapest split for every DRAM-placed case. Deferred because the built `tile_convert` already serves every L1 shape (P22, P31, P39, P46, P50) with the same kernels. The bank-local split changes only the page walk (`page_stride = NB_L1`, bank-owner cores), and Phase 0 keeps that walk parameterized (`page_stride` RT arg, value 1). One experiment settles it: P39 with `page_stride = NB_L1` against Phase 0 at equal `block_tiles`.
