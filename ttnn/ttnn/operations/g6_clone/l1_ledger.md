# L1 Ledger: g6_clone

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model.
Named block axes: `page` (extent knob `block_pages`) and `in_page` (extent `aligned_page_bytes`, fixed by the buffer). The axes `lead`, `h_tiles`, `w_tiles` and `rows` fold into `page`.

## Inventory before solve

| Question | Answer |
|----------|--------|
| Minimum buffer count | 1. A Tensix core cannot move DRAM → DRAM: a NoC read lands in local L1 and a NoC write sources from local L1. So one staging buffer is irreducible. |
| Pack into the destination? | Not applicable. There is no compute phase, and the staging CB *is* the source of the output write. |
| Transform in place? | Not applicable: no transform. |
| Alias disjoint lifetimes? | Nothing to alias: only one CB exists. |
| Fold into DEST? | Forbidden: DEST is an engine path (spec rule 1). |
| Budget solve introduced? | Only the clamp `block_pages_l1_cap`. It bounds the one irreducible buffer against the model's L1 bound for unbounded tensor sizes, and it never binds on INPUTS. It is a cap, not a search. |

## Ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_page_staging` | `cb_depth_blocks · block_pages` pages of `aligned_page_bytes`, where `block_pages = min(core_pages_max, floor(L1_BUDGET_BYTES / (CB_DEPTH_KNOB · aligned_page_bytes)))` and `cb_depth_blocks = min(CB_DEPTH_KNOB, ceil(core_pages_max / block_pages))` | `cb_depth_blocks · block_pages` pages. With ≥2 blocks per core, block k (being written) and block k+1 (being read) are live at once. With one block per core, `block_pages` pages are live. Capacity equals the live set in both cases. | `{page: spans → block_pages; in_page: spans → aligned_page_bytes (whole page); block walk (block_idx): streams → cb_depth_blocks slots; lead/h_tiles/w_tiles/rows: folded into page, streamed with it}` | `x.dtype` (Float32 in Phase 0). This is a dataflow-only payload that never passes through DEST, and `fp32_dest_acc_en` does not exist (no compute kernel). That is audit 2's "dataflow-only payload" reason. | reader | writer | whole program (both kernels) | none: it is the only CB in the program, so there is no other buffer to share an allocation with |

### Symbol table

| Symbol | Definition | Bound | Predicate that establishes the bound |
|--------|------------|-------|--------------------------------------|
| `aligned_page_bytes` | `x.buffer_aligned_page_size()` | TILE: 4096 (f32), 2048 (bf16). ROW_MAJOR: `align_up(W·elem, 64)`, at most 5120 on INPUTS. | Host guard `block_pages_l1_cap ≥ 1`, i.e. `aligned_page_bytes ≤ 673888`. Otherwise RuntimeError (BB-05); no TARGET shape reaches it. |
| `tensor_pages` | `x.buffer_num_pages()` | 1–3072 on INPUTS; unbounded by the contract | Enters capacity only through `core_pages_max`, which `block_pages` clamps |
| `G` | `grid.x · grid.y` from `device.compute_with_storage_grid_size()` | 110 on this Blackhole box family | device query |
| `num_cores` | `min(tensor_pages, G)` | ≤ G | — |
| `core_pages_max` | `ceil(tensor_pages / num_cores)` | ≤ 28 on INPUTS at G = 110; unbounded otherwise | clamped by `block_pages_l1_cap` in `block_pages` |
| `block_pages_l1_cap` | `floor(L1_BUDGET_BYTES / (CB_DEPTH_KNOB · aligned_page_bytes))` | 164 (f32 tile), 329 (bf16 tile), 131 (5120 B row), 10529 (64 B row) | — |
| `CB_DEPTH_KNOB` | host constant | 2 | — |
| `L1_BUDGET_BYTES` | host constant | 1347776 | model environment, BB-05 |

### Total per-core footprint

```
L1(core) = cb_depth_blocks · block_pages · aligned_page_bytes  ≤  L1_BUDGET_BYTES
```

- `block_pages` scales with the core's assignment (`tensor_pages / G`) up to the L1 cap.
- `cb_depth_blocks` is 1 until the cap splits a core's assignment, then 2.
- `aligned_page_bytes` is fixed by dtype and layout.
- No term scales with the tensor size once the cap binds.

| Case | `cb_depth_blocks · block_pages · aligned_page_bytes` | Bytes per core | Fraction of 1347776 |
|------|------------------------------------------------------|----------------|---------------------|
| p1 `(16,12,128,128)` f32 | 1 · 28 · 4096 | 114688 | 0.085 |
| p2 `(8,12,128,128)` f32 | 1 · 14 · 4096 | 57344 | 0.043 |
| `(1,12,128,128)` f32 | 1 · 2 · 4096 | 8192 | 0.006 |
| `(1,1)` f32 | 1 · 1 · 4096 | 4096 | 0.003 |
| p3 `(1,512,5120)` bf16 | 1 · 24 · 2048 | 49152 | 0.036 |
| p4 `(1,1,32,62080)` bf16 | 1 · 18 · 2048 | 36864 | 0.027 |
| `(16,3,2560)` bf16 RM | 1 · 1 · 5120 | 5120 | 0.004 |
| `(1,)` int32 RM | 1 · 1 · 64 | 64 | < 0.001 |

### Audits

| Audit | Result |
|-------|--------|
| 1. Capacity vs live set | Capacity = live set (`cb_depth_blocks` is derived, so no idle second slot on one-block cores). Not under-sized either: the live set spans `page` and `in_page`, and capacity scales with both. |
| 2. Page format vs DEST width | No DEST is involved. `x.dtype` is carried as raw bytes. |
| 3. Disjoint lifetimes | There is only one buffer. |
| 4. Bounds and closed form | Every symbol is bounded above, and the total is closed-form. |

## Data-movement budget (chosen split: `flat_page_range`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` | 1 (read) | Each page is read by exactly one core into its staging slot. Nothing is re-read: no page is needed twice, so there is nothing to keep resident across passes. | 0 |
| `out` | 1 (write) | Each page is written once from the slot it was read into | 0 |

| Phase 0 case | DRAM bytes (x read + out write) | Cross-core NoC bytes | Core-local L1 bytes (NoC write into CB + NoC read out of CB) |
|--------------|---------------------------------|----------------------|----------------------------------------------------------------|
| p1 `(16,12,128,128)` f32 | 12582912 + 12582912 = 25165824 | 0 | 12582912 + 12582912 |
| p2 `(8,12,128,128)` f32 | 6291456 + 6291456 = 12582912 | 0 | 6291456 + 6291456 |
| `(1,12,128,128)` f32 | 786432 + 786432 = 1572864 | 0 | 786432 + 786432 |
| `(1,1)` f32 | 4096 + 4096 = 8192 | 0 | 4096 + 4096 |

**Totals per tier** (all INPUTS, stored bytes, each case independently): DRAM = 2 × stored bytes of `x`, cross-core = 0, L1 = 2 × stored bytes of `x`. There are no fan-out and no multicast terms.

> Cheapest-traffic split considered: `logical_rows_partial_tile` (move only the logical rows of partial tiles) — DRAM −7697920 B on `[1,1,1,62080]` bf16 (7946240 → 248320 B, read + write), −634880 B on `[1,5120]` bf16 (655360 → 20480 B), −8064 B on `(1,1)` f32 (8192 → 128 B); cross-core unchanged at 0; transactions per page 1 → 2. Implemented: `flat_page_range`. Deferred because Phase 0 (float32) holds only `(1,1)` among partial-tile shapes, a one-page call bounded by the program's fixed cost; the two shapes it pays on are bfloat16, outside Phase 0 SUPPORTED. The structure keeps it reachable: same split, same CB page, same block loop; only the per-page transfer inside `load_block` / `store_block` changes. On every tile-aligned shape (p1, p2, `(1,12,128,128)` in Phase 0) the implemented split **is** the cheapest-traffic split.
