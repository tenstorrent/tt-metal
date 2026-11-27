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
| `cb_page_staging` | `cb_depth_blocks · block_pages` pages of `aligned_page_bytes`, where `block_pages = min(ceil(core_pages_max / BLOCKS_PER_CORE_KNOB), floor(L1_BUDGET_BYTES / (CB_DEPTH_KNOB · aligned_page_bytes)))` and `cb_depth_blocks = min(CB_DEPTH_KNOB, ceil(core_pages_max / block_pages))` | `cb_depth_blocks · block_pages` pages. With ≥2 blocks per core, block k (being written) and block k+1 (being read) are live at once. With one block per core, `block_pages` pages are live. Capacity equals the live set in both cases. | `{page: spans → block_pages; in_page: spans → aligned_page_bytes (whole page); block walk (block_idx): streams → cb_depth_blocks slots; lead/h_tiles/w_tiles/rows: folded into page, streamed with it}` | `x.dtype` (float32, bfloat16, int32 or uint32; labels the descriptor only). This is a dataflow-only payload that never passes through DEST, and `fp32_dest_acc_en` does not exist (no compute kernel). That is audit 2's "dataflow-only payload" reason. | reader | writer | whole program (both kernels) | none: it is the only CB in the program, so there is no other buffer to share an allocation with |

### Symbol table

| Symbol | Definition | Bound | Predicate that establishes the bound |
|--------|------------|-------|--------------------------------------|
| `aligned_page_bytes` | `x.buffer_aligned_page_size()` | TILE: 4096 (f32), 2048 (bf16). ROW_MAJOR: `align_up(W·elem, 64)`, at most 5120 on INPUTS. | Host guard `block_pages_l1_cap ≥ 1`, i.e. `aligned_page_bytes ≤ 673888`. Otherwise RuntimeError (BB-05); no TARGET shape reaches it. |
| `tensor_pages` | `x.buffer_num_pages()` | 1–3072 on INPUTS; unbounded by the contract | Enters capacity only through `core_pages_max`, which `block_pages` clamps |
| `G` | `grid.x · grid.y` from `device.compute_with_storage_grid_size()` | 110 on this Blackhole box family | device query |
| `num_cores` | `flat_page_range` / `logical_rows_partial_tile`: `min(tensor_pages, G)`. `bank_affine_runs` (`dram_rows`): the cores on the banks' anchor rows between each DRAM column and the next | ≤ G; 44 under `bank_affine_runs` on the 11×10 grid (banks of 6 and 5 cores) | — |
| `core_pages_max` | flat: `ceil(tensor_pages / num_cores)`. `bank_affine_runs`: `ceil(ceil(tensor_pages / num_banks) / smallest group)` | ≤ 18 on flat INPUTS (Q8), ≤ 77 under `bank_affine_runs` (p1); unbounded otherwise | clamped by `block_pages_l1_cap` in `block_pages` |
| `block_pages_l1_cap` | `floor(L1_BUDGET_BYTES / (CB_DEPTH_KNOB · aligned_page_bytes))` | 164 (f32 tile), 329 (bf16 tile), 131 (5120 B row), 10529 (64 B row) | — |
| `CB_DEPTH_KNOB` | host constant | 2 | — |
| `BLOCKS_PER_CORE_KNOB` | host constant: blocks a core's assignment is cut into before the L1 cap | 1 (p1, 2–7 blocks at depth 2: within noise of 1; 28 blocks: +10%) | — |
| `L1_BUDGET_BYTES` | host constant | 1347776 | model environment, BB-05 |

### Total per-core footprint

```
L1(core) = cb_depth_blocks · block_pages · aligned_page_bytes  ≤  L1_BUDGET_BYTES
```

- `block_pages` scales with the core's assignment (`tensor_pages / num_cores`), divided by `BLOCKS_PER_CORE_KNOB`, up to the L1 cap.
- `cb_depth_blocks` is 1 until the cap splits a core's assignment, then 2.
- `aligned_page_bytes` is fixed by dtype and layout.
- No term scales with the tensor size once the cap binds.

| Case | `cb_depth_blocks · block_pages · aligned_page_bytes` | Bytes per core | Fraction of 1347776 |
|------|------------------------------------------------------|----------------|---------------------|
| p1 `(16,12,128,128)` f32, `bank_affine_runs` | 1 · 77 · 4096 | 315392 | 0.234 |
| p2 `(8,12,128,128)` f32, `bank_affine_runs` | 1 · 39 · 4096 | 159744 | 0.119 |
| `(1,12,128,128)` f32, `bank_affine_runs` | 1 · 5 · 4096 | 20480 | 0.015 |
| `(1,1)` f32, `logical_rows_partial_tile` | 1 · 1 · 4096 | 4096 | 0.003 |
| p3 `(1,512,5120)` bf16, `bank_affine_runs` | 1 · 64 · 2048 | 131072 | 0.097 |
| p4 `(1,1,32,62080)` bf16, `bank_affine_runs` | 1 · 49 · 2048 | 100352 | 0.074 |
| Q8 `(1,1,1,62080)` bf16, `logical_rows_partial_tile` | 1 · 18 · 2048 | 36864 | 0.027 |
| `(16,3,2560)` bf16 RM, `bank_affine_runs` | 1 · 2 · 5120 | 10240 | 0.008 |
| `(1,)` int32 RM, `flat_page_range` | 1 · 1 · 64 | 64 | < 0.001 |

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

## Data-movement budget (regime `logical_rows_partial_tile`, Refinement 1)

Same CB, same capacity formula and same live set: the regime changes only which bytes of a staged page the
reader fills and the writer drains. Bytes outside the moved ranges stay as allocated in both the CB slot and
`out`, and are never read as data.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` | ≤ 1 per stored byte; only the logical face rows of a partial page (64 B granules, merged across gaps ≤ `PIECE_MERGE_GAP_BYTES`) | Each page is read once, by one core | 0 |
| `out` | the same ranges, written once | The writer drains exactly the ranges the reader landed | 0 |

| Case | DRAM bytes (x read + out write) | vs `flat_page_range` | Cross-core NoC bytes |
|------|---------------------------------|----------------------|----------------------|
| `[1,1,1,62080]` bf16 (Q8), 1940 pages × one 576 B span | 1117440 + 1117440 = 2234880 | 7946240 (−72%) | 0 |
| `[1,5120]` bf16 (F13), 160 pages × 576 B | 92160 + 92160 = 184320 | 655360 (−72%) | 0 |
| `(1,1)` f32 (F7), 1 page × 64 B | 64 + 64 = 128 | 8192 | 0 |
| `(1,1)` bf16 (F8), 1 page × 64 B | 64 + 64 = 128 | 4096 | 0 |

The logical minimum on Q8 is 124160 B each way. The regime moves 9× that because one-row bf16 pieces are
read-request-bound: two 64 B ranges per page (248320 B each way) measured 9.35 µs against 7.06 µs for one
576 B span, so the knob trades bytes for requests (`op_design.md` → Regimes).

> **Deferral resolved.** Phase 0 deferred `logical_rows_partial_tile` while bfloat16 was outside SUPPORTED.
> Refinement 1 added bfloat16 together with the regime: tile-aligned and ROW_MAJOR shapes stay on
> `flat_page_range`, which is the cheapest-traffic split for them, and partial-tile TILE shapes run
> `logical_rows_partial_tile`.

## Data-movement budget (regime `bank_affine_runs`, Refinement 2)

Same CB, same capacity formula and same live set as `flat_page_range`; the regime changes which pages a
core takes (one bank's run `b, b + num_banks, …` instead of a consecutive range), which cores run (44 on the
banks' anchor rows), and which DRAM endpoint both kernels address (`BANK_AFFINE_ENDPOINT_NOC`'s, on the
cores' row). Bytes per tensor are unchanged.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` | 1 (read) | Each page is read once, by the one core whose bank run holds it, as part of one contiguous transfer per block | 0 |
| `out` | 1 (write) | Each page is written once from the slot it was read into; `out` has `x`'s spec, so the run is contiguous in the same bank | 0 |

| Case | DRAM bytes (x read + out write) | vs `flat_page_range` | Cross-core NoC bytes |
|------|---------------------------------|----------------------|----------------------|
| p1 `(16,12,128,128)` f32 | 12582912 + 12582912 = 25165824 | equal | 0 |
| p2 `(8,12,128,128)` f32 | 6291456 + 6291456 = 12582912 | equal | 0 |
| `(16,3,2560)` bf16 RM | 245760 + 245760 = 491520 | equal | 0 |

The CB grows with the per-core assignment (p1: 28 → 77 pages, 0.085 → 0.234 of the budget) because 44
cores carry what 110 did; it stays one block per core on every INPUTS shape.
