# L1 Ledger: permute

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes, from `op_design.md`'s Blocking Model: `lead0`, `lead1`, `h`, `w` (rank-4 canonical input axes). Each regime is its own `ttnn.ProgramSpec`, and exactly one runs per call, so the buffers of different regimes never coexist. The ledger is read per regime.

Target for the worked values: Quasar. `ttnn.get_max_worker_l1_unreserved_size()` = 3,963,328 B (probe, 2026-11-21); 32 clusters; bf16.

## Buffers

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_pages` (`page_gather`) | `page_ring_depth × max(R, W)` = 8 × 1 = 8 entries of the tensor's aligned page slot, `align_up(page_bytes, buffer alignment)` (implementation: a STRIDED ring needs `num_entries % max(P, C) == 0`, so the depth is counted per thread of the wider endpoint; the ring cap check uses the product). `page_bytes` = `tile_bytes` (2048) in TILE, `s[3]·element_bytes` in RM. | 8 entries: 4 per implicit-sync txn-id half in flight. Capacity = live set, so nothing is idle. | `{lead0: streams, lead1: streams, h: streams (RM) / spans → 32 (TILE, in-page), w: spans → s[3] (RM, whole row in-page) / spans → 32 (TILE, in-page)}` | bf16 (the input dtype; a pure DM payload that never touches DEST) | reader | writer | whole `page_gather` program | Nothing else exists in this program. A single ring is the minimum for a reader → writer hand-off, and NoC writes must leave from L1. |
| `cb_input_tiles` (`tile_transpose`) | `transpose_ring_blocks × transpose_block_tiles × C` = 2·4·1 = 8 tiles of 2048 B | 8 tiles: one compute window being unpacked plus one being filled (double buffer) | `{lead0: streams, lead1: streams, h: spans → 32 (in-tile), w: spans → 32 (in-tile)}`. The window spans `transpose_block_tiles` tiles of the flattened run. | bf16 (16-bit DEST: `enable_32_bit_dest=False`; matches audit 2) | reader | compute | whole `tile_transpose` program | Cannot share with `cb_output_tiles`: both are live at once (pipelined), and an in-place transpose is impossible because the reader (a DM kernel) produces this ring while compute packs the other, and a ring has one producer. |
| `cb_output_tiles` (`tile_transpose`) | 2·4·1 = 8 tiles of 2048 B | 8 tiles: one window being packed plus one being written | Same as `cb_input_tiles` | bf16 (16-bit DEST) | compute | writer | whole `tile_transpose` program | See `cb_input_tiles`. Packing straight to the output tensor is not possible because the packer writes only to local L1. |
| `scratch_gather` staging (`staged_gather`; part of one `ScratchpadSpec`, one region per worker thread) | `staging_bytes(e)`, per thread: RM `Π_{j≠3} e_j × align_up(e_3·element_bytes, 64)`; TILE `e_0·e_1·ceil(e_2/32)·ceil(e_3/32)·tile_bytes` | The whole footprint of the box being shuffled. The shuffle reads arbitrary positions of the box for every output page, so everything is live together (`staging_depth` = 1). | `{lead0: spans → e_0, lead1: spans → e_1, h: spans → e_2, w: spans → e_3}` (the box, rounded to whole input pages / 64 B slots) | raw bytes, element width `element_bytes` (no format metadata: only DM touches it) | the worker thread that owns the region (NoC reads) | the same thread (RISC-V loads) | each block, inside the `staged_gather` program | Shares one scratchpad allocation with the output slots: one `ScratchpadSpec`, sized to the sum. They cannot alias because staging is read while slots are written in the same loop. Threads cannot share regions because each owns a different block at the same time. |
| `scratch_gather` output slots (`staged_gather`; same `ScratchpadSpec`) | `out_slots` × `align_up(slot_bytes, 64)` per thread: `slot_bytes` = `tile_bytes` (TILE) or `e_b·element_bytes` (RM) | `out_slots` = 8 pages in flight: one being assembled, up to 7 being written | `{lead0: streams, lead1: streams, h: streams, w: streams}`, except the output's in-page axes, which it spans: TILE `a: spans → 32`, `b: spans → 32`; RM `b: spans → e_b`, with `b` one of the four axes | raw bytes | the owning worker thread (RISC-V stores) | the same thread (NoC writes) | each block | Same allocation as staging. A slot cannot be the staging itself, because an output page gathers from many staged pages, so output assembly needs its own bytes. Capacity exceeds the 1-page minimum by 7 so that write-flush waits happen once per 8 pages (perf lamp `out_slots`). |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `element_bytes` | 2 (bf16, PHASE0); 4 later | `validate()` on `dtype` |
| `tile_bytes` | 2048 (bf16) | layout = TILE |
| `l1_align` | `ttnn.get_l1_alignment()` (16 here) | device query |
| `s[3]` (RM row elements) | ≤ 65,535 for bf16, so the ring is ≤ 1,048,560 B | `validate()` refuses a larger RM row (`NotImplementedError`, ring cap). Applies to the `page_gather` RM path. |
| `page_ring_depth` | 8 per thread of the wider ring endpoint (`× max(R, W)` entries) | host constant `PAGE_RING_DEPTH` |
| `transpose_block_tiles` | `min(4, DEST_AUTO_LIMIT)` ≤ 8 | host constant clamped by DEST capacity |
| `transpose_ring_blocks` | 2 | host constant |
| `C` (`tile_transpose` compute threads) | 1 | `THREAD_PLANS["tile_transpose"]` |
| `gather_threads` | 6 | `THREAD_PLANS["staged_gather"]` (R + W ≤ 6) |
| `threads_engaged` | `1 ≤ threads_engaged ≤ 6`, = `min(6, floor(budget / min_per_thread_bytes))` | Mechanism cap; `validate()` refuses if 0 |
| `out_slots` | 8 | host constant |
| `l1_reserve_bytes` | `262,144 + Σ_{L1 tensors} ceil(num_pages / num_l1_banks) × aligned_page_bytes` | Host. `num_l1_banks` from `ttnn.device.GetMemoryView(device, ttnn.BufferType.L1).num_banks`. |
| `budget` | `ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes` (3,701,184 B here with DRAM tensors) | device query minus reserve |
| `budget_per_thread` | `align_down(budget / threads_engaged, 64)` (616,832 B here with 6 threads) | derived |
| `gather_extent[0..3]` (`e_j`) | `per_thread_bytes(e) ≤ budget_per_thread`, with each `e_j ≤ s[j]`, a multiple of its quantum or the whole axis | `choose_gather_block()` steps 2 (fit) and 4 (coarsen only while it still fits) |
| `e_b` (RM output segment) | ≤ `s[b]`, a multiple of `align_out_elems` or the whole axis | `choose_gather_block()`, same bound |

## Total per-core footprint

| Regime | Expression | Scales with | Worked (bf16, this target) |
|--------|-----------|-------------|----------------------------|
| `page_gather` | `page_ring_depth × align_up(page_bytes, l1_align)` | `page_ring_depth`; `s[3]` in RM | TILE 16,384 B; RM W=8400 → 8 × 16,800 = 134,400 B; RM cap 1,048,560 B |
| `tile_transpose` | `2 × transpose_ring_blocks × transpose_block_tiles × C × tile_bytes` | `transpose_block_tiles`, `transpose_ring_blocks`, `C` | 32,768 B |
| `staged_gather` | `threads_engaged × (align64(staging_bytes(e)) + out_slots × align64(slot_bytes(e)))` ≤ `budget` | Staging: the product of the four box extents (a product of affine extents, rounded to page slots). Slots: `out_slots` × the output in-page extent. Total: × `threads_engaged`. | Worst PHASE0 case is TILE with `a`, `b` both leading and ≥ 32: 1 thread × (2,097,152 + 8 × 2048) = 2,113,536 B. The traced calls use ≤ 6 × 30 KB. |

## Data-movement budget

Crossings count whole-tensor passes over DRAM (over the L1 banks' NoC for an L1-resident tensor).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (`page_gather`) | 1 | Each output page names exactly one input page, read once into `cb_pages` | none |
| output (`page_gather`) | 1 | Each output page written once from `cb_pages` | none |
| input (`tile_transpose`) | 1 | Each output tile names exactly one input tile | none |
| output (`tile_transpose`) | 1 | Each output tile written once | none |
| input (`staged_gather`) | 1 | Boxes partition the input on whole pages / aligned row segments. The whole box footprint stays resident in staging until every output page drawing on it is assembled, so no input page is re-read. | none |
| output (`staged_gather`) | 1 | Each output page lies in exactly one box: the box covers the output's in-page axes in whole quanta. The output is written once, including TILE padding (written 0). | none (several clusters write disjoint aligned segments of one RM row; that is ordinary DRAM traffic, not a combine) |

Total bytes per tier, all regimes: DRAM (or L1 banks) = input bytes + output bytes (padded for TILE). Cross-core = 0. Core-local: `staged_gather` adds one RISC-V read of every staged element and one write of every output element.

> Cheapest-traffic split considered: output-page / output-major box split, with every input byte read once and every output byte written once (no cross-core traffic). Implemented: the same split, in all three regimes. Deferred: nothing on traffic grounds. The implemented split is the cheapest. The deferred rows in the regime table (`staged_gather_subtile`, `cube_fpu_assist`, `rm_plane_transpose_neo`) change L1 fit or the shuffle stage's cost, not bytes moved.
