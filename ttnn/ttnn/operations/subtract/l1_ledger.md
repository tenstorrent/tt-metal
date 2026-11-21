# L1 Ledger: subtract

Target: `quasar` (ttsim), worked values for Phase 0 (regime R0, bfloat16 operands and output, `fp32_dest_acc_en = True`, half-sync DEST, plan (1, 1, 1)). Schema: `.claude/references/l1-footprint-discipline.md`.

Named block axes (from `op_design.md` Blocking Model): `tile_id` per cluster (extent `core_tiles`, the block), `tile_id` per NEO (quantum `sync_tiles`). `N`, `Ht`, `Wt` are folded into `tile_id` and listed for completeness.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_minuend_tiles` | `ENTRIES_PER_NEO · C` = `2 · sync_tiles · C` = 8 | `sync_tiles · C` (one quantum per NEO under compute) + up to `sync_tiles · C` in flight from the reader | `{tile_id/cluster: streams → core_tiles never resident; tile_id/NEO: spans → sync_tiles; N, Ht, Wt: streams (folded)}` | bfloat16 (= `a.dtype`): dataflow payload straight from the tensor, never passed through DEST before compute reads it | reader | compute | whole kernel | cannot share with `cb_subtrahend_tiles`: concurrent lifetime (both operands of the same FPU op are live together); cannot share with `cb_difference_tiles`: concurrent (pipelined) and a ring has one producer and one consumer. Capacity = 2 × live quantum: double buffering (mechanism reason) |
| `cb_subtrahend_tiles` | `2 · sync_tiles · C` = 8 | as above | `{tile_id/cluster: streams; tile_id/NEO: spans → sync_tiles; N, Ht, Wt: streams (folded)}` | bfloat16 (= `b.dtype`), same reason | reader | compute | whole kernel | concurrent with `cb_minuend_tiles` (same op's other operand) and with `cb_difference_tiles` (pipelined); double buffering |
| `cb_difference_tiles` | `2 · sync_tiles · C` = 8 | `sync_tiles · C` being packed + up to `sync_tiles · C` draining to the writer | `{tile_id/cluster: streams; tile_id/NEO: spans → sync_tiles; N, Ht, Wt: streams (folded)}` | output dtype (bfloat16). A 16-bit page while 32-bit DEST is on: output ABI requirement — the output tensor is bfloat16 and rule R6 asks for the FP32 difference rounded **once** into it, which is this pack; no later phase reads it back | compute | writer | whole kernel | cannot pack in place into an input ring: an input ring's producer is the reader (one producer per ring) and the output ring's consumer is the writer; concurrent lifetime; double buffering |

No intermediate ring exists: the subtract packs straight into the destination ring (Rule 3, pattern 1), so the phase boundary is not storage.

## Symbol table

| Symbol | Bound | Established by |
|---|---|---|
| `sync_tiles` | `1 ≤ sync_tiles ≤ dest_tiles(fp32_dest_acc_en, dst_full_sync_en)` ∈ {4, 8, 16}; Phase 0 = 4 | host clamp `min(SYNC_TILES, dest_tiles(...))`, same table as `get_dest_limit()` (`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:87-100`) |
| `ENTRIES_PER_NEO` | `= 2 · sync_tiles` ≤ 32 | derived on host from `sync_tiles` (single source) |
| `C` | ∈ {1, 2, 4}; Phase 0 = 1 | `THREAD_PLAN`, checked by `validate_thread_plan` |
| `page_bytes` | 2,048 (bfloat16 tile); 4,096 (float32 tile, later regimes) | tensor `buffer_page_size()` |
| `core_tiles`, `tensor_tiles` | unbounded op dimensions — **appear in no capacity** | — |
| L1 budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 B on this target (probe, 2026-11-21) | run-time query; no Phase 0 predicate needs it |

## Total per-core footprint

`L1 = 3 · ENTRIES_PER_NEO · C · page_bytes = 6 · sync_tiles · C · page_bytes`

- Phase 0 on this target: `6 · 4 · 1 · 2,048 B = 49,152 B` (48 KiB, 1.2 % of the unreserved budget).
- Worst case of every knob at its bound: `6 · 16 · 4 · 4,096 B = 1,572,864 B` (1.5 MiB) — fits; each ring `16 · 2 · 4 · 4,096 = 524,288 B` is under the 1,048,560 B ring cap.
- Scales linearly with `sync_tiles` (handshake quantum), `C` (thread plan) and `page_bytes` (dtype); constant in the op's shape. No budget predicate, safety fraction or blocking search is needed.

## Data-movement budget (regime R0, the chosen split)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `a` (`tensor_tiles · 2,048 B`) | 1 | each tile is read by exactly one cluster, once, and consumed from the ring; nothing is re-read because no tile is needed twice | none |
| `b` (`tensor_tiles · 2,048 B`) | 1 | same (Phase 0: `b` has `a`'s shape, so it is not shared) | none |
| `out` (`tensor_tiles · 2,048 B`) | 1 | each tile written once by its owning cluster | none |

(For an L1 cell the "DRAM" tier is the L1 banks of the interleaved buffer, reached over the NoC; the counts are the same.)

Total per tier: DRAM `3 · tensor_tiles · 2,048 B` (e.g. `[1,2,33600]`: `3 · 1,050 · 2,048 = 6,451,200 B`); cross-core 0 B; core-local: each tile passes through its ring once.

> Cheapest-traffic split considered: flat `tile_id` contiguous ranges — `a`, `b`, `out` once each across DRAM, 0 B cross-core. Implemented: the same split. It is also the cheapest: every candidate split (tile-rows, tile-columns, 2D blocks) moves the same bytes because no operand is shared and no footprint overlaps; the flat split wins the tiebreaker on occupancy (the YOLO shapes are one tile-row tall). Nothing is deferred on traffic grounds in Phase 0; the broadcast regimes' traffic (R1 → R1b) is recorded in `op_design.md`'s regime table.
