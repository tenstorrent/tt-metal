# L1 Ledger: g6_expand

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `pattern`
(extent `block_pattern_tiles`) and `replica` (extent `block_replica_rows = core_replica_rows`);
`image` is folded into `pattern` (see `op_design.md`, Blocking Model).

## Inventory

One buffer. Every phase boundary that could be storage is not:

| Phase boundary | Expressed as | Why no buffer |
|----------------|--------------|---------------|
| DRAM fetch of row 0 → replicate | pack into the destination | the fetch lands row 0 directly in the pattern page at its final offsets (0, 512); no staging slot |
| replicate → multicast | transform in place | rows 1–31 are filled from row 0 inside the same page; the multicast sources the same pages |
| multicast → receiver | pack into the destination | the payload lands in the receiver's reserved `cb_pattern_tiles` pages at the same address |
| pattern page → every output page of its column | resident source, streamed writes | the writer re-reads the same resident pages for every replica row; no per-row buffer |

## Table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_pattern_tiles` (index 0, page 2048 B) | `pattern_cb_depth_blocks × block_pattern_tiles` | `block_pattern_tiles` pages (one block: every page of it is read for every replica row, so all are live at once) | `{pattern: spans → block_pattern_tiles, replica: streams → 0 pages (writes stream over resident pages), image: folded into pattern}` | bfloat16 (raw bits: dataflow-only payload, never unpacked or packed, no DEST; `fp32_dest_acc_en` does not exist for this op) | reader | writer | one block per CB cycle; every block of the kernel | Shares with nothing: it is the only buffer. Capacity equals the live set at depth 1. At depth 2 the gap is the explicit pipelining decision (block `k+1` fetch/replicate/multicast overlapping block `k` writes), used only when `num_blocks_this_core > 1`. |

Semaphores (not CBs, listed for the footprint): the `ttnn.Mcast` channel's ready and data
semaphores, on the rectangle, bounded by `L1_RESERVED_BYTES`.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tile_bytes` | 2048 | dtype ∈ SUPPORTED = TARGET = {bfloat16}, 32×32 tile |
| `block_pattern_tiles` | `1 ≤ block_pattern_tiles ≤ max_block_pattern_tiles = (L1_OP_BUDGET_BYTES − L1_RESERVED_BYTES) // (pattern_cb_depth_blocks × tile_bytes)` = 656 at depth 1, 327 at depth 2 | clamp in `select_plan`; `block_pattern_tiles = min(pattern_tiles_per_group, max_block_pattern_tiles)` |
| `pattern_cb_depth_blocks` | ∈ {1, 2}; Phase 0 = 1 | host constant |
| `L1_OP_BUDGET_BYTES` | 1,347,776 | spec, Model environment (BB-05) |
| `L1_RESERVED_BYTES` | 4,096 | host constant; covers the multicast semaphores (2 × L1-aligned words) and allocation alignment |
| `core_replica_rows` | unbounded (≤ `tensor_replica_rows`) | not in any capacity expression (`replica` streams) |
| `tensor_pattern_tiles`, `tensor_replica_rows`, W, H_out | unbounded | not in any capacity expression: they set the plan and trip counts, never a page count |

## Total per-core footprint

```
L1(core) = pattern_cb_depth_blocks × block_pattern_tiles × tile_bytes  +  mcast semaphores
         ≤ (L1_OP_BUDGET_BYTES − L1_RESERVED_BYTES) + L1_RESERVED_BYTES = 1,347,776 B
```

The only term that scales with a knob is the first: linear in `block_pattern_tiles` and in
`pattern_cb_depth_blocks`. Nothing scales with `core_replica_rows`, W or H_out. Production
(depth 1): 4,096 B at W=512 (`block_pattern_tiles = 2`), 2,048 B for every other width.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` | **1** for the 64 B of row 0 per input tile (two 32-B reads); **0** for rows 1–31 | each pattern range is fetched once, by its grid column's top core (injector), and shared down the column by multicast | per grid column, per block: `this_block_pattern_tiles × 2048` B multicast once to `replica_groups − 1` receivers, plus the handshake (one ready increment per receiver, one flag multicast) |
| `out` | **1** (write) | every output page is written exactly once, from a resident pattern page | none |

Totals per production case (11×10 grid):

| W | `x` DRAM read | `out` DRAM write | Multicast injected (per column × columns) | Receivers per column |
|---|---------------|------------------|-------------------------------------------|----------------------|
| 512 | 1,024 B | 524,288 B | 4,096 B × 8 | 7 |
| 256 | 512 B | 131,072 B | 2,048 B × 8 | 7 |
| 207 | 448 B | 100,352 B | 2,048 B × 7 | 6 |
| 196 | 448 B | 100,352 B | 2,048 B × 7 | 6 |
| 185 | 384 B | 73,728 B | 2,048 B × 6 | 5 |

> Cheapest-traffic split considered: pattern-only (`replica_groups = 1`, regime R2): same DRAM
> crossings as R1 (`x` row 0 once, `out` once), and **0 B** cross-core instead of R1's
> 8 × 4 KB (W=512) to 6 × 2 KB (W=185) multicast. Implemented: R1, 2D column multicast.
> R2 is R1 at `max_replica_groups = 1`: the same kernels with no receivers, reachable by one
> host-knob turn. It is not the Phase-0 value because its core count is
> `pattern_groups ≤ ceil(W / 32)` (6–16 cores), fixed by the tensor's shape, which the spec
> forbids for multi-tile outputs. The two differ only in cross-core bytes, which R1 keeps to
> a multicast of at most 4 KB per column.
