# L1 Ledger: g6_expand

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions:
`op_design.md` → Blocking Model (regime R1).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_bcast_tiles` | `bcast_depth * col_block`, `col_block = min(max_cols_per_core, COL_BLOCK_CAP)` | `col_block` pages while a block is fetched (all of the block's row-0 fetches in flight under one barrier); `bcast_depth * col_block` when block k+1 is fetched while block k drains (only if `num_col_blocks_max > 1`) | `{image: streams → folded into col, col (wt): spans → col_block, ht: streams → one page serves the whole segment, tile rows (32): spans → whole tile, tile cols (32): spans → whole tile, grid: n/a}` | Float16_b (bf16 bytes; dataflow-only payload, never through DEST — no `fp32_dest_acc_en` exists for this op) | reader | writer | whole program | Only CB of the op. The fetch destination, the replicate target and the write source are the **same slot** (fetch lands in place, replicate in place, write reads it in place) — no staging CB and no output CB exist to share with. Capacity = live set (depth 2 only when there are ≥ 2 blocks to overlap) |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `max_cols_per_core` | `≤ ceil(max n_t / Ht) + 1` | column-major contiguous ranges (`op_design.md` Work Distribution); production ≤ 2 |
| `COL_BLOCK_CAP` | 64 | host constant (one source) |
| `col_block` | `1 ≤ col_block ≤ 64` | `min(max_cols_per_core, COL_BLOCK_CAP)` |
| `bcast_depth` | `∈ {1, 2}` | `1` iff `ceil(max_cols_per_core / col_block) == 1` |
| page size | 2048 B | bf16 32×32 tile |

## Total per-core footprint

`L1 = bcast_depth * col_block * 2048 B` — scales with `col_block` (the W-axis knob) and
`bcast_depth`; independent of `Ht`, `W`, `N_img` beyond the `col_block` clamp.
Bound: `2 * 64 * 2048 = 262 144 B` < 1 347 776 B (BB-05) for every shape. Production:
W=512 → `1 * 2 * 2048 = 4 KiB`; W ≤ 256 → `2 KiB`.

## Data-movement budget (regime R1)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| out | 1 (writes only, every page once) | each output tile is written by exactly one core straight from the resident broadcast tile; out is never read | none |
| x | row 0 of each input page (`2 × row_fetch_bytes` = 128 B on BH, 64 B on WH) fetched once per core whose range touches that column: `Σ_cores cols_this_core ≤ num_cores + num_cols` fetch pairs; padding rows/the other faces never cross | the broadcast tile is resident per core for its whole column segment, so x is never re-read within a core | none |

Totals per tier (W = 512, 128 cores, BH): DRAM writes 524 288 B; DRAM reads ≤ (128 + 16) × 128 B ≈ 18 KiB (≈ 3.5 % of writes); NoC cross-core 0 B; core-local L1 replicate 2 KiB per (core, column).

> Cheapest-traffic split considered: R2, column-group multicast — x DRAM reads drop from ≈ 18 KiB to 2 KiB (16 KiB, ≈ 3 % of the op's DRAM bytes) at the cost of ≈ 256 KiB of on-chip multicast and a semaphore handshake per column group. Implemented: R1, column-major linear split with per-core row-0 fetch. Deferred because the saved DRAM bytes are a few percent of a write-bound op and R2 puts a cross-core handshake on the critical path of a ~1.5 µs call; R1's column-major order keeps each column's cores a consecutive run, so R2 layers onto it without changing the writer.
