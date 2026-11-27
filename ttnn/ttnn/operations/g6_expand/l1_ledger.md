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
| `split_cores` | `min(grid cores, MAX_CORES=80, ceil(total_tiles / MIN_TILES_PER_CORE))` | `_split_core_count` (Refinement 1; W=512 → 80 cores, `max_cols_per_core` = 1 → `col_block` = 1, 2 KiB) |
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

## Implementation notes (ttnn-implementer)

Inventory unchanged: `cb_bcast_tiles` is still the only CB, sized `bcast_depth * col_block * 2048 B`.
Two perf lamps of the design were turned on as live CT knobs (host constants in
`g6_expand_program_descriptor.py`); neither changes DRAM crossings:

- `REPLICATE_CPU_ROWS = 8` (replication mechanism): the reader RISC-V builds rows 0..7 of faces 0/1,
  then local NoC L1→L1 copies double them to 16 rows and copy faces 0/1 onto faces 2/3, inside the same
  slot. That adds 2 KiB of core-local NoC traffic per (core, column) and no DRAM bytes. `2 * FACE_ROWS`
  restores the all-RISC-V copy.
- `READER_WRITE_EIGHTHS = 4` (write issue on one RISC): the reader writes the second half of every column
  segment on NoC0 from the slot it built, and the writer writes the rest on NoC1. Every output page is
  still written once, by one core. The reader flushes before any re-reserve and barriers before return.
- `CORE_ORDER_STRIDE = 1` (core placement): measured flat at strides 7, 13 and 37, so it stays at the
  split's own order.

### Refinement 2 (critical-path restructure)

- `cb_bcast_tiles` page = **slot** = `slot_bytes` = 1 KiB when `HALF_TILE=1` (default; faces 0/1
  only — an output tile is `[F0 F1 F0 F1]`, so each output page is written as two 1 KiB writes from
  the one slot), 2 KiB when `HALF_TILE=0`. Capacity `bcast_depth * col_block` pages unchanged →
  production footprint W=512 `2 * 1 KiB`, W ≤ 256 `1 KiB`; bound `2 * 64 * 1 KiB` (half) /
  `2 * 64 * 2 KiB`.
- Still the only CB. Producer of the CB handshake stays the reader; with `REPLICATE_SPLIT=1` (default,
  gated to pure-RISC-V build + one block per core) the **writer also writes bytes into the slot**:
  it fetches row 0 of face 1 (and builds face 1/3) into the slot at `get_read_ptr + j * slot_bytes`,
  while the reader fetches/builds face 0/2 only — disjoint byte ranges, no slot reuse (single block).
  BRISC→NCRISC "face 1 built" is a local counter semaphore `SEM_FACE1_BUILT` (no NoC, no CB); the
  reader polls it before its write share. Multi-block cores (`max_cols_per_core > COL_BLOCK_CAP`)
  keep the reader-only build (slot reuse would race the writer's build against the reader's in-flight
  writes).
- Data-movement budget unchanged: x — still `2 × row_fetch_bytes` per (core, column), now one request
  on each RISC/NoC; out — every page written once (as 2 × 1 KiB with `HALF_TILE=1`). Core-local
  replicate traffic: 0 B NoC (pure RISC-V stores, 15 rows per face, faces split over the two RISCs).
