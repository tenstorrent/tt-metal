# L1 Ledger: nomic_qkv_rotary_heads

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes, from op_design.md § Blocking Model:
**M** (`block_m_tiles`), **N** (`LANE_N_TILES`, the core's whole lane), **K** (`K_BLOCK_TILES` per block, `K_TILES`
in total). Every CB is declared on every core of the grid with the same size and order: the in0 and in1 multicasts need
equal slot addresses everywhere.

## Inventory decisions (Rule 1 / Rule 3, before any sizing)

| Candidate buffer | Decision | Pattern |
|------------------|----------|---------|
| bias CB | **Removed.** The bias is K tile 24 of the weights (with a ones tile in x's K tile 24), so the matmul adds it in fp32 DEST | fold into the accumulator |
| fp32 → bf16 output pass / finished-sum scratch | **Removed.** The last K block reloads the partials into DEST, and the helper packs the finished tile straight into `cb_heads_out` (`LastBlockTarget::Out`) | pack into the destination |
| rotary intermediates (t·C, t·S products, rotated-partner tiles) | **Removed.** The rotary runs in DEST (t0, t1, C, S in the four slots of a half) and packs once | fold into the accumulator |
| ones tile scratch | **Removed.** The column's top core writes the ones pattern into K block 4's slot positions in place (NoC `async_write_zeros` of the tile, then 32 bf16 stores of 1.0 in column 0) | transform in place |
| separate q, k, v output CBs | **Removed.** One `cb_heads_out`; the writer picks the tensor per tile | pack into the destination |

## Ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_block` (0) | `X_BLOCK_DEPTH * block_m_tiles * K_BLOCK_TILES` = 2·3·5 = 30 | 30: compute reads slot b while the multicast fills slot b+1 | `{M: spans → block_m_tiles, N: streams (x is the same for every lane; nothing N-sized held), K: spans → K_BLOCK_TILES, streams over num_k_blocks}` | Float16_b (x is bf16 tiles) | reader | compute | per K block, every M block | None: live in every K block of every M block, concurrent with all the others. Depth 2 is the double-buffer mechanism (catalog `double_buffer`; pirotski §3.2: a third slot buys nothing). |
| `cb_weights` (1) | `K_TILES * LANE_N_TILES` = 25·8 = 200 | 200: resident; every K block is re-read in every M block | `{M: streams (re-used by every M block, nothing M-sized), N: spans → LANE_N_TILES, K: spans → K_TILES (all of K)}` | Float16_b (W and bias bf16-exact) | writer | compute | whole op | None: live for the whole op. Capacity equals the live set and is pinned by the resident-ring mechanism (capacity must equal one full K x lane cycle). |
| `cb_rotary_table` (2) | `ROTARY_TABLE_DEPTH * TABLE_TILES_PER_ROW * block_m_tiles` = 1·2·3 = 6 (host passes `table_block_tiles = TABLE_TILES_PER_ROW * block_m_tiles` to reader and compute) | 6: every row's C and S are live through the block's last K block (each head of a row copies them into DEST) | `{M: spans → block_m_tiles, N: streams (one C/S pair serves every head of the row), K: streams (needed in the last K block only)}` | Float32 (C/S fp32; also equals `cb_partials`' format, so the hook's copy needs no srcA reconfig after the reload) | reader | compute | from its request (after in0 K block 0 of block m) to the end of block m's call | Not with `cb_partials` (concurrent in the last K block) or `cb_heads_out` (block m−1's drain overlaps block m). Declared but unused on v cores, which keeps equal addresses for the multicast CBs. |
| `cb_partials` (3) | `block_m_tiles * LANE_N_TILES` = 3·8 = 24 | 24: the whole output block accumulates across K blocks 0..3 (packer L1 accumulation) | `{M: spans → block_m_tiles, N: spans → LANE_N_TILES, K: streams (accumulated over num_k_blocks − 1 blocks)}` | Float32 (fp32_dest_acc_en; packer L1 accumulation needs the DEST format) | compute | compute | K blocks 0..3 of each M block, plus the per-subblock reload in the last K block | Not with `cb_heads_out`: block m+1 accumulates while block m drains (explicit pipelining decision), and the formats differ (fp32 vs bf16). Not with `cb_x_block`: concurrent. |
| `cb_heads_out` (4) | `OUT_DEPTH_BLOCKS * block_m_tiles * LANE_N_TILES` = 1·3·8 = 24 | at most 24. Peak is the whole block when the writer has not yet started draining it; compute pushes all 24 during the last K block. | `{M: spans → block_m_tiles, N: spans → LANE_N_TILES, K: streams (filled only in the last K block)}` | Float16_b (the one rounding to bf16) | compute | writer | from block m's last K block until drained during block m+1's K loop | Not with `cb_partials` (above). Capacity is one whole block on purpose (pipelining decision): compute never waits on the writer. The writer pops per row (8 pages); 3 pops close the ring. |

## Symbols

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_m_tiles` | 1 ≤ `block_m_tiles` ≤ `BLOCK_M_TILES_MAX` = 3 | `num_m_blocks = ceil(core_m_tiles / 3)`, `block_m_tiles = ceil(core_m_tiles / num_m_blocks)` |
| `core_m_tiles` | 3 ≤ `core_m_tiles` ≤ 12 | `core_m_tiles = ceil(M / 11)`, 32 ≤ M ≤ 128 (validate) |
| `LANE_N_TILES` | = 8 (constant) | 72 N tiles / 9 lanes; even (head-pair cap) |
| `K_TILES` | = 25 (constant) | 24 hidden tiles + 1 bias tile |
| `K_BLOCK_TILES` | = 5 (constant) | divides `K_TILES` (mechanism cap) |
| `X_BLOCK_DEPTH`, `ROTARY_TABLE_DEPTH`, `OUT_DEPTH_BLOCKS` | 2, 1, 1 | buffer-depth knobs (op_design.md) |
| `TABLE_TILES_PER_ROW` | = 2 (constant) | one C and one S tile per position tile row (`RotaryTables` page order) |

## Total per core

```
bytes = 2048 * (X_BLOCK_DEPTH*block_m_tiles*K_BLOCK_TILES + K_TILES*LANE_N_TILES + OUT_DEPTH_BLOCKS*block_m_tiles*LANE_N_TILES)
      + 4096 * (ROTARY_TABLE_DEPTH*TABLE_TILES_PER_ROW*block_m_tiles + block_m_tiles*LANE_N_TILES)
      = 409,600 + 77,824 * block_m_tiles        (Phase 0 depths)
```

| block_m_tiles | bytes | KiB | vs 640 KiB budget |
|---------------|-------|-----|-------------------|
| 2 (M = 32..39 when core_m_tiles = 4) | 565,248 | 552 | fits |
| 3 (every other M class) | 643,072 | 628 | fits (12 KiB margin) |
| 4 (not admitted) | 720,896 | 704 | over: this is why `BLOCK_M_TILES_MAX` = 3 |

The constant term (400 KiB) is the resident weights: they scale with `K_TILES * LANE_N_TILES`, not with M. Every other
term scales linearly with `block_m_tiles`. `circular_buffer_bytes(input_shape)` returns this expression for the shape's
plan, and `validate()` refuses a plan above 640 KiB (none exists with the Phase 0 knobs).

## Data-movement budget

At M = 128 (8x512). Sizes: x 128·24 tiles = 6.29 MB, W + bias 25·72 tiles = 3.69 MB, q/k/v 3 · 6.29 MB = 18.87 MB,
tables 32 fp32 tiles = 0.13 MB.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 0 (L1 shards) | Read in place from the norm's shards (ln_layout) or the model's block shards (model_layout); never copied | 1 gather (6.29 MB: each tile read once by its column's top core) + 1 multicast per column to 8 receivers. Under model_layout column c's rows sit in grid column c, so the gather is in-column reads (3-tile runs) |
| W + bias (`QkvWeights`) | 1 | Each 16 KB lane page (one K row x lane) is read by exactly one row sender, in M block 0 only. The resident `cb_weights` (whole K x lane) serves every later M block. | 1 multicast per row to `num_columns − 1` = 10 receivers |
| rotary tables | 0 (L1 interleaved) | Each q/k core reads its block's rows once per M block | 66 q/k cores x 12 rows x 2 x 4 KB = 6.3 MB of unicast L1 reads (the 6 q/k cores of a column read the same pages) |
| q, k, v | 1 write each when output_buffer = dram; 0 when l1 | Every output tile is packed once and written once to its page | 18.87 MB of page writes (L1 or DRAM banks) |
| QKV product | 0 | Never leaves `cb_partials` / DEST / `cb_heads_out` | 0 |

Totals at M = 128: DRAM reads 3.69 MB (the minimum: W once). DRAM writes 18.87 MB with DRAM outputs, otherwise 0.
Cross-core: x 6.29 MB gather + 11 column multicasts; W 9 row multicasts of 0.41 MB each; tables 6.3 MB unicast;
outputs 18.87 MB.

> Cheapest-traffic split considered: 2D with holder-direct x multicast (shard holders multicast into the column's
> slots). It saves 6.29 MB of L1→L1 x gather at M = 128; DRAM is identical (W once). With the table broadcast also
> folded in, it saves 5.2 MB of table reads more. Implemented: 2D with column-sender gather and per-core table reads.
> Deferred because both savings are on RISCV_0 traffic that is off the critical path. Per K block the gather reads
> `block_m_tiles·5` tiles while compute does `block_m_tiles·8·5` tile products. The table read is once per M block.
> Holder-direct needs a many-to-many slot handshake. Folding the tables into the in0 multicast would break
> `cb_x_block`'s uniform ring. Both stay reachable: the receiver side and slot layout are unchanged (op_design.md,
> Dataflow Strategy, contracts of the deferred regimes). The implemented split is the cheapest for DRAM.

## Implementation notes (ttnn-implementer)

- CB set, sizes, formats and order are exactly as in the ledger above (no CB added, merged or removed).
- `cb_partials` / `cb_heads_out` are driven by `matmul_block` with `OutputCBLayout::TileRowMajor`
  (`OUT_LAYOUT_ROW_MAJOR = 1`): spills, reloads and output pushes go per tile row (`LANE_N_TILES` tiles) rather
  than per subblock, and the helper makes no whole-block `cb_heads_out` reserve at K block 0, so compute no longer
  waits for the previous block's drain. Capacities are unchanged (`cb_partials` is still exactly one block, so every
  K block's row-group spills land on the same addresses for packer L1 accumulation).
- The data-movement budget is unchanged: W read once from DRAM (M block 0, row senders), x gathered once per column
  and multicast, tables read once per q/k core per M block, every q/k/v tile written once.

- Refinement 1 (`model_layout`): no CB change. x's BLOCK_SHARDED shard (32·bh x 96, at most 384 x 96 bf16 = 72 KiB
  at M = 128) is an io tensor beside the 628 KiB of CBs on the 8 x C shard cores, as ln_layout's shards are. The data-
  movement budget is unchanged (same gather volume, same multicasts).

## Verifier audit (2026-11-03)

- **Currency**: the five rows match the declared CBs (index, page size, format, capacity expression) in
  `nomic_qkv_rotary_heads_program_descriptor.py`. Fixed: the `cb_rotary_table` expression now names
  `TABLE_TILES_PER_ROW` (host constant, passed to the kernels as `table_block_tiles`); the ones-tile mechanism wording.
- **Capacity vs live set (`cb_heads_out`, open)**: the capacity is one whole M block (`OUT_DEPTH_BLOCKS = 1`), justified
  by "compute never waits on the writer". Under `TileRowMajor` compute reserves and pushes one tile row at a time, and
  the writer is already parked on block m's first row before compute reaches block m's last K block (it re-pushes
  the resident K blocks 0..3 of block m+1 first, which only needs block m's K block 0 popped). The steady-state live set
  is therefore about one to two tile rows, not `block_m_tiles`. A row-window capacity (`OUT_DEPTH_ROWS * LANE_N_TILES`)
  would free 16 KiB a core at `block_m_tiles = 3`. Alone, that does not admit `block_m_tiles = 4` (672 KiB against
  640 KiB). Folded into the block-size co-tune of op_requirements.md Refinement 4, which must measure it.
- **Page format vs DEST width**: `fp32_dest_acc_en` is required. `cb_partials` Float32 (packer L1 accumulation needs
  the DEST format). `cb_rotary_table` Float32: a dataflow payload copied into fp32 DEST for the rotary (precision of
  C/S). `cb_x_block`, `cb_weights`, `cb_heads_out` Float16_b (input and output ABI). No finding.
- **Disjoint lifetimes**: every pair has a recorded reason. No finding.
- **Bounds**: every symbol is bounded; the total is closed-form in `block_m_tiles`. No op dimension in a capacity.

- Refinement 4: no CB change (capacities, formats and order as above). `cb_heads_out` is now pushed per head by the
  q/k finish call (`OutputCBLayout::SubblockMajor`, identical layout at h = 1) and popped by the writer in
  `OUT_DRAIN_TILES` steps (default one head), so its steady-state live set is a few heads, not a block. The capacity
  stays one M block: the row-window reclaim (16 KiB at `block_m_tiles = 3`) frees nothing usable on its own.
  `BLOCK_M_TILES_MAX = 4` measured −0.9 µs at 8x512 with the budget lifted, but needs 704 → ≤ 640 KiB: a 1-row
  `cb_heads_out` (−48 KiB at 4 rows) plus a per-row C/S ring (−24 KiB). Recorded as a finding, not built. Data-movement
  budget unchanged (same reads, multicasts and writes; only the table request point and drain granularity moved).
