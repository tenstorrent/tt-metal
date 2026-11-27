# L1 Ledger: g6_to_layout (Phase 0, regime R1 `tilerow_colgroup_untilize`)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model: `tile_row` (extent `block_height_tiles`, folds `n` and `ht`), `wt` (extent `block_width_tiles`), `row_in_tile` (fixed 32), `col_in_tile` (fixed 32). DEST is 16-bit (`fp32_dest_acc_en = false`); no CB passes a value through FP arithmetic.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_tiles_in` | `TILES_IN_DEPTH · block_height_tiles · block_width_tiles` (page = 2048 B) | `block_height_tiles · block_width_tiles` written by the reader while compute holds the previous block's last tile-row → peak `2 · block_height_tiles · block_width_tiles` at depth 2 | `{tile_row: spans → block_height_tiles, wt: spans → block_width_tiles, row_in_tile: spans → 32 (whole tile page), col_in_tile: spans → 32 (whole tile page), blocks: streams → TILES_IN_DEPTH}` | UInt16 (bf16 bytes as opaque 16-bit; a dataflow payload that goes through the untilize datacopy only — UInt16 is the format that keeps the MOVA2D move bit-exact, `ckernel_defs.h:19-31`) | reader | compute | whole kernel (every block) | Not shared with `cb_rows_out`: both are live for the whole kernel at once (reader fills block k+1 while the writer drains block k), and untilize cannot run in place (`untilize_helpers.inl` static_assert `input_dfb != output_dfb`). Capacity = 2× the block for the reader/compute overlap (double buffering). |
| `cb_rows_out` | `ROWS_OUT_DEPTH · block_height_tiles · block_width_tiles` (page = 2048 B, tile-sized pages emitted by `pack_untilize`) | one block being drained by the writer + the next block being packed → peak `2 · block_height_tiles · block_width_tiles` at depth 2 | `{tile_row: spans → block_height_tiles, wt: spans → block_width_tiles (row stride block_width_tiles·64 B), row_in_tile: spans → 32, col_in_tile: spans → 32, blocks: streams → ROWS_OUT_DEPTH}` | UInt16 (same bytes, packed row-major) | compute | writer | whole kernel (every block) | Not shared with `cb_tiles_in`: concurrent lifetime (see above). Not replaceable by writing straight from `cb_tiles_in`: a row of a W > 16 block is `2·block_cols` non-contiguous 32-B face rows there, i.e. `2·block_cols` NoC writes per row instead of 1 (lamp L6 covers the W ≤ 16 case, where the row is contiguous). Capacity = 2× the block for the compute/writer overlap. |

Pattern check (Rule 3): pack-into-destination is not available (the destination is a DRAM/L1 tensor written by NoC, not a CB); transform-in-place is forbidden by the helper; alias is blocked by concurrent lifetimes; fold-into-accumulator does not apply (no arithmetic). Two CBs is the minimum inventory for a compute-side reorder.

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `tile_bytes` | bytes of one 32×32 UInt16/bf16 tile page | 2048 | format |
| `TILES_IN_DEPTH`, `ROWS_OUT_DEPTH` | buffer depths | 2 each (Phase 0) | module constants |
| `L1_OP_BUDGET_BYTES` | L1 per core the model leaves to the op (BB-05) | 1347776 | spec |
| `out_l1_bytes_per_bank` | this core's share of an L1 output | `ceil(N·H / num_l1_banks) · round_up(W·2, l1_alignment)` if the output is L1, else 0 | output placement |
| `block_tiles_cap` | max tiles per block | `floor((L1_OP_BUDGET_BYTES − out_l1_bytes_per_bank) / ((TILES_IN_DEPTH + ROWS_OUT_DEPTH) · tile_bytes))`; 164 for a DRAM output at depth 2 | must be ≥ 1, else host `RuntimeError` (no golden shape reaches it) |
| `col_group_width_tiles` | tiles of one column group | ≤ `tensor_tile_cols` | occupancy rule in `op_design.md` → Work Distribution |
| `block_width_tiles` | block extent on `wt` | `≤ min(col_group_width_tiles, block_tiles_cap)` | `plan()` |
| `core_rows_max` | most tile-rows of one column group a core holds | `≤ min(ceil(num_units / num_cores), tensor_tile_rows)` | `split_work_to_cores` |
| `block_height_tiles` | block extent on `tile_row` | `≤ min(core_rows_max, floor(block_tiles_cap / block_width_tiles))` | `plan()` |

## Total per-core footprint

```
L1_op(core) = (TILES_IN_DEPTH + ROWS_OUT_DEPTH) · block_height_tiles · block_width_tiles · tile_bytes
            + out_l1_bytes_per_bank
            ≤ L1_OP_BUDGET_BYTES = 1347776
```

Both CB terms scale with `block_height_tiles · block_width_tiles` (the block knobs) and linearly with their depth knob; `out_l1_bytes_per_bank` scales with the output size and is zero for a DRAM output; nothing scales with `N`, `H` or `W` except through the bounded block knobs. Example: `[16,4096,2560]` → `4 · 2 · 80 · 2048 = 1310720` B; `[1,2048,2048,3]` → `4 · 145 · 1 · 2048 = 1187840` B; `[1,32,2560]` in L1 → `4 · 1 · 1 · 2048 + 5120 = 13312` B.

## Data-movement budget

Boundary: the DRAM (or L1-bank) side of the interleaved tensors. "Live bursts" = the 32-B face rows that hold at least one logical value.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (TILE) | 1 × live bursts; dead faces and dead rows 0 | each tile belongs to exactly one block of exactly one core; live-extent reads skip faces 1/3 when `live_cols ≤ 16` and faces 2/3 (or the dead row suffix of faces 0/1) when `live_rows ≤ 16` | 0 |
| output (ROW_MAJOR) | 1 × live bytes | each row segment is produced in exactly one block and written once; `num_col_groups` segments per row | 0 |

Totals on the flagged cases:

| Case | Input read | Output written | Write transactions |
|------|-----------|----------------|--------------------|
| 308 `[1,2048,2048,3]` | 131072 tiles × 1024 B (faces 0, 2) = 128 MiB | 4194304 rows × 6 B = 24 MiB | 4194304 |
| 304 `[1,1024,2048,128]` | 262144 × 2048 B = 512 MiB | 512 MiB | 2097152 (256 B each) |
| 286 `[1,1024,1024,3]` | 32768 × 1024 B = 32 MiB | 1048576 × 6 B = 6 MiB | 1048576 |
| 1978 `[16,4096,2560]` | 163840 × 2048 B = 320 MiB | 320 MiB | 65536 (5120 B each) |

Note on case 308: a face row is one 32-B DRAM burst and holds 6 live bytes, and every 6-B output row is its own transaction into its own 64-B page slot, so the burst-granular floor is 128 MiB read + 4194304 write bursts. The lead's 109.23 µs roofline counts logical bytes (24 MiB each way) and sits below that floor; the perf pass measures against the burst-granular bound.

> Cheapest-traffic split considered: `tile_row` only (R4) — same bytes per tier as the implemented split; it differs only in write transactions on shapes with fewer tile-rows than cores, where it leaves cores idle. Implemented: `tile_row × column groups` (R1), which **is** the `tile_row`-only split whenever that split balances the grid within `COL_SPLIT_TOLERANCE`. The implemented split is the cheapest in bytes on every tier; nothing is deferred on traffic grounds.
