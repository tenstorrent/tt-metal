# L1 Ledger: g6_typecast (Phase 0, regime R1 `tile_stream`)

Schema: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` Blocking Model):
`tile` (flat; extent `block_tiles`), the folded `batch` / `tile_row` / `tile_col` (no extent of their own), and `bf8b_group` (inside a page).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `effective_depth · block_tiles` | `effective_depth · block_tiles` (one block landing while one is converted, when depth 2) | `{tile: spans → block_tiles · effective_depth, batch: streams (folded into tile), tile_row: streams (folded into tile), tile_col: streams (folded into tile), bf8b_group: inside one page (constant)}` | input dtype, Phase 0 `Float16_b` (2048 B). Dataflow payload in the source format; never written by the packer, so the DEST-width audit does not apply | reader | compute | whole kernel, streaming | cannot share with `cb_output_tiles`: lifetimes are concurrent (compute reads block k while writer drains block k−1 / packer fills it) and formats/page sizes differ; in-place transform impossible (src and dst page formats differ) |
| `cb_output_tiles` | `effective_depth · block_tiles` | `effective_depth · block_tiles` | `{tile: spans → block_tiles · effective_depth, batch: streams (folded into tile), tile_row: streams (folded into tile), tile_col: streams (folded into tile), bf8b_group: inside one page (constant)}` | output dtype: `Float32` (4096 B) with `fp32_dest_acc_en=True` — a 32-bit page is the **output ABI** of the float32 destination, not a widened intermediate; `Int32` (4096 B) with `fp32_dest_acc_en=True` (output ABI); `Bfp8_b` (1088 B) with 16-bit DEST | compute | writer | whole kernel, streaming | cannot share with `cb_input_tiles`: concurrent lifetime, different format |

Audit notes:
1. Capacity = live set for both CBs. Depth is `min(buffer_depth, max_blocks_per_core)`, so a core with one block never allocates an unused second slot.
2. Page format vs DEST: Float32/Int32 pages only with `fp32_dest_acc_en=True` (output ABI); Bfp8_b with 16-bit DEST. The bf16 input page never passes through the packer.
3. No disjoint lifetimes: both CBs live for the whole kernel.
4. Symbols below.

## Symbol table

| Symbol | Meaning | Bound | Established by |
|--------|---------|-------|----------------|
| `block_tiles` | tiles per block (single source: host constant → CT arg) | `1 ≤ block_tiles ≤ min(core_tiles_max, block_tiles_l1_cap)` | host clamp in the descriptor builder |
| `core_tiles_max` | max tiles on any core (`tiles_per_core_g1`) | `ceil(tensor_tiles / num_cores)` | `split_work_to_cores` |
| `buffer_depth` | depth knob | Phase 0 = 2 | host constant |
| `effective_depth` | allocated depth | `min(buffer_depth, ceil(core_tiles_max / block_tiles))` ∈ {1, 2} | host |
| `in_page_bytes`, `out_page_bytes` | page sizes | ≤ 4096 B each (fp32/int32 tile) | dtype |
| `L1_OP_BUDGET_BYTES` | model's L1 grant per core | 1 347 776 B | spec (BB-05) |
| `L1_SLACK_BYTES` | headroom for allocator alignment / kernel-side scratch | 65 536 B | host constant |
| `out_l1_bytes_per_bank` | op's own output share per L1 bank when the output is in L1 | `ceil(out_pages / num_l1_banks) · out_page_bytes` (P39: 8192 bf8b tiles → ~70–75 KB) | output memory config |
| `block_tiles_l1_cap` | `floor((L1_OP_BUDGET_BYTES − L1_SLACK_BYTES − out_l1_bytes_per_bank) / (effective_depth · (in_page_bytes + out_page_bytes)))` | ≥ 1 asserted on host (worst case depth 2 × 6144 B = 12 KB ≪ budget) | host |

## Total per-core footprint

`L1_op = effective_depth · block_tiles · (in_page_bytes + out_page_bytes) + out_l1_bytes_per_bank ≤ L1_OP_BUDGET_BYTES − L1_SLACK_BYTES`

Both CB terms scale with `block_tiles · effective_depth`; `out_l1_bytes_per_bank` scales with the tensor (L1 outputs only). Production peaks (≈120–130-core grid): P39 bf16→bf8b ≈ 70 tiles × 3136 B ≈ 220 KB + ~75 KB output share; Q2185 bf16→fp32 ≈ 16 tiles × 6144 B ≈ 100 KB.

## Data-movement budget (chosen split: flat contiguous `tile` ranges, full grid)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 (DRAM or L1 banks, per the input's buffer) | every tile is owned by exactly one core and converted from the block resident in its CB | none |
| output | 1 | packed straight into `cb_output_tiles` and written once | none |

Totals per tier: buffer tier = `tensor_tiles · (in_page_bytes + out_page_bytes)` (e.g. P39: 8192 × 3136 B ≈ 25.7 MB, L1 banks; Q2185: 2048 × 6144 B ≈ 12.6 MB, DRAM); cross-core NoC = 0; core-local L1 = one write + one read of each page per CB.

> Cheapest-traffic split considered: flat contiguous `tile` ranges across the full grid — every candidate split (row-only, column-only, bank-affine) moves the same bytes, and no split adds a combine. Implemented: the same split. The implemented split is the cheapest; bank-affine assignment is a perf lamp (fewer NoC hops, same bytes), not a traffic delta.

## Implementation notes (ttnn-implementer, Phase 0)

- CB inventory and sizing implemented exactly as above (`cb_pages = effective_depth · block_tiles`, both CBs).
- Ragged-tail synchronization: the eltwise chain caps a `PerBlockSize`/`FullBlock` block at the DEST
  capacity, so `convert_block` is one chain call per block with Upfront wait / AtEnd pop and Upfront
  reserve / AtEnd push over the block's **actual** extent. Reader, compute and writer therefore all
  reserve/push/wait/pop the actual extent of a ragged last block (ValidTiles semantics) instead of the
  nominal `block_tiles`. Capacity is unchanged; the tail block starts at offset 0 or `block_tiles`
  of the ring, so it is always contiguous.
- Data-movement budget unchanged: input ×1, output ×1, no cross-core traffic.

## Refinement 1 — regime R2 `rm_pseudo_tile_stream` (ROW_MAJOR)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` (0) | `effective_depth · block_tiles` pseudo-tiles (1024 values each, `elem · 1024` B) | same | `{row: spans → rows_per_block · effective_depth (dense, rows·W ≤ block_tiles·1024)}` | input dtype (Float16_b / UInt32 / Int32); 32-bit sources tagged `UnpackToDestFp32` (consumed only by `copy_tile`) | reader (compaction) | compute | whole kernel, streaming | not with `cb_input_staging`: staging holds aligned-stride pages while the dense block is written from it |
| `cb_output_tiles` (1) | `effective_depth · block_tiles` pseudo-tiles | same | same | output dtype (UInt8 / Int32 / UInt32) — output ABI | compute | writer (expansion) | whole kernel, streaming | not with `cb_output_staging` (copy source vs destination) |
| `cb_input_staging` (2) | `rows_per_block + ceil(64 / in_aligned)` aligned pages | one block of bank-run reads | `{row: spans → rows_per_block}` | raw aligned pages (64 B DRAM slots for F317/F322) | reader (NoC) | reader (compaction) — reader-private scratch, no push/pop | whole kernel | cannot alias the dense CBs (both live during compaction); reused per block after compaction |
| `cb_output_staging` (3) | `rows_per_block + ceil(64 / out_aligned)` aligned pages | one block of bank-run writes | `{row: spans → rows_per_block}` | raw aligned pages | writer (expansion) | writer (NoC) — writer-private scratch | whole kernel | reused per block after `noc_async_writes_flushed` |

`plan_rm_blocks`: `rows_per_block · (in_aligned + out_aligned) + effective_depth · block_tiles · (in_tile + out_tile) ≤ L1_OP_BUDGET_BYTES − L1_SLACK_BYTES − out_l1_bytes_per_bank − 128`.
F322: core_rows_max = 38131 (110 cores), rows_per_block ≈ 8.8 K, 5 blocks/core, depth 2.

Data-movement budget (R2): input ×1, output ×1 (logical bytes); physical DRAM traffic = aligned 64 B slot per
row (F322: 4 194 304 × 64 B in and out — the slot is the DRAM transfer granule, the payload is 6 B / 3 B).
One NoC transaction per bank run per block (≤ num_banks per block), not per row. No cross-core traffic.

## Refinement 3 — R2 staging depth + block size (perf)

| CB | Change | Shares with / why not |
|----|--------|-----------------------|
| `cb_input_staging` (2) | capacity `staging_depth · rows_per_block + ceil(64 / in_aligned)` aligned pages; `staging_depth = min(STAGING_DEPTH, blocks per core)` (default 2) | slot b+1 receives the bank-run DMA while slot b is compacted — the two slots are concurrently live, so they cannot alias each other or the dense CBs |
| `cb_output_staging` (3) | capacity `staging_depth · rows_per_block + ceil(64 / out_aligned)` aligned pages | slot b is expanded while slot b−1's writes leave L1; the slot is reused after the flush that follows the next expansion |
| `cb_input_tiles` / `cb_output_tiles` | unchanged formula; `block_tiles` now ≤ `RM_BLOCK_TILES` (2) on R2 | unchanged |

`plan_rm_blocks`: `rows_per_block · STAGING_DEPTH · (in_aligned + out_aligned) + effective_depth · block_tiles · (in_tile + out_tile) ≤ L1_OP_BUDGET_BYTES − L1_SLACK_BYTES − out_l1_bytes_per_bank − 128`.
F322 / F317 (bf16 → u8, 6 B / 3 B rows, 64 B slots): `rows_per_block = 682` (2 pseudo-tiles), staging 2 × 682 × 128 B ≈ 175 KB,
dense CBs 2 × 2 × 3 KB = 12 KB per core. The L1 footprint fell (old single 8.8 K-row staging ≈ 1.1 MB).
Data-movement budget unchanged: input ×1, output ×1, one NoC transaction per bank run per block, no cross-core traffic.
