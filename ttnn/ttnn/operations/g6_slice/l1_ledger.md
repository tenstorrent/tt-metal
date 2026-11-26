# L1 Ledger: g6_slice

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model.
Block axes: the four output axes `lead_0, lead_1, out_tile_row, out_tile_col` fold into the
**flattened output-tile axis** (`flat`), block extent `block_tiles`; R2 adds the per-tile source
footprint `src` (≤ `src_per_out_cap` tiles).

There is no compute kernel and no DEST: every page is a raw-byte copy in the input dtype, so the
page-format audit (DEST width) does not apply — formats are the input dtype by ABI (bit-exact copy).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_output_tiles` | `cb_output_depth · block_tiles` (Phase 0: `2 · block_tiles`) | `2 · block_tiles` (block k draining in the writer while block k+1 fills in the reader) | `{lead_0: streams, lead_1: streams, out_tile_row: streams, out_tile_col: streams, flat: spans → block_tiles, src: streams (R2: filled from staging, one output tile per source footprint)}` | input dtype (bf16 Phase 0), page = `input_tensor.buffer_page_size()` | reader | writer | whole kernel | The only cross-RISC buffer. No input CB exists: R1 reads x pages straight into this CB (pack-into-destination, Rule 3.1); capacity above one block is the double-buffer (depth) decision. |
| `cb_source_tiles` (R2 only; absent in R1) | `cb_source_depth · src_per_out_cap · block_tiles` (Phase 0: `1 · src_per_out_cap · block_tiles`) | `src_per_out_cap · block_tiles` (the whole block's source tiles are live from the read barrier until the block is assembled) | `{lead_0: streams, lead_1: streams, out_tile_row: streams, out_tile_col: streams, flat: spans → block_tiles, src: spans → src_per_out_cap}` | input dtype | reader | reader (scratch, no push) | `load_block` → end of `assemble_block`, every block | Cannot alias `cb_output_tiles`: concurrent lifetime (assemble reads staging while writing output slots, and the writer drains the other output half). Cannot be avoided: DRAM read alignment forbids reading sub-tile row runs directly into output positions (Mechanism caps). Transform-in-place impossible: output tile ≠ any single source tile. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `page_size` | 2048 B (bf16 tile) in Phase 0; ≤ 4096 B (fp32/int32 tile) in TARGET | dtype ∈ TARGET, TILE layout |
| `block_tiles` | `1 ≤ block_tiles ≤ min(core_tiles_max, block_tiles_l1_max)` | host computes `block_tiles = max(1, min(core_tiles_max, block_tiles_l1_max))` |
| `block_tiles_l1_max` | `floor((L1_BUDGET_BYTES − out_l1_bytes_per_core) / ((cb_output_depth + cb_source_depth·src_per_out_staging) · page_size))`; bf16 Phase 0: 329 (R1), 164 (R2 rows), 109 (R2 rows+cols) | definition |
| `L1_BUDGET_BYTES` | `min(1347776, device allocatable CB bytes per core)` | BB-05 |
| `out_l1_bytes_per_core` | 0 (Phase 0, out in DRAM); R4: `ceil(out_pages / num_l1_banks) · page_size`; R5: shard bytes (16384 B production) | memory_config |
| `core_tiles_max` | `ceil(total_out_tiles / num_cores_used)` | `split_work_to_cores` |
| `src_per_out_cap` | 1, 2 or 4 = `(1+[o_h≠0])·(1+[o_w≠0])`; `src_per_out_staging = 0` in R1, `= src_per_out_cap` in R2 | regime predicate (op_design Regimes) |
| `cb_output_depth`, `cb_source_depth` | 2, 1 (Phase 0) | host constants |

## Total per-core footprint

```
L1_op = page_size · block_tiles · (cb_output_depth + cb_source_depth · src_per_out_staging)
      ≤ L1_BUDGET_BYTES − out_l1_bytes_per_core        (by construction of block_tiles_l1_max)
```

Scales with: `block_tiles` (both terms), `cb_output_depth` (output term), `src_per_out_cap` (R2 staging term only). No term scales with any tensor dimension except through the bounded `block_tiles ≤ core_tiles_max`. Phase 0 worst case: R1 bf16 `2048 · 329 · 2 = 1347584 B`; R2 rows `2048 · 164 · 4 = 1343488 B`; R2 rows+cols `2048 · 109 · 6 = 1339392 B` — all ≤ 1347776 B. Every golden INPUTS shape fits (the footprint is independent of the shape beyond the cap).

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | R1: the selected tiles once (x_sel). R2 production (77 cases, `M_{r-2}=1`): x_sel once. R2 general: ≤ `src_per_out_cap` · x_sel (shared source tiles re-read per output tile) | every output tile reads only its own source page(s), straight into L1; no tile row is read whole (rule 5) | none |
| out | once | written from `cb_output_tiles` | none |
| starts (device form, R7) | one 4·r-byte page per working core | every core needs the begin before its first page id | none (R8: 1 read + one 4 B multicast) |

Totals, perf-focus case p1 `(16,4096,8704) → (16,4096,4352)` bf16: DRAM read 570425344 B, DRAM write 570425344 B, cross-core 0 B, L1-local copies 0 B (R1). Row-shift flagged case 587 `(16,4096,1280)` row 4095: DRAM read 1310720 B (640 tiles), write 1310720 B, L1-local assemble 1310720 B.

> Cheapest-traffic split considered: flattened contiguous output-tile split (R1/R2) — 0 B delta against the minimum on every golden INPUTS case. Implemented: the same split. The implemented split is also the cheapest. For shifted shapes outside INPUTS (multi-row row shifts, column shifts) the cheaper split is R3 (rectangular blocks with source reuse), saving up to `(src_per_out_cap − 1) · x_sel` DRAM read bytes and adding nothing; deferred because no INPUTS case re-reads a byte under R2, and R2's staging CB and slot addressing stay intact under R3 (only the block shape changes).
