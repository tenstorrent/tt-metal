# L1 Ledger: linear_gelu

Schema: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`): `m` (extent `block_h`), `n` (extent `block_w`), `k` (extent `k_ext_tiles`, whole). `tile_bytes = 2048` (bf16 32x32). `fp32_dest_acc_en = True` in every supported cell.

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_activation_block` | `ACT_DEPTH * block_h * k_ext_tiles` | `2 * block_h * k_ext_tiles` at peak: the slot being computed (retained across all n-block calls) plus the slot being filled | `{m: spans -> block_h, k: spans -> k_ext_tiles, n: streams (retained, not scaled)}` | Float16_b: unpacked input (x's dtype), not packed from DEST, so DEST width does not apply | reader | compute | whole kernel; slot recycled per row-block | Capacity = live set; the 2x is the prefetch depth (`ACT_DEPTH`). Cannot alias `cb_weight_resident` or `cb_output_tiles`: all three are live concurrently in every row-block. |
| `cb_weight_resident` | `k_ext_tiles * core_n_tiles_max` | the same: the whole slice stays resident because every row-block re-reads it | `{m: streams (reused by every row-block), n: spans -> core_n_tiles_max = block_w * core_n_blocks, k: spans -> k_ext_tiles}` | Float16_b: unpacked input (W / bias dtype) | reader | compute | whole kernel; never recycled | Concurrent with every other CB, so no alias. Streaming W instead (re-delivery per row-block) would multiply the multicast traffic by `num_row_blocks` (R4-like); residency is the reason it exists at this size. Uniform size on every core is required by the multicast landing address. |
| `cb_output_tiles` | `OUT_DEPTH * block_h * block_w` (= 6 * 4 * 1 = 24 at 8x512) | `2 * sb_h * sb_w` minimum (one subblock packing, one draining); capacity holds six whole calls | `{m: spans -> block_h, n: spans -> block_w, k: neither (accumulated in DEST, never stored)}` | Float16_b: the output tensor's dtype. This is the final store, not a phase boundary, so packing bf16 from fp32 DEST is the output contract and not truncation of an accumulation. | compute | writer | whole kernel; recycled per subblock | Capacity above the minimal live set: the writer can lag up to `OUT_DEPTH - 1` calls without stalling the packer. Mechanism reason: DRAM write latency. At `block_w = 1`, depth 2 (8 pages) measured 143..146 us at 8x512 DRAM out against 138.7 us at depth 6 (the old 24 pages). Each subblock is drained as one contiguous span, so the ring must never wrap inside a subblock: the host restricts the tail subblock area to divide `block_h * block_w` (every nominal call leaves the ring at a multiple of it). No alias: live concurrently with the activation and weight CBs. |
| `cb_bias_ones_tile` | `1` (constant: one tile, column 0 = 1.0, else 0) | 1 | `{m: neither, n: neither, k: neither}` (constant) | Float16_b: must match `cb_activation_block`'s format, because it is copied into in0 position k = 24 | reader (writes once at boot) | reader (raw local-read source; no CB credit) | whole kernel | Could be written straight into every activation slot's k = 24 position. Rejected: a separate constant keeps the activation ring's pages owned by the reserve/push cycle, and costs 2 KB. Cannot share with any other CB: all are live concurrently. |

No intermediate CBs. The two phase boundaries (matmul -> bias, bias -> GELU) are folded into DEST: the bias is the 25th K tile of the same accumulation, and GELU runs in place on the DEST subblock on the packer thread before the pack (pattern 4, fold into the accumulator, and pattern 1, pack into the destination).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `k_ext_tiles` | = 25 | `validate()` refuses K != 768 (`UnsupportedAxisValue`); 768/32 + 1 |
| `core_n_tiles_max` | `block_w * ceil(ceil(96 / block_w) / n_slots)`; `n_slots` = grid_x (11 on p100a, 9 tiles), or grid_y when transposed (10, 10 tiles) | `validate()` refuses N != 3072; grid from `device.compute_with_storage_grid_size()`; orientation from `select_blocking` |
| `block_w` | = 1 (host constant `BLOCK_W`, divides 96; Refinement 3, was 3) | design constant |
| `block_h` | `1 <= block_h <= min(core_m_tiles, floor((CB_BUDGET_BYTES - fixed_bytes) / per_row_bytes))`; = 4 on p100a untransposed, 3 transposed (10-tile slice) | host solve; `RuntimeError` if the right side is < 1 (unreachable on p100a / 8x8 grids) |
| `core_m_tiles` | `ceil(tensor_m_tiles / m_slots)` (`m_slots` = grid_y, or grid_x when transposed), `tensor_m_tiles <= 4096` | `validate()`: B*S <= 4096, S <= 512 |
| `ACT_DEPTH` | = 2 | design constant |
| `OUT_DEPTH` | = 6 (Refinement 3, was 2 at `block_w = 3`) | design constant |
| `CB_BUDGET_BYTES` | = 900 * 1024 | host constant (sibling op's production-validated value with x and outputs in L1) |
| `sb_h`, `sb_w` | `sb_h * sb_w <= 4`, `sb_h \| rows`, `sb_w \| block_w`; tail area divides `block_h * block_w` | fp32 DEST half-sync capacity; host-derived, does not size any CB |

## Total per-core footprint

```
fixed_bytes   = (k_ext_tiles * core_n_tiles_max + 1) * tile_bytes                  # weight slice + ones tile
per_row_bytes = (ACT_DEPTH * k_ext_tiles + OUT_DEPTH * block_w) * tile_bytes      # scales with block_h
footprint     = fixed_bytes + block_h * per_row_bytes
```

| Term | Scales with | p100a value |
|------|-------------|-------------|
| `cb_weight_resident` | `core_n_tiles_max` (grid, orientation, block_w) | 25 * 9 * 2048 = 460,800 B |
| `cb_bias_ones_tile` | nothing | 2,048 B |
| `cb_activation_block` | `block_h`, `ACT_DEPTH` | 2 * 4 * 25 * 2048 = 409,600 B |
| `cb_output_tiles` | `block_h`, `block_w`, `OUT_DEPTH` | 6 * 4 * 1 * 2048 = 49,152 B |
| **total** | | **921,600 B = CB_BUDGET_BYTES** (block_h = 4 solves exactly at the budget; block_h = 3 gives 806,912 B). Transposed (2x512, 4x512): 25 * 10 * 2048 + 2048 + 3 * (50 + 6) * 2048 = 858,112 B |

## Data-movement budget

Built split R1 + R2 at the primary shape 8x512 (tensor_m_tiles = 128, 110 cores). Production placement: x and y in L1, W and b in DRAM.

R2 (x row multicast) was moved forward from the DRAM refinement: measured on p100a, R1's 11x unicast x reads bound the op (8x512: 306 us dataflow-only, 36 us with the x reads stubbed). With R2, `cb_activation_block` is also a multicast landing buffer, so its address must be uniform across the rectangle (one CB, same index and size on every core; all cores of a grid row push the same slot sequence).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| W (4.72 MB) | 1 | resident per core (`cb_weight_resident`); chunk j of a column is read once by that column's rotating owner (row `(j + col + 1) mod active_rows`) and multicast along the column. Owners of chunk 0 issue their read at kernel start, the rest in their round (more prefetch measured slower: the reads are DRAM-bandwidth-bound) | multicast to `active_rows - 1 = 9` receivers per byte: 42.5 MB delivered |
| bias (6 KB) | 1 | travels as the 25th K row of each W chunk | same multicast, 9 receivers |
| x (6.29 MB) | 0 (L1-resident); 1 if `buffer_type = DRAM` | one core per grid row reads each row-block (sender rotates with the row-block, offset by grid row so concurrent senders sit on different columns) and multicasts it along the row | 6.29 MB of NoC reads + multicast to `active_cols - 1 = 10` receivers per byte (62.9 MB delivered) |
| y (25.2 MB) | 0 (L1) / 1 (DRAM refinement) | written once, per subblock, as produced | 25.2 MB of NoC writes |
| ones tile | 0 | built in L1 | local reads only (`rows` per row-block) |

Totals per tier (8x512, production placement): DRAM 4.72 MB; NoC 42.5 MB W multicast delivery + 6.29 MB x reads + 62.9 MB x multicast delivery + 25.2 MB y writes; core-local: the row-block sender's ones-tile copies, plus every row-block re-unpacks the resident W slice (`num_row_blocks * 225` tiles per core, L1 to SrcA).

> Cheapest-traffic split considered and built: R1 + R2 (2D split, W column multicast and x row multicast). The planner deferred R2 for L1-resident x; measurement overturned that (see above). The `buffer_type = DRAM` refinement now needs only the gate flip for x.
