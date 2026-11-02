# L1 Ledger: linear_gelu

Schema: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`): `m` (extent `block_h`), `n` (extent `block_w`), `k` (extent `k_ext_tiles`, whole). `tile_bytes = 2048` (bf16 32x32). `fp32_dest_acc_en = True` in every supported cell.

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_activation_block` | `ACT_DEPTH * block_h * k_ext_tiles` | `2 * block_h * k_ext_tiles` at peak: the slot being computed (retained across all n-block calls) plus the slot being filled | `{m: spans -> block_h, k: spans -> k_ext_tiles, n: streams (retained, not scaled)}` | Float16_b: unpacked input (x's dtype), not packed from DEST, so DEST width does not apply | reader | compute | whole kernel; slot recycled per row-block | Capacity = live set; the 2x is the prefetch depth (`ACT_DEPTH`). Cannot alias `cb_weight_resident` or `cb_output_tiles`: all three are live concurrently in every row-block. |
| `cb_weight_resident` | `k_ext_tiles * core_n_tiles_max` | the same: the whole slice stays resident because every row-block re-reads it | `{m: streams (reused by every row-block), n: spans -> core_n_tiles_max = block_w * core_n_blocks, k: spans -> k_ext_tiles}` | Float16_b: unpacked input (W / bias dtype) | reader | compute | whole kernel; never recycled | Concurrent with every other CB, so no alias. Streaming W instead (re-delivery per row-block) would multiply the multicast traffic by `num_row_blocks` (R4-like); residency is the reason it exists at this size. Uniform size on every core is required by the multicast landing address. |
| `cb_output_tiles` | `OUT_DEPTH * block_h * block_w` | `2 * sb_h * sb_w` minimum (one subblock packing, one draining); capacity holds two whole calls | `{m: spans -> block_h, n: spans -> block_w, k: neither (accumulated in DEST, never stored)}` | Float16_b: the output tensor's dtype. This is the final store, not a phase boundary, so packing bf16 from fp32 DEST is the output contract and not truncation of an accumulation. | compute | writer | whole kernel; recycled per subblock | Capacity above the minimal live set: sized to a whole call per buffer so the writer can lag up to one call (`block_h x block_w`) without stalling the packer. Mechanism reason: double buffering at call granularity. No alias: live concurrently with the activation and weight CBs. |
| `cb_bias_ones_tile` | `1` (constant: one tile, column 0 = 1.0, else 0) | 1 | `{m: neither, n: neither, k: neither}` (constant) | Float16_b: must match `cb_activation_block`'s format, because it is copied into in0 position k = 24 | reader (writes once at boot) | reader (raw local-read source; no CB credit) | whole kernel | Could be written straight into every activation slot's k = 24 position. Rejected: a separate constant keeps the activation ring's pages owned by the reserve/push cycle, and costs 2 KB. Cannot share with any other CB: all are live concurrently. |

No intermediate CBs. The two phase boundaries (matmul -> bias, bias -> GELU) are folded into DEST: the bias is the 25th K tile of the same accumulation, and GELU runs in place on the DEST subblock on the packer thread before the pack (pattern 4, fold into the accumulator, and pattern 1, pack into the destination).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `k_ext_tiles` | = 25 | `validate()` refuses K != 768 (`UnsupportedAxisValue`); 768/32 + 1 |
| `core_n_tiles_max` | `block_w * ceil(ceil(96 / block_w) / grid_x)`; = 9 on p100a (grid_x = 11), 12 on an 8-wide grid | `validate()` refuses N != 3072; `grid_x` from `device.compute_with_storage_grid_size()` |
| `block_w` | = 3 (host constant `BLOCK_W`, divides 96) | design constant |
| `block_h` | `1 <= block_h <= min(core_m_tiles, floor((CB_BUDGET_BYTES - fixed_bytes) / per_row_bytes))`; = 4 on p100a | host solve; `RuntimeError` if the right side is < 1 (unreachable on p100a / 8x8 grids) |
| `core_m_tiles` | `ceil(tensor_m_tiles / grid_y)`, `tensor_m_tiles <= 4096` | `validate()`: B*S <= 4096, S <= 512 |
| `ACT_DEPTH` | = 2 | design constant |
| `OUT_DEPTH` | = 2 | design constant |
| `CB_BUDGET_BYTES` | = 900 * 1024 | host constant (sibling op's production-validated value with x and outputs in L1) |
| `sb_h`, `sb_w` | `sb_h * sb_w <= 4`, `sb_h \| rows`, `sb_w \| block_w` | fp32 DEST half-sync capacity; host-derived, does not size any CB |

## Total per-core footprint

```
fixed_bytes   = (k_ext_tiles * core_n_tiles_max + 1) * tile_bytes                  # weight slice + ones tile
per_row_bytes = (ACT_DEPTH * k_ext_tiles + OUT_DEPTH * block_w) * tile_bytes      # scales with block_h
footprint     = fixed_bytes + block_h * per_row_bytes
```

| Term | Scales with | p100a value |
|------|-------------|-------------|
| `cb_weight_resident` | `core_n_tiles_max` (grid_x, block_w) | 25 * 9 * 2048 = 460,800 B |
| `cb_bias_ones_tile` | nothing | 2,048 B |
| `cb_activation_block` | `block_h`, `ACT_DEPTH` | 2 * 4 * 25 * 2048 = 409,600 B |
| `cb_output_tiles` | `block_h`, `block_w`, `OUT_DEPTH` | 2 * 4 * 3 * 2048 = 49,152 B |
| **total** | | **921,600 B = CB_BUDGET_BYTES** (block_h = 4 solves exactly at the budget; block_h = 3 gives 806,912 B) |

## Data-movement budget

Chosen split R1 at the primary shape 8x512 (tensor_m_tiles = 128, 110 cores). Production placement: x and y in L1, W and b in DRAM.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| W (4.72 MB) | 1 | resident per core (`cb_weight_resident`), read once by each column's sender and multicast down the column | multicast to `active_rows - 1 = 9` receivers per byte: 42.5 MB delivered |
| bias (6 KB) | 1 | travels as the 25th K row of each W chunk | same multicast, 9 receivers |
| x (6.29 MB) | 0 (L1-resident); 11 if `buffer_type = DRAM` under R1 | each grid row's 11 cores read their shared token rows themselves (unicast) | 69.2 MB of NoC reads (11x) |
| y (25.2 MB) | 0 (L1) / 1 (DRAM refinement) | written once, per subblock, as produced | 25.2 MB of NoC writes |
| ones tile | 0 | built in L1 | local reads only (`rows` per row-block) |

Totals per tier (8x512, production placement): DRAM 4.72 MB; NoC 42.5 MB multicast delivery + 69.2 MB x reads + 25.2 MB y writes; core-local: the ones-tile copies, plus every row-block re-unpacks the resident W slice (`num_row_blocks * 225` tiles per core, L1 to SrcA).

> Cheapest-traffic split considered: R1 + R2 (2D split, W column multicast and x row multicast). Delta: x NoC reads 69.2 MB -> 6.29 MB read by 10 row injectors + 56.6 MB multicast delivery; with x in DRAM, x DRAM crossings 11 -> 1 (69.2 MB -> 6.29 MB). Implemented: R1. Deferred because in every Phase 0 cell x is L1-resident: R2 saves no DRAM bytes there, and it adds a per-row-block rendezvous coupling 11 cores. It is built with the `buffer_type = DRAM` refinement, where it removes 63 MB of DRAM traffic. R1 already writes x into a uniform-address CB slot per row-block, so R2 replaces only the fill source.
