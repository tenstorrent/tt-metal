# L1 Ledger: qkv_proj_heads_rope

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking model: `op_design.md`.
Block axes: `token_rows` (extent `block_m`), `head_cols` (extent `N_c = 2*h_c`), `head_dim_tile` (fixed 2, inside `head_cols`), `qkv_third` (1 per core), `hidden_k` (K' = 25, extent `k_block`).
Values in brackets: 8x512 Phase 0 (`block_m = 11`, `h_c = 4`, `N_c = 8`, `k_block = 5`).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| cb_activation_block | `act_depth * block_m * k_block` [110] | `block_m * k_block` (one K-block being multiplied) [55] | `{token_rows: spans -> block_m, head_cols: none (x does not vary along it), head_dim_tile: none, qkv_third: none, hidden_k: streams -> k_block window}` | Float16_b (bf16 input tiles; a matmul source, not a DEST spill) | reader | compute | K-loop of each block | Capacity = 2x live set for double buffering (`act_depth`). Cannot alias cb_weight_block (concurrent: both operands of every K-block) nor cb_matmul_partials (concurrent across the K-loop; different format). Could alias cb_rotary_table / cb_out_tiles (epilogue-only lifetime) but the reader prefetches table rows during the last K-blocks, which makes them concurrent: pipelining decision. |
| cb_weight_block | `weight_depth * k_block * N_c` [80] | `k_block * N_c` [40] | `{token_rows: none (W does not vary along it), head_cols: spans -> N_c, head_dim_tile: spans (inside N_c), qkv_third: none, hidden_k: streams -> k_block window}` | Float16_b | writer (BRISC W-reader) | compute | K-loop of each block | 2x live set for double buffering. Concurrent with cb_activation_block and cb_matmul_partials (same K-loop). The bias tiles ride in K-block 4 row 4 of this CB, so no separate bias CB exists. |
| cb_matmul_partials | `block_m * N_c` [88] | `block_m * N_c` [88] (the whole block accumulates in place across all 5 K-blocks, then the epilogue reads it) | `{token_rows: spans -> block_m, head_cols: spans -> N_c, head_dim_tile: spans (rotary reads both tiles of a head), qkv_third: none, hidden_k: none (accumulated away)}` | Float32 (fp32_dest_acc_en = True is the only supported value; partials carry the fp32 L1 accumulation across K-blocks - a 16-bit page would truncate the reduction at every K-block boundary) | compute (pack) | compute (unpack) | K-block 0 of a block through the end of that block's epilogue | Capacity == live set (no gap). Is its own L1-acc target and the matmul helper's `interm` placeholder (`out_buf == interm_buf`, legal: `matmul_block_helpers.hpp:307-314`). Cannot alias cb_out_tiles (concurrent during the epilogue; different format) nor the K-block CBs (concurrent). Pack-into-destination considered: the final destination is a bf16 head-layout tensor that needs rotary first, so the fp32 block is the irreducible hand-off. |
| cb_rotary_table | `table_depth * 4` [8] | 4 (one token row's `[cos_lo, -sin_lo, sin_hi, cos_hi]`) | `{token_rows: streams -> 1 row window, head_cols: none (shared by all h_c heads of the row), head_dim_tile: spans (both d-tiles of the table row), qkv_third: none, hidden_k: none}` | Float16_b (bf16 table tiles, FPU srcB operand) | reader | compute | epilogue (q/k cores; allocated but unused on v cores) | Capacity = 2x live set: reader stays one row ahead of rotary. Constant 4 per row (`TILES_PER_HEAD * 2` tables) - a genuine constant. Concurrent with cb_matmul_partials and cb_out_tiles; prefetch overlaps the K-loop tail (see cb_activation_block). |
| cb_out_tiles | `out_depth * N_c` [16] | `N_c` [8] (one token row of the core's heads) | `{token_rows: streams -> 1 row window, head_cols: spans -> N_c, head_dim_tile: spans (inside N_c), qkv_third: none, hidden_k: none}` | Float16_b (bf16 output; pack rounds fp32 DEST once) | compute | writer | epilogue | Capacity = 2x live set: writer drains row i while compute packs row i+1. Pack-into-destination is not available: q/k/v are interleaved tensors whose head-layout pages are scattered across banks, so a row is staged here for the NoC write. Cannot alias cb_matmul_partials (concurrent, different format). |

Inventory rule check (Rule 3, in order): (1) pack into destination - the epilogue packs straight into the writer's staging CB, no scratch; the matmul packs straight into its accumulator. (2) transform in place - bias is folded into the matmul (no bias pass), rotary and convert read partials and write the output CB once (no in-place pass, so no cross-phase credit hazard). (3) alias disjoint lifetimes - no two CBs have disjoint lifetimes once the table prefetch overlaps the K-loop. (4) fold into the accumulator - bias folded into the K-reduction; rotary's two products folded into DEST (`DestAccumulation::PerRow`), no product CB. No scratch CB exists for the ones tile (synthesized inside cb_activation_block) or the sin sign flip (in cb_rotary_table).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `h_c` (heads per core) | `{1,2,3,4,6,12}` | divisors of 12 (one q/k/v third per core); selection function in `op_design.md` |
| `N_c = 2*h_c` | `<= 24` | `TILES_PER_HEAD = 2` (D = 64) |
| `k_block` | `5` | `k_block | K' = 25` and neither 1 nor 25 |
| `block_m` | `1 <= block_m <= block_m_cap(N_c)` | `block_m = min(ceil(Mt/grid_m), block_m_cap(N_c))` |
| `block_m_cap(N_c)` | `floor((cb_budget_bytes - 2048*(weight_depth*k_block*N_c + table_depth*4 + out_depth*N_c)) / (2048*act_depth*k_block + 4096*N_c))` -> 29 / 21 / 16 / 13 / 8 / 2 for `N_c` = 2 / 4 / 6 / 8 / 12 / 24 | host constant `cb_budget_bytes = 921600` (900 KiB) |
| `act_depth`, `weight_depth`, `table_depth`, `out_depth` | `2` | Phase 0 double buffering |
| `Mt = B*ceil(S/32)` | `<= 128` (tile_aligned), `<= 4096` (any alignment) | `S % 32 == 0` and `B*S <= 4096`; `Mt` enters no capacity, only `block_m` via the `min` |
| `cb_budget_bytes` | `921600` | must leave room for L1-resident x + q/k/v + tables (~230 KiB/core at 8x512 on 110 banks) and the firmware region within 1.5 MiB L1 |

## Footprint

```
footprint_bytes = 2048 * ( act_depth*block_m*k_block      # cb_activation_block  (scales with block_m, k_block)
                         + weight_depth*k_block*N_c       # cb_weight_block      (scales with N_c, k_block)
                         + table_depth*4                  # cb_rotary_table      (constant)
                         + out_depth*N_c )                # cb_out_tiles         (scales with N_c)
                + 4096 * block_m*N_c                      # cb_matmul_partials   (scales with block_m x N_c)
```

| Shape class | block_m | N_c | Footprint |
|-------------|---------|-----|-----------|
| Mt = 128 (8x512 primary) | 11 | 8 | 798,720 B (780 KiB) |
| Mt = 96 (8x384) | 8 | 8 | 638,976 B (624 KiB) |
| Mt = 64 (4x512, 8x256) | 11 | 4 | 520,192 B (508 KiB) |
| Mt = 32 (2x512, 4x256) | 11 | 2 | 380,928 B (372 KiB) |

By construction `footprint_bytes <= cb_budget_bytes` for every `(block_m, N_c)` the selection function emits.

## Data-movement budget

Chosen split: 2D (token_rows x head_cols), regime R1 (unicast). Counted at 8x512 (grid 12 x 9, 108 cores; x / cos / sin / q / k / v in L1, W / bias in DRAM).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (6.29 MB) | 0 | L1-resident (Phase 0 `buffer_type = L1`); with `buffer_type = DRAM` it becomes `grid_n` = 9 crossings until R3 | `grid_n` = 9 unicast re-reads from L1 banks = 56.6 MB (x is reuse-shared across N-groups) |
| W (3.54 MB) | `grid_m * num_blocks_this_core` = 12 (42.5 MB) | reuse-shared across the 12 M-groups, unicast per core; read once per core because the whole per-core row range is one block | 0 (direct DRAM -> core) |
| bias (0.15 MB, 72 full tiles) | 12 (1.77 MB) | rides in the W stream (K tile 24) | 0 |
| cos, sin (0.13 MB used) | 0 | L1-resident (`table_buffer_type = L1`) | each q/k core reads 4 tiles per valid row: 72 cores x 11 rows x 4 x 2 KiB = 6.5 MB vs a 0.13 MB minimum (reuse-shared across q/k N-groups and images; small in absolute terms) |
| q, k, v (3 x 6.29 MB) | 0 (L1 outputs) / 1 each (DRAM outputs) | written once, by the core that computed them | 18.9 MB written once |
| partials | 0 | core-local | none; 5 Float32 L1-acc packs per output tile stay in the core's own L1 (88 x 4 KiB x 5 = 1.8 MB per core, local) |

Totals per tier at 8x512: DRAM reads 44.3 MB (W + bias); NoC L1<->L1 82.0 MB (x 56.6 + tables 6.5 + outputs 18.9); core-local L1 1.8 MB/core of L1-acc traffic.

> Cheapest-traffic split considered: 2D + W multicast + x multicast (R1+R2+R3) - DRAM 3.7 MB instead of 44.3 MB (-40.6 MB), x NoC 6.3 MB delivered by multicast instead of 56.6 MB unicast (-50.3 MB), adding two multicast channels (one W K-block of `k_block*N_c` tiles per N-group per round, one x K-block of `block_m*k_block` tiles per M-group per round). Implemented: R1, 2D unicast. Deferred because R1 is the stepping stone both multicasts layer onto without changing the compute kernel, the CB contracts (the multicast payload is exactly one cb_weight_block / cb_activation_block K-block) or the selection function; logical core order already makes each N-group's receivers consecutive for `ttnn.Mcast` (`op_design.md`, Dataflow Strategy).
