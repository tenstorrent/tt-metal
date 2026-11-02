# L1 Ledger: qkv_proj_heads_rope

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking model: `op_design.md`.
Block axes: `token_rows` (extent `block_m`), `head_cols` (extent `N_c = 2*h_c`), `head_dim_tile` (fixed 2, inside `head_cols`), `qkv_third` (1 per core), `hidden_k` (K' = 25, extent `k_block`).
Values in brackets: 8x512 Phase 0 (`block_m = 11`, `h_c = 4`, `N_c = 8`, `k_block = 5`).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| cb_activation_block | `act_depth * block_m * k_block` [110] | `block_m * k_block` (one K-block being multiplied) [55] | `{token_rows: spans -> block_m, head_cols: none (x does not vary along it), head_dim_tile: none, qkv_third: none, hidden_k: streams -> k_block window}` | Float16_b (bf16 input tiles; a matmul source, not a DEST spill) | reader | compute | K-loop of each block | Capacity = 2x live set for double buffering (`act_depth`). Cannot alias cb_weight_block (concurrent: both operands of every K-block) nor cb_matmul_partials (concurrent across the K-loop; different format). Could alias cb_rotary_table / cb_out_tiles (epilogue-only lifetime) but the reader prefetches table rows during the last K-blocks, which makes them concurrent: pipelining decision. |
| cb_weight_block | `weight_depth * k_block * N_c` [80] | `k_block * N_c` [40] | `{token_rows: none (W does not vary along it), head_cols: spans -> N_c, head_dim_tile: spans (inside N_c), qkv_third: none, hidden_k: streams -> k_block window}` | Float16_b | writer (BRISC W-reader; under R2 the N-group sender's writer fills the slot on every group core by multicast) | compute | K-loop of each block | 2x live set for double buffering. Concurrent with cb_activation_block and cb_matmul_partials (same K-loop). The bias tiles ride in K-block 4 row 4 of this CB, so no separate bias CB exists. |
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

Chosen split: 2D (token_rows x head_cols), regime R2 (W multicast within each N-group, Refinement 3; R1 W unicast where R2 is gated off) + R3 (x multicast within each M-group, Refinement 1). Counted at 8x512 (grid 12 x 9, 108 cores; x / cos / sin / q / k / v in L1, W / bias in DRAM).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (6.29 MB) | 0 (L1) / 1 (DRAM) | read once per M-group by the group's sender (R3); `buffer_type = DRAM` therefore crosses DRAM once | 6.29 MB read by the 12 senders + one multicast per K-block per M-group (`rows*k_block` tiles to `grid_n - 1` = 8 receivers; 6.29 MB delivered per receiver column, 50.3 MB landed in total over multicast instead of 56.6 MB of unicast reads) |
| W (3.54 MB) | `num_blocks_this_core` = 1 under R2 (3.54 MB); `grid_m * num_blocks_this_core` = 12 under R1 (42.5 MB) | R2: one N-group core reads each W K-block and multicasts it to the other `grid_m - 1` | R2: 3.54 MB x 11 receivers delivered by multicast (one `k_block*N_c` K-block per N-group per round) |
| bias (0.15 MB, 72 full tiles) | 1 under R2 (0.15 MB); 12 under R1 | rides in the W stream (K tile 24) | rides the W multicast |
| cos, sin (0.13 MB used) | 0 | L1-resident (`table_buffer_type = L1`) | each q/k core reads 4 tiles per valid row: 72 cores x 11 rows x 4 x 2 KiB = 6.5 MB vs a 0.13 MB minimum (reuse-shared across q/k N-groups and images; small in absolute terms) |
| q, k, v (3 x 6.29 MB) | 0 (L1 outputs) / 1 each (DRAM outputs) | written once, by the core that computed them | 18.9 MB written once |
| partials | 0 | core-local | none; 5 Float32 L1-acc packs per output tile stay in the core's own L1 (88 x 4 KiB x 5 = 1.8 MB per core, local) |

Totals per tier at 8x512 (R1 + R3; R2 cuts the DRAM term to 3.7 MB): DRAM reads 44.3 MB (W + bias; +6.3 MB with x in DRAM); NoC L1<->L1 31.7 MB of unicast (x sender reads 6.3 + tables 6.5 + outputs 18.9) plus the x multicast; core-local L1 1.8 MB/core of L1-acc traffic.

> Cheapest-traffic split considered: 2D + W multicast + x multicast (R1+R2+R3) - DRAM 3.7 MB instead of 44.3 MB (-40.6 MB), x NoC 6.3 MB delivered by multicast instead of 56.6 MB unicast (-50.3 MB), adding two multicast channels (one W K-block of `k_block*N_c` tiles per N-group per round, one x K-block of `block_m*k_block` tiles per M-group per round). Implemented: R3 (x multicast, Refinement 1) and R2 (W multicast, Refinement 3, gated to layouts where every group is <= 3 rectangles and grid_m >= 4). Originally deferred because R1 is the stepping stone both multicasts layer onto without changing the compute kernel, the CB contracts (the multicast payload is exactly one cb_weight_block / cb_activation_block K-block) or the selection function. Both channels use one `ttnn.Mcast` per group on a layout chosen so every group is <= 3 rectangles (`op_design.md`, Work Distribution).

### Measured (R1 built, Blackhole p100a, default config, x/tables/outputs in L1)

| (B,S) | 8x512 | 16x256 | 32x128 | 8x384 | 4x512 | 8x256 | 2x512 | 4x256 |
|-------|-------|--------|--------|-------|-------|-------|-------|-------|
| device us | 278.6 | 278.1 | 275.2 | 252.4 | 215.4 | 212.8 | 206.0 | 204.7 |
| stock us | 226.3 | 228.5 | 246.4 | 178.8 | 123.5 | 124.1 | 75.0 | 79.8 |

Ablation at 2x512 (h_c=1, grid 3x36): stubbing all K-loop reads drops the matmul phase 185 -> 21 us (the
compute floor); stubbing only W leaves it at ~170 us; stubbing only x drops it to ~45 us. TensorAccessor
address math is not the cost (address-only run: 16.7 us). The x unicast re-read (`grid_n` x 57 MB
all-to-all over NoC0 from L1 banks) is the binding resource, ~15x slower than an uncontended read of
the same 540 KB per core. Sweeping `heads_per_core` trades x re-reads for W re-reads and never beats
stock (best: 2x512 123 us at h_c=2, 8x512 278 us at h_c=4). R1 is data-movement bound by design;
the next refinements are R3 (x multicast within each M-group) and R2 (W multicast within each N-group),
which cut x/W crossings to 1x. Per-core read stagger (both halves) is in place and cut 2x512 table-read
stalls in the epilogue (68 -> 40 us).

### Measured placement sensitivity (verifier, R1 built, 8x512, SUPPORTED widened at runtime for the probe only)

| x | q/k/v | cos/sin | device us |
|---|-------|---------|-----------|
| L1 | L1 | L1 | 279.4 |
| DRAM | L1 | L1 | 316.7 |
| L1 | DRAM | L1 | 275.0 |
| L1 | L1 | DRAM | 281.1 |
| DRAM | DRAM | DRAM | 377.9 |

At 2x512 (h_c = 1, grid 3x36): x L1 189.5 us, x DRAM 221.1 us. Output and table placement are
traffic-neutral under R1; x in DRAM adds the `grid_n` DRAM re-crossings counted above, which is why
`buffer_type = DRAM` ships with R3 (x multicast) rather than alone.

### Measured (R1 + R3 built, Refinement 1, Blackhole p100a, default config)

| (B,S) | 8x512 | 16x256 | 32x128 | 8x384 | 4x512 | 8x256 | 2x512 | 4x256 |
|-------|-------|--------|--------|-------|-------|-------|-------|-------|
| x in L1, device us | 265.4 | 267.3 | 269.8 | 252.5 | 144.7 | 146.6 | 93.4 | 96.6 |
| R1 (before), device us | 278.6 | 278.1 | 275.2 | 252.4 | 215.4 | 212.8 | 206.0 | 204.7 |

x in DRAM (q/k/v and tables in L1): 8x512 272.1 us (+2.5% vs x in L1; R1 was 316.7), 4x512 146.7 us,
2x512 91.9 us (R1 221.1). Ablation at 8x512: stubbing the W reads drops 265 -> 134 us, stubbing the
whole x path (no reads, no multicast) leaves it at 277 us (2x512: 93 -> 88 us). x is off the critical
path; the W unicast re-read (`grid_m` = 12 DRAM crossings) is the binding resource, which is R2
(Refinement 3).
