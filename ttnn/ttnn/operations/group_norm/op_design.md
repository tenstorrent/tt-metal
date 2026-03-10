# Operation Design: group_norm

## Overview
- **Operation Name**: group_norm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), batch_norm (compute_core)

## Mathematical Definition
```
Input x: (N, 1, H*W, C),  G groups,  K = H*W * C/G
For group g covering channels [g*C/G, (g+1)*C/G):
  mean_g = sum(x[..., g*C/G:(g+1)*C/G]) / K
  var_g  = sum(x[..., g*C/G:(g+1)*C/G]^2) / K  -  mean_g^2
  y[..., c] = gamma[c] * (x[...,c] - mean_g) / sqrt(var_g + eps) + beta[c]
```
Variance uses the E[x^2] - E[x]^2 formula to avoid two passes over centered data.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| num_groups | uint32_t | Yes | Divides C; C/G divisible by 32 | - | Number of groups G |
| eps | float | No | > 0 | 1e-5 | Stability constant |
| gamma | Tensor | Yes | (1,1,32,C) TILE_LAYOUT bf16 | - | Per-channel scale (host replicates single row 32x, tilizes) |
| beta | Tensor | Yes | (1,1,32,C) TILE_LAYOUT bf16 | - | Per-channel bias (same host prep as gamma) |

### Input/Output Tensor Spec
| Property | Input | Output |
|----------|-------|--------|
| Shape | (N, 1, H\*W, C) | Same |
| Layout | ROW_MAJOR | ROW_MAJOR |
| Memory | Interleaved DRAM | Interleaved DRAM |
| Dtype | bfloat16 | bfloat16 |
| Constraints | H\*W % 32 == 0, C % 32 == 0 | - |

### Derived Constants
| Symbol | Formula | Meaning |
|--------|---------|---------|
| Ht | H\*W / 32 | Tile-rows per sample |
| Ct | C / 32 | Tile-columns |
| Ct_g | Ct / G | Tile-columns per group |
| K | H\*W \* C/G | Elements per group |
| stick_size | C \* 2 | RM stick width in bytes |

### Component Sources
| Component | Source | Role | Modifications |
|-----------|--------|------|---------------|
| Reader (RM sticks) | tilize ref | input_stage | Also reads gamma/beta tiles, fills eps/scaler CBs |
| Compute (tilize) | tilize ref | input_stage | `compute_kernel_lib::tilize` helper, pushes to persistent CB |
| Compute (stats) | batch_norm ref | compute_core | Manual `reduce_tile` with indexed access for per-group column subsets |
| Compute (normalize) | batch_norm ref | compute_core | FPU sub/mul/add like batch_norm, but per-group mean/den lookup by column |
| Compute (untilize) | untilize ref | output_stage | `compute_kernel_lib::untilize` helper with NoWait mode |
| Writer (RM sticks) | untilize ref | output_stage | 32 sticks per tile-row via TensorAccessor |

### Work Distribution
- **Grid**: 1x1 (single core)
- **Per sample**: Tilize all Ht\*Ct tiles into persistent CB, compute stats + normalize, untilize

### Data Flow
```
DRAM RM sticks --[Reader: 32 sticks/tile-row]--> c_0 (cb_input_rm)
  --[Compute: tilize]--> c_1 (cb_tilized, PERSISTENT Ht*Ct tiles)
  --[Compute: stats pass (reduce_tile indexed)]--> c_6 (mean), c_7 (den) per group
  --[Compute: normalize pass (sub/mul/add per tile)]--> c_16 (cb_normalized)
  --[Compute: untilize]--> c_17 (cb_output_rm)
  --[Writer: 32 sticks/tile-row]--> DRAM RM sticks
```
Key: Input is read from DRAM once per sample. The persistent cb_tilized is read twice by compute (stats + normalize) before being popped.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Pages | Lifetime |
|-------|------|---------|-------|----------|
| c_0 | cb_input_rm | RM sticks packed as tile-sized pages | Ct | Per tile-row |
| c_1 | cb_tilized | Persistent tilized input | Ht\*Ct | Per sample |
| c_2 | cb_gamma | Gamma tiles (TILE_LAYOUT) | Ct | Program |
| c_3 | cb_beta | Beta tiles (TILE_LAYOUT) | Ct | Program |
| c_4 | cb_eps | Epsilon scalar broadcast tile | 1 | Program |
| c_5 | cb_scaler | 1/K scaler tile (bf16) | 1 | Program |
| c_6 | cb_mean | Group mean scalar tile | 1 | Per group |
| c_7 | cb_den | Group rsqrt(var+eps) scalar tile | 1 | Per group |
| c_16 | cb_normalized | Normalized output tiles | Ct | Per tile-row |
| c_17 | cb_output_rm | Untilized RM data | Ct | Per tile-row |
| c_24 | cb_sq_sum | E[x^2] accumulator | 1 | Per group |
| c_25 | cb_tmp | Scratch (squared tile staging) | 1 | Per tile |

All pages are tile-sized (2048 bytes bf16). Scaler CB c_5 is bf16.

### Kernel Arguments

**Reader compile-time**: stick_size, TensorAccessorArgs(input), TensorAccessorArgs(gamma), TensorAccessorArgs(beta)

**Reader runtime**: input_addr, gamma_addr, beta_addr, num_sticks(=N\*H\*W), Ct, block_width_size(=Ct\*64), gamma_num_tiles(=Ct), beta_num_tiles(=Ct), packed_eps

**Compute compile-time**: Ht, Ct, G, Ct_g(=Ct/G), N

**Writer compile-time**: cb_output_rm(=c_17), output_stick_size(=C\*2), tile_height(=32), num_tile_rows(=N\*Ht), Ct, TensorAccessorArgs(output)

**Writer runtime**: output_addr

### Hardware Constraints Checklist
- [x] cb_tilized wait: always Ht\*Ct pages
- [x] cb_input_rm wait: always Ct pages (per tilize block)
- [x] Scaler CB c_5 is bf16
- [x] DEST usage: 1 tile for reduce accumulation, 1 tile for binary ops
- [x] L1 budget: (Ht\*Ct + 4\*Ct + 5) \* 2048 must fit in L1

---

## Part 2: Kernel Implementation

### TDD Stage Plan

| # | Name | New Compute Phases | Expected Output |
|---|------|-------------------|-----------------|
| 1 | data_pipeline | tilize + untilize (0 custom) | output == input |
| 2 | group_mean_subtract | reduce_sum, mean_broadcast_sub (3) | x - group_mean |
| 3 | normalize | sq_reduce, var_rsqrt, mul_den (3) | (x-mean)/sqrt(var+eps) |
| 4 | affine | mul_gamma, add_beta (2) | full group_norm |

### Stage 1: data_pipeline
- **Kernel files**: reader, compute, writer (all new)
- **Reference**: `torch_output = torch_input.clone()`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(2,1,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **Compute flow**: For each sample: tilize Ht tile-rows (Ct tiles each) into cb_tilized. Wait for all Ht\*Ct tiles. Copy tiles row-by-row to cb_normalized (identity: just repack). Untilize each row to cb_output_rm. Pop cb_tilized.
- **CB bypass**: Stats/gamma/beta CBs allocated but unused. Reader fills cb_input_rm only. cb_normalized gets unmodified tiles.

### Stage 2: group_mean_subtract
- **Kernel files**: compute (add stats phases), reader (add scaler fill)
- **Reference**: see reference_body in TDD registration
- **Delta**: After tilize, before copy to cb_normalized, compute now: (P1) for each group g, iterate reduce_tile<SUM,REDUCE_SCALAR> over tiles at columns [g\*Ct_g, (g+1)\*Ct_g) across all Ht rows in cb_tilized using indexed access. Scaler = 1/K so output is mean directly. Pack to cb_mean. (P2) Wait cb_mean. (P3) For each tile-row, for each tile c: determine g=c/Ct_g, sub tile by cb_mean[g] (scalar broadcast), push to cb_normalized. Pop cb_mean after group done.

### Stage 3: normalize
- **Kernel files**: compute (add variance phases), reader (add eps fill)
- **Delta**: After mean computed, compute adds: (P4) For each group g, iterate over group tiles in cb_tilized -- for each tile, square via mul_tiles(tile,tile) -> pack to cb_tmp, then reduce_tile<SUM,REDUCE_SCALAR>(cb_tmp, cb_scaler) with manual accumulation via cb_sq_sum. Result is E[x^2]. (P5) Compute var = E[x^2] - mean^2. Add eps. rsqrt -> den. Pack to cb_den. (P6) During normalize pass, multiply centered tile by cb_den (scalar broadcast) instead of just subtracting mean.

### Stage 4: affine
- **Kernel files**: compute (add affine phases), reader (add gamma/beta read)
- **Delta**: After normalization multiply, compute adds: (P7) mul_tiles(normalized, cb_gamma[c]) per-channel. (P8) add_tiles(result, cb_beta[c]) per-channel. Reader now reads gamma/beta tile pages from DRAM into persistent cb_gamma/cb_beta at program start.

### Reader Kernel
Based on tilize reference reader pattern. Uses TensorAccessor with page_size=stick_size for input RM tensor.

Startup sequence:
1. Read gamma tiles (Ct pages from DRAM) into cb_gamma using TensorAccessor (tile page size)
2. Read beta tiles (Ct pages from DRAM) into cb_beta
3. Fill cb_eps: FILL_WITH_VALUE(cb_eps, eps) -- single tile, once
4. Fill cb_scaler: prepare_reduce_scaler or manual fill with 1/K as bf16 scalar tile

Main loop (per sample n):
```
for tile_row = 0..Ht-1:
    resolve 32 stick DRAM addresses
    cb_reserve_back(cb_input_rm, Ct)
    for stick = 0..31:
        noc_async_read(stick_addr, l1_addr, stick_size)  // full width C
        advance l1_addr by stick_size
    noc_async_read_barrier()
    cb_push_back(cb_input_rm, Ct)
```

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_output_rm)`

#### Phase 0: Tilize (per sample)
```cpp
compute_kernel_lib::tilize<cb_input_rm, cb_tilized>(Ct, Ht);
```
After all Ht blocks, cb_tilized holds Ht\*Ct tiles. Wait for them:
```cpp
cb_wait_front(cb_tilized, Ht * Ct);
```

#### Phase 1: Reduce Sum -> Mean (per group)
```cpp
// For each group g:
tile_regs_acquire();
reduce_init<SUM, REDUCE_SCALAR>(cb_tilized, cb_scaler, cb_mean);
cb_wait_front(cb_scaler, 1);  // scaler = 1/K, persists
for (ht = 0; ht < Ht; ++ht)
    for (c = g*Ct_g; c < (g+1)*Ct_g; ++c)
        reduce_tile<SUM, REDUCE_SCALAR>(cb_tilized, cb_scaler, ht*Ct + c, 0, 0);
reduce_uninit();
cb_reserve_back(cb_mean, 1);
tile_regs_commit(); tile_regs_wait();
pack_tile(0, cb_mean);
tile_regs_release();
cb_push_back(cb_mean, 1);
```
- cb_tilized: [Ht\*Ct tiles, indexed access, NOT popped]
- cb_scaler: [1 tile, waited once, NOT popped]
- cb_mean: [1 tile, freshly pushed, contains mean_g as scalar broadcast]

#### Phase 2: Reduce Sum-of-Squares -> E[x^2] (per group)
For each tile in the group (total Ht\*Ct_g tiles):
```cpp
// Square tile: mul_tiles(cb_tilized, cb_tilized, idx, idx, 0) -> DST[0]
// Pack to cb_tmp
// Reduce-accumulate from cb_tmp into cb_sq_sum
// Uses manual accumulator reload pattern (copy_tile from cb_sq_sum on iterations > 0)
```
After all tiles: cb_sq_sum contains E[x^2] (since scaler = 1/K).

**CB state after Phase 2:**
| CB | State |
|----|-------|
| cb_tilized | Ht\*Ct tiles, still persistent |
| cb_mean | 1 tile, waited (for Phase 5 variance calc and Phase 3 subtract) |
| cb_sq_sum | 1 tile, freshly pushed |

#### Phase 3: Compute Variance -> Den (per group)
```cpp
// var = E[x^2] - mean^2
// den = rsqrt(var + eps)
cb_wait_front(cb_sq_sum, 1);
cb_wait_front(cb_mean, 1);     // still available from Phase 1
cb_wait_front(cb_eps, 1);      // persistent
tile_regs_acquire();
// DST[0] = mean^2
mul_tiles(cb_mean, cb_mean, 0, 0, 0);
// DST[1] = E[x^2]
// Actually need to load sq_sum to DST[1] or use dest_reuse pattern
// Approach: sub_tiles(cb_sq_sum, cb_mean_sq) -> need mean^2 in a CB first
// Simpler: pack mean^2 to cb_tmp, then sub_tiles(cb_sq_sum, cb_tmp)
```
Detailed approach (kernel writer implements):
1. Compute mean^2: mul_tiles(cb_mean, cb_mean) -> pack to cb_tmp
2. Compute var: sub_tiles(cb_sq_sum, cb_tmp) -> DST = E[x^2] - mean^2
3. Add eps: add_tiles(DST, cb_eps) -- needs dest_reuse or pack/unpack
4. rsqrt_tile(DST) -> den
5. Pack to cb_den

Output: cb_den has 1 tile containing 1/sqrt(var+eps) as scalar broadcast.

#### Phase 4: Normalize Pass (per tile-row, per tile)
For each tile-row ht:
```
for c = 0..Ct-1:
    g = c / Ct_g
    // Wait for group g's mean and den (already computed, persistent per group)
    // centered = tile[ht*Ct + c] - mean_g   (scalar broadcast sub)
    // normalized = centered * den_g           (scalar broadcast mul)
    // Push to cb_normalized
```
cb_mean and cb_den for the current group must be available. Since we process groups sequentially and tiles in column order, we pop old group stats and push new ones as we cross group boundaries.

#### Phase 5: Affine (integrated into Phase 4 loop)
```
    // scaled = normalized * gamma[c]    (element-wise mul, gamma has per-channel values)
    // output = scaled + beta[c]         (element-wise add)
    // Push to cb_normalized
```
gamma[c] and beta[c] are tile indices into persistent cb_gamma and cb_beta.

#### Phase 6: Untilize (per tile-row)
```cpp
compute_kernel_lib::untilize<Ct, cb_normalized, cb_output_rm,
    untilize_config::InitUninitMode::InitAndUninit,
    untilize_config::WaitMode::NoWait>(1);  // 1 block (tile-row)
```
NoWait because cb_normalized was just filled by the normalize phase.

After all tile-rows, pop cb_tilized:
```cpp
cb_pop_front(cb_tilized, Ht * Ct);
```

### Writer Kernel
Based on untilize reference writer pattern. For each tile-row:
1. Pre-compute 32 DRAM stick addresses via TensorAccessor
2. cb_wait_front(cb_output_rm, Ct)
3. For each of 32 rows: noc_async_write(l1_addr, noc_addr, output_stick_size)
4. noc_async_write_barrier()
5. cb_pop_front(cb_output_rm, Ct)

### Critical Notes
1. **cb_tilized persistence**: The persistent CB holding Ht\*Ct tiles is the core design enabler. It is waited once after tilize, read by index during stats and normalize passes, and popped only after the normalize pass completes for the sample. This avoids re-reading input from DRAM.
2. **Reduce with indexed access**: reduce_tile is called with explicit tile_idx into cb_tilized. The CB is not popped between reduce_tile calls. The scaler CB (1/K) is persistent and not popped during the stats pass.
3. **Group boundary handling**: During the normalize pass, as the column index c crosses a group boundary (c % Ct_g == 0), the compute kernel must pop the old group's mean/den and wait for the new group's stats. Since stats are computed for all groups before normalization begins, the compute kernel stores them by computing all G groups' stats first, then normalizing.
4. **Stats storage**: With G groups, we need G mean tiles and G den tiles. Approach: compute and store in cb_mean/cb_den sequentially. During normalize, recompute or store in an array. For v1 with small G: store all G stats in a single large CB or recompute per-group stats during normalize by re-running the reduce pass. Alternative: use G pairs of small CBs (not scalable). The kernel writer should store stats in L1 directly (outside CBs) or recompute during normalize. Recomputation is simpler for v1.
5. **L1 budget**: Total CB memory = (Ht\*Ct + 4\*Ct + 5) \* 2048. For (2,1,64,64) G=2: Ht=2, Ct=2 -> (4+8+5)\*2048 = 34KB. Fits easily. For larger inputs, the program factory should validate L1 fits.
6. **Init/uninit management**: The compute kernel uses tilize, untilize, reduce, and binary ops. Each requires init/uninit. Use explicit init/uninit calls (not the helper's built-in ones where necessary) to switch between operation types within the kernel.

### Implementation Checklist
- [ ] Reader: RM stick reading (tilize pattern), gamma/beta tile reading, eps/scaler fill
- [ ] Compute: tilize (helper), stats (manual reduce_tile), normalize (manual FPU), untilize (helper)
- [ ] Writer: RM stick writing (untilize pattern)
- [ ] CB push/pop balance: cb_tilized waited once, popped once per sample; all other CBs balanced per phase
