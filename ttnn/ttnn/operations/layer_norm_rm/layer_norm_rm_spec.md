# layer_norm_rm Functional Specification

## Overview
- **Operation Name**: `layer_norm_rm`
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (single-core), softmax general (W-dimension), untilize (single-core)
- **Reference Analyses**:
  - `ttnn/ttnn/operations/layer_norm_rm/agent_logs/tilize_single_core_analysis.md` (role: input_stage)
  - `ttnn/ttnn/operations/layer_norm_rm/agent_logs/softmax_general_analysis.md` (role: compute_core)
  - `ttnn/ttnn/operations/layer_norm_rm/agent_logs/untilize_single_core_analysis.md` (role: output_stage)
- **Workflow**: Generic Op (Python-based, no C++ scaffolding)

## Mathematical Definition

### Formula
```
Given input X of shape [..., H, W]:

1. Row-wise mean:       mu_i    = (1/W) * SUM_j( X[i, j] )
2. Centering:           X_hat[i,j] = X[i, j] - mu_i
3. Row-wise variance:   var_i   = (1/W) * SUM_j( X_hat[i, j]^2 )
4. Inverse std dev:     rstd_i  = 1 / sqrt(var_i + epsilon)
5. Standardization:     Y[i, j] = X_hat[i, j] * rstd_i
6. Affine transform:    Z[i, j] = gamma_j * Y[i, j] + beta_j
```

### Semantic Description
Layer normalization normalizes each row of the input tensor independently. For each row, it computes the mean and variance across all W elements, standardizes the row to zero mean and unit variance (with a small epsilon for numerical stability), then applies a learned per-element affine transformation using gamma (scale) and beta (shift) parameters. The input and output are both in row-major layout; internally the operation tilizes the input, performs all compute in tiled format, then untilizes the result back to row-major.

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | rank >= 2 | - | Input tensor in ROW_MAJOR layout |
| gamma | Tensor | Yes | last dim = W | - | Per-element scale parameter |
| beta | Tensor | Yes | last dim = W | - | Per-element shift parameter |
| epsilon | float | No | > 0 | 1e-5 | Numerical stability constant |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Device | Must be on device | "input must be on device" |
| Layout | ROW_MAJOR | "input must be in ROW_MAJOR layout" |
| Memory | DRAM, interleaved | "input must be DRAM interleaved" |
| Dtype | bfloat16 or float32 | "unsupported dtype, must be bfloat16 or float32" |
| Rank | >= 2 | "input must have rank >= 2" |
| Last dim (W) | Multiple of 32 | "last dimension must be a multiple of 32" |
| Second-to-last dim (H) | Multiple of 32 | "second-to-last dimension must be a multiple of 32" |
| Gamma shape | Last dim matches input W | "gamma last dim must match input last dim" |
| Gamma layout | ROW_MAJOR | "gamma must be in ROW_MAJOR layout" |
| Gamma dtype | Same as input | "gamma dtype must match input dtype" |
| Beta shape | Last dim matches input W | "beta last dim must match input last dim" |
| Beta layout | ROW_MAJOR | "beta must be in ROW_MAJOR layout" |
| Beta dtype | Same as input | "beta dtype must match input dtype" |

### Output Tensor Specification
- **Shape**: Same as input tensor shape
- **Dtype**: Same as input tensor dtype
- **Layout**: ROW_MAJOR
- **Memory**: DRAM, interleaved

## Component Sources

This operation is composed from three reference operations.

### Input Stage (from tilize single-core)

| Component | Source | Modifications |
|-----------|--------|---------------|
| Reader kernel pattern | tilize split-rows reader | Extended to also read gamma/beta sticks and generate scaler/epsilon tiles |
| CB c_0 (input staging) | tilize CB c_0 | Capacity = Wt tiles (entire tile-row width for WSmall pattern) |
| Compute (tilize phase) | tilize compute kernel | Use `compute_kernel_lib::tilize<c_0, c_2>()` to tilize input blocks; tilize gamma/beta similarly |

### Compute Stage (from softmax general WSmall)

| Component | Source | Modifications |
|-----------|--------|---------------|
| Row-wise SUM reduction | softmax reduce<SUM, REDUCE_ROW> | Used for both mean (scaler=1/W) and variance (scaler=1/W) computations |
| sub_bcast_cols | softmax Step 2 (x - max) | Adapted to subtract mean instead of max |
| mul_bcast_cols | softmax Step 5 (exp * 1/sum) | Adapted to multiply by rstd instead of 1/sum |
| square operation | New (from binary_op_helpers) | `compute_kernel_lib::square<>()` for (x - mean)^2 |
| rsqrt operation | New (rsqrt_tile) | Replace softmax's recip_tile with rsqrt_tile in post-reduce lambda |
| Gamma/beta application | New | `mul<NONE>` for gamma, `add<NONE>` for beta (element-wise, not broadcast) |
| Intermediate CBs | softmax c_24-c_28 pattern | Extended with additional CBs for gamma, beta, epsilon, centered, squared, rstd |

### Output Stage (from untilize single-core)

| Component | Source | Modifications |
|-----------|--------|---------------|
| Compute (untilize phase) | untilize compute | Use `compute_kernel_lib::untilize<Wt, c_out_tilized, c_16>()` to untilize final result |
| CB c_16 (output staging) | untilize CB c_16 | Capacity = Wt tiles (entire tile-row for WSmall) |
| Writer kernel pattern | untilize split-rows writer | Write untilized sticks to DRAM using TensorAccessor |

### Interface Compatibility

| Interface | Component A | Component B | Format A | Format B | Compatible? |
|-----------|------------|-------------|----------|----------|-------------|
| Reader -> Compute (tilize) | Reader fills c_0 with RM sticks | Compute tilizes c_0 | Row-major sticks in c_0 | Row-major expected by tilize_block | Yes |
| Compute (tilize) -> Compute (norm) | tilize outputs to c_2 | norm reads from c_2 | Tiled tiles in c_2 | Tiled tiles expected by reduce/sub/mul | Yes |
| Compute (norm) -> Compute (untilize) | norm outputs to c_out_tilized | untilize reads c_out_tilized | Tiled tiles | Tiled tiles expected by untilize_block | Yes |
| Compute (untilize) -> Writer | untilize outputs to c_16 as RM | Writer reads c_16 | Row-major sticks in c_16 | Row-major expected by stick writer | Yes |

### CB ID Resolution

| Logical CB | Source Ref | Original ID | Final ID | Notes |
|------------|-----------|-------------|----------|-------|
| cb_input_rm | tilize | c_0 | c_0 | Input RM sticks from DRAM |
| cb_reduce_scaler | softmax | c_2 | c_1 | Reduce scaler tile (1/W) |
| cb_input_tilized | tilize output / softmax input | c_16 / c_0 | c_2 | Tilized input tiles (Wt capacity) |
| cb_gamma_rm | New | - | c_3 | Gamma RM sticks from DRAM |
| cb_beta_rm | New | - | c_4 | Beta RM sticks from DRAM |
| cb_gamma_tilized | New | - | c_5 | Tilized gamma tiles (Wt capacity) |
| cb_beta_tilized | New | - | c_6 | Tilized beta tiles (Wt capacity) |
| cb_eps_scalar | New | - | c_7 | Epsilon scalar tile |
| cb_output_rm | untilize | c_16 | c_16 | Output RM sticks to DRAM |
| cb_mean | softmax c_26 | c_26 | c_24 | Row-wise mean (1 tile) |
| cb_centered | softmax c_27 | c_27 | c_25 | x - mean intermediate (Wt tiles) |
| cb_squared | New | - | c_26 | (x - mean)^2 intermediate (Wt tiles) |
| cb_var | New | - | c_27 | Variance + epsilon result (1 tile) |
| cb_rstd | softmax c_25 | c_25 | c_28 | 1/sqrt(var+eps) result (1 tile) |
| cb_normalized | New | - | c_29 | (x-mean)*rstd intermediate (Wt tiles) |
| cb_gamma_applied | New | - | c_30 | gamma * normalized intermediate (Wt tiles) |
| cb_out_tilized | New | - | c_31 | Final tilized output before untilize (Wt tiles) |

## Design Decisions

### Decision 1: WSmall-only (All tiles fit in L1)

- **Choice**: Only implement the WSmall variant (entire tile-row loaded at once). No WLarge streaming variant.
- **Rationale**: The spec states single-core execution with W and H both multiples of 32. For typical layer norm use cases (W <= ~4096 with bfloat16), all CBs fit in L1. The WSmall approach is simpler and avoids triple-reading from DRAM. If W is very large, the operation can be extended later.
- **Alternatives Considered**: WLarge with triple-read pattern (like softmax). Rejected for initial implementation due to complexity.
- **Tradeoffs**: Simpler implementation, better performance for typical sizes. Will not work for extremely large W dimensions that exceed L1 budget.

### Decision 2: Combined tilize/compute/untilize in single kernel set

- **Choice**: Use a single set of reader/compute/writer kernels that handle tilize, layer norm compute, and untilize phases sequentially within the compute kernel.
- **Rationale**: Since this is single-core, the reader fills input RM sticks into c_0, compute tilizes them into c_2, performs layer norm, untilizes into c_16, and writer drains c_16 to DRAM. This avoids multiple passes and is the most efficient single-core approach.
- **Alternatives Considered**: Separate tilize/compute/untilize operations chained together. Rejected because it would require writing tilized intermediates to DRAM and reading them back, which is wasteful.
- **Tradeoffs**: More complex compute kernel, but much better performance.

### Decision 3: Row-by-row processing with persistent gamma/beta

- **Choice**: Process one tile-row at a time. Reader reads gamma/beta once upfront, tilizes them once in compute. For each subsequent tile-row, only the input is re-read and re-tilized.
- **Rationale**: Gamma and beta are the same for every row (they are shape [1,...,1,W]). Reading and tilizing them once avoids redundant DRAM reads and compute. The tilized gamma/beta tiles persist in their CBs across all tile-rows.
- **Alternatives Considered**: Re-reading gamma/beta per row. Rejected as wasteful.
- **Tradeoffs**: Requires dedicated CBs for gamma/beta that persist for the entire program lifetime (using WaitUpfrontNoPop or similar pattern where they are not popped).

### Decision 4: Use reduce scaler = 1/W for mean and variance

- **Choice**: The reduce scaler for both SUM reductions (mean and variance) is set to 1/W, so `reduce<SUM, REDUCE_ROW>` directly produces the mean (or variance) without a separate division step.
- **Rationale**: The hardware reduce operation multiplies by the scaler tile automatically. Setting scaler = 1/W computes `(1/W) * SUM(x)` = mean in a single operation. This is the standard pattern used in existing operations.
- **Alternatives Considered**: Using scaler = 1.0 and dividing by W separately. Rejected as it wastes a compute step.
- **Tradeoffs**: None -- strictly better.

### Decision 5: Epsilon handled via add_bcast_scalar after variance reduction

- **Choice**: After computing variance via reduce, add epsilon using a scalar broadcast add (`add<SCALAR>` with epsilon scalar tile). Then apply rsqrt_tile.
- **Rationale**: Epsilon must be added to variance before computing rsqrt. The scalar broadcast add is the most efficient way since epsilon is a single value applied to the entire variance tile.
- **Alternatives Considered**: Packing epsilon into the reduce scaler. Not possible because the reduce scaler is 1/W. Considered adding epsilon inside the post_reduce_op lambda but the tile must first be packed to a CB before the add.
- **Tradeoffs**: Requires an additional CB for the epsilon scalar tile, but this is a single tile and trivial memory cost.

### Decision 6: Use kernel helper libraries for all compute phases

- **Choice**: Use `compute_kernel_lib::tilize<>`, `compute_kernel_lib::reduce<>`, `compute_kernel_lib::sub<COL>`, `compute_kernel_lib::mul<COL>`, `compute_kernel_lib::square<>`, `compute_kernel_lib::add<SCALAR>`, `compute_kernel_lib::mul<NONE>`, `compute_kernel_lib::add<NONE>`, and `compute_kernel_lib::untilize<>` helpers wherever possible.
- **Rationale**: These helpers manage DST registers, CB synchronization, init/uninit, and data format reconfiguration automatically. Using them reduces error risk and produces cleaner code.
- **Alternatives Considered**: Raw LLK calls. Rejected for complexity and error risk.
- **Tradeoffs**: None -- helpers are strictly easier and safer.

### Decision 7: Reduce scaler format uses generate_reduce_scaler

- **Choice**: Use `dataflow_kernel_lib::generate_reduce_scaler()` from `reduce_helpers_dataflow.hpp` to generate the 1/W scaler tile. The scaler must be packed as `(bf16 << 16 | bf16)` for bfloat16 or raw float32 bits for float32.
- **Rationale**: This is the standard helper for generating reduce scaler tiles in the correct hardware format. The MEMORY.md notes confirm that `generate_reduce_scaler()` expects `(bf16 << 16 | bf16)` format, NOT IEEE 754 float32.
- **Alternatives Considered**: Manual tile generation. Rejected as error-prone.
- **Tradeoffs**: None.

### Decision 8: Epsilon scalar format uses generate_bcast_scalar_bfloat16 / generate_bcast_scalar

- **Choice**: Use `dataflow_kernel_lib::generate_bcast_scalar_bfloat16()` (for bfloat16) or `dataflow_kernel_lib::generate_bcast_scalar()` (for float32) from `scalar_helpers.hpp` to generate the epsilon scalar tile for the `add<SCALAR>` operation.
- **Rationale**: The scalar must be placed at position [0,0] of face 0 for scalar broadcast operations. The helper functions handle this correctly.
- **Alternatives Considered**: Using `generate_bcast_col_scalar_*` which fills column 0 of faces 0 and 2. This would be needed for COL broadcast, not SCALAR broadcast. The distinction is critical.
- **Tradeoffs**: None.

### Decision 9: Compute kernel processes tile-rows in a loop, not the entire tensor at once

- **Choice**: The compute kernel outer loop iterates over tile-rows (Ht * batch). Within each tile-row, it processes all Wt tiles for tilize, norm, and untilize.
- **Rationale**: This mirrors the softmax WSmall pattern exactly. Each tile-row is independent for layer norm (normalization is per-row). Processing one tile-row at a time keeps intermediate CB usage bounded to Wt tiles, which is the WSmall assumption.
- **Alternatives Considered**: Processing all tiles at once. Not feasible due to L1 memory constraints.
- **Tradeoffs**: Natural fit for the row-wise normalization pattern.

### Decision 10: block_width_tiles for tilize/untilize = Wt (entire tile-row width)

- **Choice**: Set `num_tiles_per_block = Wt` for the tilize and untilize phases, processing the entire tile-row width in a single block.
- **Rationale**: The WSmall pattern requires all Wt tiles to be present in the CB simultaneously for the normalization compute. Therefore, the tilize and untilize must process Wt tiles at a time to match. The L1 budget must accommodate this.
- **Alternatives Considered**: Smaller block sizes for tilize/untilize. Not compatible with WSmall normalization which needs all Wt tiles simultaneously.
- **Tradeoffs**: L1 usage is higher, but this is inherent to the WSmall design.

## Work Distribution

### Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row (one horizontal strip of Wt tiles) |
| **Unit size** | Wt tiles |
| **Total units** | `num_tile_rows = (total_elements / W) / 32 = num_sticks / 32` |
| **Loop structure** | Outer loop: tile-rows (0 to num_tile_rows-1). Per tile-row: read RM sticks -> tilize -> norm -> untilize -> write RM sticks |

### Parallelization Strategy
- **Grid**: 1 x 1 (single core)
- **Work per core**: All tile-rows
- **Load balancing**: N/A (single core)

## Data Flow

### High-Level Flow

```
For each tile-row r in [0, num_tile_rows):
  1. Reader: Read 32 input sticks from DRAM -> cb_input_rm (c_0), Wt tiles worth
  2. Compute: tilize cb_input_rm (c_0) -> cb_input_tilized (c_2)
  3. Compute: reduce<SUM, REDUCE_ROW>(c_2, scaler=1/W) -> cb_mean (c_24), 1 tile
  4. Compute: sub<COL>(c_2, c_24) -> cb_centered (c_25), Wt tiles
  5. Compute: square<>(c_25) -> cb_squared (c_26), Wt tiles
  6. Compute: reduce<SUM, REDUCE_ROW>(c_26, scaler=1/W) -> cb_var (c_27), 1 tile
  7. Compute: add<SCALAR>(c_27, c_7) -> cb_var (c_27), 1 tile (var + eps)
     Then rsqrt_tile in post-op -> cb_rstd (c_28), 1 tile
  8. Compute: mul<COL>(c_25, c_28) -> cb_normalized (c_29), Wt tiles
  9. Compute: mul<NONE>(c_29, c_5) -> cb_gamma_applied (c_30), Wt tiles
  10. Compute: add<NONE>(c_30, c_6) -> cb_out_tilized (c_31), Wt tiles
  11. Compute: untilize cb_out_tilized (c_31) -> cb_output_rm (c_16)
  12. Writer: Write 32 sticks from cb_output_rm (c_16) -> DRAM

One-time setup (before the loop):
  - Reader: Generate reduce scaler (1/W) -> cb_reduce_scaler (c_1)
  - Reader: Generate epsilon scalar -> cb_eps_scalar (c_7)
  - Reader: Read gamma sticks -> cb_gamma_rm (c_3)
  - Reader: Read beta sticks -> cb_beta_rm (c_4)
  - Compute: Tilize gamma -> cb_gamma_tilized (c_5), persist
  - Compute: Tilize beta -> cb_beta_tilized (c_6), persist
```

### Kernel Data Movement

| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| Reader | RISCV_0 | NOC0 | Read input RM sticks from DRAM (split-rows pattern), read gamma/beta RM sticks once, generate reduce scaler and epsilon scalar tiles |
| Compute | RISCV_2 (TRISC) | N/A | Tilize input, compute layer norm (mean, center, variance, rsqrt, normalize, gamma/beta), untilize output |
| Writer | RISCV_1 | NOC1 | Write output RM sticks to DRAM (split-rows pattern) |

### Circular Buffer Requirements

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input_rm | Input RM sticks staging for tilize | Wt tiles | Wt tiles | Single | Reader | Compute (tilize) | Block (per tile-row) |
| c_1 | cb_reduce_scaler | Reduce scaler tile (1/W) | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_2 | cb_input_tilized | Tilized input tiles | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (norm) | Block (per tile-row) |
| c_3 | cb_gamma_rm | Gamma RM sticks staging for tilize | Wt tiles | Wt tiles | Single | Reader | Compute (tilize) | One-shot then freed |
| c_4 | cb_beta_rm | Beta RM sticks staging for tilize | Wt tiles | Wt tiles | Single | Reader | Compute (tilize) | One-shot then freed |
| c_5 | cb_gamma_tilized | Tilized gamma tiles | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (gamma mul) | Program (persistent) |
| c_6 | cb_beta_tilized | Tilized beta tiles | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (beta add) | Program (persistent) |
| c_7 | cb_eps_scalar | Epsilon scalar tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_16 | cb_output_rm | Output RM sticks for writer | Wt tiles | Wt tiles | Single | Compute (untilize) | Writer | Block (per tile-row) |
| c_24 | cb_mean | Row-wise mean (reduce output) | 1 tile | 1 tile | Single | Compute | Compute | Block (per tile-row) |
| c_25 | cb_centered | x - mean intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Block (per tile-row) |
| c_26 | cb_squared | (x - mean)^2 intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Block (per tile-row) |
| c_27 | cb_var | Variance + epsilon (before rsqrt) | 1 tile | 1 tile | Single | Compute | Compute | Block (per tile-row) |
| c_28 | cb_rstd | 1/sqrt(var + eps) | 1 tile | 1 tile | Single | Compute | Compute | Block (per tile-row) |
| c_29 | cb_normalized | (x - mean) * rstd | Wt tiles | Wt tiles | Single | Compute | Compute | Block (per tile-row) |
| c_30 | cb_gamma_applied | gamma * normalized | Wt tiles | Wt tiles | Single | Compute | Compute | Block (per tile-row) |
| c_31 | cb_out_tilized | Final tilized output (before untilize) | Wt tiles | Wt tiles | Single | Compute | Compute (untilize) | Block (per tile-row) |

**Data format notes**:
- c_0, c_3, c_4, c_16: Input data format (bfloat16 or float32) -- these hold RM sticks
- c_1, c_7: Input data format -- scalar tiles
- c_2, c_5, c_6: Input data format -- tilized data
- c_24 through c_31: Intermediate data format. Use Float32 if `fp32_dest_acc_en` is true, otherwise same as input. This matches the softmax pattern for numerical precision.

**L1 budget estimation** (bfloat16, tile_size = 2048 bytes, intermed_tile_size = 2048 or 4096 for fp32):
- Input CBs: c_0(Wt) + c_1(1) + c_2(Wt) + c_3(Wt) + c_4(Wt) + c_5(Wt) + c_6(Wt) + c_7(1) = 6*Wt + 2 tiles
- Output CB: c_16(Wt) = Wt tiles
- Intermediate CBs: c_24(1) + c_25(Wt) + c_26(Wt) + c_27(1) + c_28(1) + c_29(Wt) + c_30(Wt) + c_31(Wt) = 5*Wt + 3 tiles
- Total: 12*Wt + 5 tiles
- For W=1024 (Wt=32): 12*32+5 = 389 tiles * 2048 = ~778 KB (bfloat16). This exceeds typical L1 of 1.5MB minus allocator base (~100KB), so it fits but is tight.
- For W=2048 (Wt=64): 12*64+5 = 773 tiles * 2048 = ~1.5 MB -- too large.
- **Optimization note**: Several intermediate CBs can be overlapped if they don't coexist. See CB lifetime analysis below.

**CB Lifetime Optimization**: Within each tile-row iteration:
- c_0 is freed after tilize (step 2). c_3 and c_4 are freed after gamma/beta tilize (one-shot).
- c_2 is consumed by step 3 (mean reduce, WaitUpfrontNoPop) and step 4 (sub). Freed after step 4.
- c_25 (centered) is consumed by step 5 (square) and step 8 (mul rstd). Must persist through step 8.
- c_26 (squared) is consumed by step 6 (variance reduce). Freed after step 6.
- c_24 (mean) is consumed by step 4 (sub). Freed after step 4.
- c_27 (var) is consumed by step 7 (add eps + rsqrt). Freed after step 7.
- c_28 (rstd) is consumed by step 8. Freed after step 8.
- c_29 (normalized) is consumed by step 9. Freed after step 9.
- c_30 (gamma_applied) is consumed by step 10. Freed after step 10.
- c_31 (out_tilized) is consumed by step 11 (untilize). Freed after step 11.

**Key overlaps possible**: c_0 and c_input_tilized(c_2) could potentially share memory since c_0 is freed before c_2 is fully consumed, but the tilize helper needs both simultaneously. After tilize, c_0 is freed and c_26 could reuse its memory, etc. However, for simplicity of initial implementation, we allocate all CBs separately. The generic_op_builder will handle the actual CB allocation.

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**One-time setup**:
1. Generate reduce scaler tile (1/W) into c_1 using `dataflow_kernel_lib::generate_reduce_scaler(c_1, packed_scaler)`. Push 1 tile.
2. Generate epsilon scalar tile into c_7 using `dataflow_kernel_lib::generate_bcast_scalar_bfloat16(c_7, packed_eps)` (bfloat16) or `dataflow_kernel_lib::generate_bcast_scalar(c_7, packed_eps)` (float32). Push 1 tile.
3. Read gamma sticks: `cb_reserve_back(c_3, Wt)`, read Wt tiles worth of sticks (same split-rows pattern as input but only 1 tile-row of 32 sticks from gamma tensor), `cb_push_back(c_3, Wt)`. If gamma has fewer than 32 rows, read the available rows and zero-pad (but since W must be multiple of 32 and gamma is [1,...,1,W], gamma is effectively a single row of W elements repeated -- only need to read row 0 and replicate for 32 sticks).
4. Read beta sticks: Same pattern as gamma into c_4.

**Per tile-row** (repeated `num_tile_rows` times):
1. Read 32 input sticks using split-rows pattern:
   - Resolve 32 NoC addresses for sticks `[row_id*32 ... row_id*32 + 31]`
   - For each width block (just 1 block since block_width = Wt):
     - `cb_reserve_back(c_0, Wt)`
     - For k in [0..31]: `noc_async_read(base_src_noc_addr[k], l1_write_addr, block_width_size)`
     - `noc_async_read_barrier()`
     - `cb_push_back(c_0, Wt)`

**DRAM access pattern**: 32 reads per tile-row, each reading `Wt * 32 * element_size` bytes from a potentially different DRAM bank (interleaved round-robin).

### RISCV_1 ("writer" / NCRISC) Access

**Per tile-row** (repeated `num_tile_rows` times):
1. Wait for untilized output in c_16: `cb_wait_front(c_16, Wt)`
2. Get L1 read address from CB
3. For each of 32 rows within the tile-row:
   - Compute destination NoC address: `s.get_noc_addr(stick_id)`
   - Write `output_block_width_size` bytes from L1 to DRAM
   - Advance L1 read pointer by `output_block_width_size`
4. `noc_async_write_barrier()`
5. `cb_pop_front(c_16, Wt)`

**DRAM access pattern**: 32 writes per tile-row, each writing `Wt * 32 * element_size` bytes to a potentially different DRAM bank.

### Compute Access

**One-time setup** (gamma/beta tilize):
1. `compute_kernel_hw_startup(c_0, c_1, c_16)` -- initialize TRISC threads
2. Tilize gamma: `compute_kernel_lib::tilize<c_3, c_5>(Wt, 1)` -- 1 block, Wt tiles wide
3. Tilize beta: `compute_kernel_lib::tilize<c_4, c_6>(Wt, 1)` -- 1 block, Wt tiles wide
4. Do NOT pop c_5 or c_6 -- they persist for the entire program.

**Per tile-row** (the main loop):

Step 1: Tilize input
- `compute_kernel_lib::tilize<c_0, c_2>(Wt, 1)` -- tilize 1 block of Wt tiles
- c_0 is popped internally by tilize helper. c_2 now has Wt tilized tiles.

Step 2: Compute mean
- `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>(c_2, c_1, c_24, ReduceInputBlockShape::row(Wt))`
- c_2 tiles stay in CB (WaitUpfrontNoPop). c_24 gets 1 tile with mean in col-0.

Step 3: Subtract mean (centering)
- `compute_kernel_lib::sub<COL, NoWaitNoPop, WaitUpfrontNoPop>(c_2, c_24, c_25, BinaryInputBlockShape::row(Wt))`
- c_2 tiles consumed (popped by input_a). c_24 stays (NoPop for input_b since it is COL broadcast with WaitUpfrontNoPop).
- Actually: after sub completes, we need c_24 to NOT be popped. The sub<COL> helper with input_b_policy=WaitUpfrontNoPop will wait for c_24 upfront and not pop it. But we need to explicitly pop c_24 after this step.
- c_25 now has Wt tiles of (x - mean).
- Pop c_2 (consumed by sub's input_a_policy -- use WaitUpfrontPopAtEnd for c_2 to pop after sub).
- Pop c_24 (mean no longer needed).

Step 4: Square the centered values
- `compute_kernel_lib::square<WaitUpfrontNoPop>(c_25, c_26, BinaryInputBlockShape::row(Wt))`
- c_25 tiles stay (WaitUpfrontNoPop) since we need centered values again in step 7.
- c_26 now has Wt tiles of (x - mean)^2.

Step 5: Compute variance
- `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>(c_26, c_1, c_27, ReduceInputBlockShape::row(Wt))`
- c_26 tiles consumed and popped. c_27 gets 1 tile with variance in col-0.

Step 6: Add epsilon and compute rsqrt
- This step needs special care. We need to:
  a. Add epsilon scalar to the variance tile, producing (var + eps)
  b. Compute rsqrt of (var + eps), producing rstd
- Use `compute_kernel_lib::add<SCALAR>(c_27, c_7, c_28, BinaryInputBlockShape::single())` with rsqrt as post_op lambda:
  ```
  add<SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop, PerTile, INPUT_AND_OUTPUT, true>(
      c_27, c_7, c_28,
      BinaryInputBlockShape::single(), {},
      NoAccumulation{},
      [](uint32_t dst_idx) { rsqrt_tile_init(); rsqrt_tile(dst_idx); }
  )
  ```
- c_27 consumed. c_7 stays (epsilon persists). c_28 gets 1 tile with rstd in all positions (rsqrt operates element-wise but only col-0 has meaningful values from the reduce).

Step 7: Multiply centered by rstd (normalization)
- `compute_kernel_lib::mul<COL, WaitUpfrontPopAtEnd, WaitUpfrontNoPop>(c_25, c_28, c_29, BinaryInputBlockShape::row(Wt))`
- c_25 consumed and popped. c_28 stays (rstd for COL broadcast, NoPop).
- c_29 now has Wt tiles of normalized values.
- Pop c_28 after this step.

Step 8: Multiply by gamma
- `compute_kernel_lib::mul<NONE, WaitAndPopPerTile, NoWaitNoPop>(c_29, c_5, c_30, BinaryInputBlockShape::row(Wt))`
- c_29 consumed and popped. c_5 stays (gamma is persistent, NoWaitNoPop).
- c_30 now has Wt tiles of gamma * normalized.

Step 9: Add beta
- `compute_kernel_lib::add<NONE, WaitAndPopPerTile, NoWaitNoPop>(c_30, c_6, c_31, BinaryInputBlockShape::row(Wt))`
- c_30 consumed and popped. c_6 stays (beta is persistent, NoWaitNoPop).
- c_31 now has Wt tiles of the final result (tilized).

Step 10: Untilize output
- `compute_kernel_lib::untilize<Wt, c_31, c_16>(1)` -- untilize 1 block of Wt tiles
- c_31 consumed. c_16 now has Wt tiles of RM data for the writer.

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one input stick in bytes (W * element_size) |
| 1 | is_float32 | uint32_t | 1 if dtype is float32, 0 for bfloat16 |
| 2+ | TensorAccessorArgs (input) | uint32_t[] | Bank distribution metadata for input buffer |
| N+ | TensorAccessorArgs (gamma) | uint32_t[] | Bank distribution metadata for gamma buffer |
| M+ | TensorAccessorArgs (beta) | uint32_t[] | Bank distribution metadata for beta buffer |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output CB index (c_16) |
| 1 | output_stick_size | uint32_t | Size of one output stick in bytes (W * element_size) |
| 2 | tile_height | uint32_t | Height of a tile (32) |
| 3 | num_blocks_across_height | uint32_t | Total number of tile-rows (num_sticks / 32) |
| 4 | num_blocks_per_row | uint32_t | Number of width blocks per tile-row (1 for WSmall) |
| 5 | num_tiles_per_block | uint32_t | Tiles per block (Wt) |
| 6 | block_width_bytes | uint32_t | Bytes per block-row written (Wt * 32 * element_size) |
| 7+ | TensorAccessorArgs (output) | uint32_t[] | Bank distribution metadata for output buffer |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tile_rows | uint32_t | Total tile-rows to process (num_sticks / 32) |
| 1 | Wt | uint32_t | Number of tiles along W dimension (W / 32) |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_addr | uint32_t | Input buffer base address in DRAM |
| 1 | gamma_addr | uint32_t | Gamma buffer base address in DRAM |
| 2 | beta_addr | uint32_t | Beta buffer base address in DRAM |
| 3 | num_sticks | uint32_t | Total number of input sticks |
| 4 | Wt | uint32_t | Tiles along W dimension |
| 5 | block_width_size | uint32_t | Bytes per block per stick (Wt * 32 * element_size) |
| 6 | reduce_scaler | uint32_t | Packed scaler value for 1/W (bf16<<16\|bf16 for bfloat16, float32 bits for float32) |
| 7 | eps_scalar | uint32_t | Packed epsilon value (bf16<<16\|bf16 for bfloat16, float32 bits for float32) |
| 8 | gamma_num_sticks | uint32_t | Number of gamma sticks (may be 1 if gamma is 1D) |
| 9 | gamma_stick_size | uint32_t | Size of one gamma stick in bytes |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_addr | uint32_t | Output buffer base address in DRAM |

## Edge Cases

| Condition | Expected Behavior |
|-----------|------------------|
| Single tile (W=32, H=32) | Wt=1, single tile per row, single row. Reduce of 1 tile. Must still work correctly. |
| W=32 (single tile width) | Wt=1. Reduce<SUM,REDUCE_ROW> on single tile. Sub/mul broadcast with single tile. |
| H=32 (single tile-row) | num_tile_rows=1 (for single batch). Loop body executes once. |
| Large H (H=4096) | num_tile_rows=128. Reader/writer loop many times. Each iteration independent. |
| Large W (W=1024) | Wt=32. All intermediate CBs are 32 tiles. L1 budget: ~12*32+5=389 tiles. Must fit. |
| Multi-batch (shape=[2,3,64,128]) | All outer dims collapsed. num_sticks = 2*3*64 = 384. num_tile_rows = 384/32 = 12. |
| 3D input (shape=[1,64,128]) | num_sticks = 1*64 = 64. num_tile_rows = 64/32 = 2. |
| bfloat16 dtype | tile_size = 2048 bytes. Scaler packed as bf16<<16\|bf16. |
| float32 dtype | tile_size = 4096 bytes. fp32_dest_acc_en = true. Intermediates use Float32 format. Scaler packed as raw float32 bits. |
| Epsilon = 0 | Valid but may produce NaN/Inf if variance is 0. User responsibility. |
| Gamma all 1.0, beta all 0.0 | Should produce same result as standardization without affine transform. |
| Constant input (all same value) | Mean = that value, variance = 0, rstd = 1/sqrt(eps). Output = gamma * 0 + beta = beta. |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-generic-op-builder** | API Specification, Input Tensor Requirements, Output Tensor Specification, Circular Buffer Requirements, Work Distribution, Runtime Arguments, Compile-Time Arguments |
| **ttnn-kernel-designer** | Data Flow, Circular Buffer Requirements, Compute Access, Memory Access Patterns, Mathematical Definition, Component Sources |
| **ttnn-kernel-dataflow** | Kernel Data Movement, Memory Access Patterns, Component Sources (input_stage, output_stage) |
| **ttnn-kernel-compute** | Compute Access, Mathematical Definition, Component Sources (compute_core) |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Wrong layout (TILE input) -> error containing "must be in ROW_MAJOR layout"
- Wrong dtype (bfloat8_b) -> error containing "unsupported dtype"
- W not multiple of 32 -> error containing "must be a multiple of 32"
- H not multiple of 32 -> error containing "must be a multiple of 32"
- Gamma/beta shape mismatch -> error containing "must match input last dim"
- Input not on device -> error containing "must be on device"

### Shape Behavior
- Output shape == input shape for all test cases
- Output layout == ROW_MAJOR
- Output dtype == input dtype

### Functional Behavior
- Single tile [1,1,32,32]: PCC > 0.99 vs torch.nn.functional.layer_norm
- Multi-tile width [1,1,32,1024]: PCC > 0.99
- Multi-tile height [1,1,1024,32]: PCC > 0.99
- Square [1,1,128,128]: PCC > 0.99
- Large [1,1,4096,32]: PCC > 0.99
- Wide [1,1,512,512]: PCC > 0.99
- Multi-batch [2,3,64,128]: PCC > 0.99
- 3D [1,64,128]: PCC > 0.99
- Both bfloat16 and float32 dtypes tested

### Test Implementation Notes
- Use `torch.nn.functional.layer_norm(input, normalized_shape=[W], weight=gamma, bias=beta, eps=epsilon)` as reference
- Use PCC (Pearson Correlation Coefficient) > 0.99 for correctness, NOT torch.allclose with tight tolerances
- bfloat16 accumulates differently on hardware vs PyTorch (which uses float32 intermediates), so exact match is not expected

## Open Questions

1. **CB memory optimization**: The current design uses 12*Wt + 5 tiles of CB memory. For large W (e.g., W=2048, Wt=64), this is 12*64+5=773 tiles * 2048 bytes = ~1.5 MB for bfloat16, which may exceed available L1 after accounting for the allocator base. Should we implement a WLarge variant for large W, or cap W to values that fit in L1? For the initial single-core implementation, we assume W fits in the WSmall pattern.

2. **Gamma/beta tilize**: Gamma and beta have shape [1,...,1,W], which means they are a single row of W elements. When tilizing, we need 32 rows of sticks (one tile-height). The reader should replicate the single gamma/beta row 32 times in the input RM CB before tilizing. Alternatively, the reader could fill 1 real row and 31 zero-rows, and the compute would handle it. The simplest approach: since gamma/beta broadcast across rows, we just need the first row of each tile to have the correct values. Reading gamma's W-element row and replicating it 32 times into the CB (like 32 identical sticks) then tilizing would produce tiles where every row has the same gamma values, which is correct for element-wise multiplication with the normalized result. **The reader should replicate the single gamma row 32 times to fill a proper tile-row.**

3. **Untilize block_width_tiles as compile-time template parameter**: The `compute_kernel_lib::untilize<>` helper requires `block_width_tiles` as a compile-time template parameter. Since Wt depends on the input tensor shape and is determined at runtime, the compute kernel must receive Wt as a compile-time argument. This is fine since compile-time args are set per-program and the program is compiled fresh for each unique Wt value.

## References

- Reference analyses:
  - `ttnn/ttnn/operations/layer_norm_rm/agent_logs/tilize_single_core_analysis.md`
  - `ttnn/ttnn/operations/layer_norm_rm/agent_logs/softmax_general_analysis.md`
  - `ttnn/ttnn/operations/layer_norm_rm/agent_logs/untilize_single_core_analysis.md`
- Kernel helper libraries:
  - `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` -- tilize<cb_in, cb_out>(block_width, num_blocks)
  - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` -- untilize<width, cb_in, cb_out>(num_blocks)
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` -- reduce<SUM, REDUCE_ROW>(...)
  - `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` -- sub<COL>(), mul<COL>(), square<>(), add<SCALAR>(), mul<NONE>(), add<NONE>()
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` -- generate_reduce_scaler()
  - `ttnn/cpp/ttnn/kernel_lib/scalar_helpers.hpp` -- generate_bcast_scalar_bfloat16(), generate_bcast_scalar()
- DeepWiki queries:
  - rsqrt_tile: Available via `compute_kernel_api/eltwise_unary/rsqrt.h`, `rsqrt_tile_init()` + `rsqrt_tile(dst_idx)`
  - add_tiles_bcast with BroadcastType::COL: `C[h,w] = A[h,w] + B[h,0]`, init with `add_bcast_cols_init_short`
- Documentation:
  - `METALIUM_GUIDE.md` (CB patterns, kernel types)
  - `tech_reports/tensor_layouts/tensor_layouts.md` (RM vs tiled layout)
  - `tech_reports/tensor_accessor/tensor_accessor.md` (TensorAccessor usage)
