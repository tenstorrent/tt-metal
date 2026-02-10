# Layer Norm RM Functional Specification

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operation(s)**: tilize (single-core), softmax (general W-dimension), untilize (single-core)
- **Reference Analyses**:
  - `ttnn/ttnn/operations/layer_norm_rm/agent_logs/tilize_single_core_analysis.md` (role: input_stage)
  - `ttnn/ttnn/operations/layer_norm_rm/agent_logs/softmax_general_analysis.md` (role: compute_core)
  - `ttnn/ttnn/operations/layer_norm_rm/agent_logs/untilize_single_core_analysis.md` (role: output_stage)

## Mathematical Definition

### Formula
```
Given input X of shape [..., H, W]:

1. mean_i      = (1/W) * sum_j( X[i, j] )              for each row i
2. x_centered  = X[i, j] - mean_i                       for each element (i, j)
3. var_i       = (1/W) * sum_j( x_centered[i, j]^2 )   for each row i
4. rstd_i      = 1 / sqrt( var_i + epsilon )            for each row i
5. y_normed    = x_centered[i, j] * rstd_i              for each element (i, j)
6. output      = gamma[j] * y_normed[i, j] + beta[j]   for each element (i, j)
```

### Semantic Description
Layer normalization computes statistics (mean and variance) across the last dimension (W) for each row independently. Each row is then centered (subtract mean), scaled by the inverse standard deviation (1/sqrt(var + epsilon)), and finally transformed by learnable affine parameters gamma (scale) and beta (bias) that are shared across all rows.

The input and output are both in row-major layout stored in DRAM. Internally, the data must be tilized for compute operations and untilized back to row-major for the output. Gamma and beta are also row-major and must be tilized once, then reused for every row.

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | rank >= 2 | - | Input tensor in ROW_MAJOR layout |
| gamma | Tensor or None | No | shape [1,...,1, W] | None | Per-element scale (broadcastable along last dim) |
| beta | Tensor or None | No | shape [1,...,1, W] | None | Per-element bias (broadcastable along last dim) |
| epsilon | float | No | > 0 | 1e-5 | Small constant for numerical stability |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Layout | ROW_MAJOR | "input must be in ROW_MAJOR layout" |
| Dtype | bfloat16 or float32 | "unsupported dtype, must be bfloat16 or float32" |
| Device | Must be on device | "input must be on device" |
| Memory | DRAM, interleaved | "input must be interleaved in DRAM" |
| Rank | >= 2 | "input must have rank >= 2" |
| W (last dim) | multiple of 32 | "last dimension must be a multiple of 32" |
| H (second-to-last dim) | multiple of 32 | "second-to-last dimension must be a multiple of 32" |

### Gamma/Beta Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Layout | ROW_MAJOR | "gamma/beta must be in ROW_MAJOR layout" |
| Dtype | must match input dtype | "gamma/beta dtype must match input dtype" |
| Device | Must be on device | "gamma/beta must be on device" |
| Memory | DRAM, interleaved | "gamma/beta must be interleaved in DRAM" |
| Shape | last dim = input W | "gamma/beta last dim must match input last dim" |
| Shape | all dims except last = 1 | "gamma/beta must be broadcastable to input shape" |

### Output Tensor Specification
- **Shape**: Same as input `[..., H, W]`
- **Dtype**: Same as input
- **Layout**: ROW_MAJOR
- **Memory**: DRAM, interleaved

## Component Sources (Hybrid Mode)

This operation is composed from three references:

### Input Stage (from tilize single-core analysis)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Reader kernel pattern | tilize.reader_unary_stick_layout_split_rows_interleaved | Extended to also read gamma and beta sticks. Uses same split-rows pattern for input (32 sticks at a time, block-width at a time). Gamma/beta read as Wt full sticks (one complete tile-row of parameters). |
| CB c_0 (input RM sticks) | tilize.cb_in0 | Same purpose and sizing as tilize: holds `Wt` tiles worth of row-major stick data (32 sticks x block_width). |
| Compute tilize phase | tilize.compute kernel | Reused as the first phase of the compute kernel. Tilizes input data from c_0 to c_1. Also tilizes gamma from c_2 to c_3, and beta from c_4 to c_5 (once at program start). |

### Compute Stage (from softmax general W-dimension analysis)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Row-wise SUM reduction for mean | softmax.reduce MAX -> changed to SUM | Use `reduce<SUM, REDUCE_ROW>` with scaler = 1/W to compute mean directly. No masking needed since W is guaranteed tile-aligned. |
| Broadcast subtract (x - mean) | softmax.sub_tiles_bcast<COL> | Same pattern: broadcast the mean tile (1 tile per row) across all Wt tiles. |
| Square for variance | New (not in softmax) | Use `square()` helper from binary_op_helpers.hpp on centered tiles. |
| Row-wise SUM reduction for variance | softmax.reduce SUM | Use `reduce<SUM, REDUCE_ROW>` with scaler = 1/W on squared centered tiles. |
| Rsqrt (1/sqrt(var+eps)) | New (not in softmax, which uses recip) | Add epsilon via `add<SCALAR>` then apply rsqrt_tile as post-reduce op or separate step. |
| Multiply by rstd | softmax.mul_tiles_bcast_cols | Same pattern: broadcast rstd (1 tile) across all Wt tiles using `mul<COL>`. |
| Multiply by gamma | New affine step | Use `mul<COL>` with Wt gamma tiles. Since gamma has Wt tiles (not 1), this is a per-element multiply NOT a broadcast. Use `mul<NONE>`. |
| Add beta | New affine step | Use `add<NONE>` with Wt beta tiles. Per-element add, NOT a broadcast. |
| CB configuration | softmax intermediate CBs | Adapted for layer norm intermediate results (mean, centered, variance, rstd, gamma_tilized, beta_tilized). |
| Reduce scaler setup | softmax.generate_reduce_scaler | Same pattern: reader generates reduce scaler tile with value = packed bfloat16 of 1/W. CB must be bfloat16 format regardless of input dtype. |

### Output Stage (from untilize single-core analysis)
| Component | Source | Modifications |
|-----------|--------|---------------|
| Compute untilize phase | untilize.compute kernel | Reused as the final phase of the compute kernel. Untilizes result from c_24 (output tiles) to c_16 (output RM sticks). |
| Writer kernel pattern | untilize.writer_unary_stick_layout_split_rows_single_core | Writes row-major sticks from c_16 to DRAM. Same split-rows write pattern. |
| CB c_16 (output RM sticks) | untilize.cb_output | Same purpose: holds untilized row-major data for writer to drain. |

### Interface Compatibility
| Interface | Component A | Component B | Format A | Format B | Compatible? |
|-----------|------------|-------------|----------|----------|-------------|
| Reader -> Tilize | tilize.reader (c_0 RM sticks) | tilize.compute (reads c_0) | Row-major sticks | Row-major sticks | Yes |
| Tilize -> Norm Compute | tilize output (c_1, tile format) | softmax-style compute (tile input) | TILE | TILE | Yes |
| Norm Compute -> Untilize | norm output (c_24, tile format) | untilize.compute (reads tiles) | TILE | TILE | Yes |
| Untilize -> Writer | untilize output (c_16, RM sticks) | untilize.writer (reads c_16) | Row-major sticks | Row-major sticks | Yes |

### CB ID Resolution

| Logical CB | Source Ref | Original ID | Final ID | Notes |
|------------|-----------|-------------|----------|-------|
| cb_input_rm | tilize | c_0 | c_0 | Input row-major sticks from reader |
| cb_tilized_input | tilize (output) | c_16 | c_1 | Tilized input tiles. Remapped from c_16 to avoid conflict with output. |
| cb_gamma_rm | New | - | c_2 | Gamma row-major sticks from reader |
| cb_gamma_tilized | New | - | c_3 | Gamma tilized tiles (persistent, program lifetime) |
| cb_beta_rm | New | - | c_4 | Beta row-major sticks from reader |
| cb_beta_tilized | New | - | c_5 | Beta tilized tiles (persistent, program lifetime) |
| cb_reduce_scaler | softmax | c_2 | c_6 | Reduce scaler tile (1/W). MUST be bfloat16 format. Remapped from c_2 to avoid conflict with cb_gamma_rm. |
| cb_eps_scalar | New | - | c_7 | Epsilon scalar tile for add-epsilon step |
| cb_mean | softmax | c_26 | c_24 | Row-wise mean (1 tile). Remapped to intermediates range. |
| cb_centered | softmax | c_27 | c_25 | x - mean intermediate (Wt tiles) |
| cb_centered_sq | New | - | c_26 | (x - mean)^2 intermediate (Wt tiles) |
| cb_var | New | - | c_27 | Row-wise variance (1 tile) |
| cb_rstd | softmax (recipsumexps) | c_25 | c_28 | 1/sqrt(var+eps) (1 tile) |
| cb_normed | New | - | c_29 | x_centered * rstd intermediate (Wt tiles) |
| cb_gamma_applied | New | - | c_30 | gamma * normed intermediate (Wt tiles) |
| cb_output_tiles | New | - | c_8 | Final output in tile format (Wt tiles), before untilize |
| cb_output_rm | untilize | c_16 | c_16 | Output row-major sticks for writer |

## Design Decisions

### Decision 1: Single-Core Execution
- **Choice**: Run all computation on a single Tensix core.
- **Rationale**: This is the stated requirement. Single-core simplifies the reader/writer logic (no work splitting, no tile_offset calculations) and matches both the tilize and untilize single-core reference patterns. Multi-core can be added later as an optimization.
- **Alternatives Considered**: Multi-core with row-based work splitting (like softmax). Deferred for a future version.
- **Tradeoffs**: Lower throughput for large tensors, but simpler implementation and debugging.

### Decision 2: WSmall-Only Pattern (All Tiles in L1)
- **Choice**: Use the WSmall pattern from softmax -- load an entire tile-row (Wt tiles) into L1 at once.
- **Rationale**: Layer norm requires two passes over the data (once for mean, once for variance), so tiles must be accessible multiple times. The WSmall pattern keeps all Wt tiles in L1, avoiding DRAM re-reads. Since W must be a multiple of 32 and we are single-core, the L1 budget is generous (1.5MB).
- **Alternatives Considered**: WLarge pattern (3-pass DRAM re-reads). Not needed unless W is extremely large.
- **Tradeoffs**: Higher L1 usage but much better performance (no DRAM re-reads).

### Decision 3: Gamma/Beta Tilized Once and Reused
- **Choice**: Read gamma and beta once at program start, tilize them in the compute kernel, and store tilized results in persistent CBs (c_3, c_5) for reuse across all rows.
- **Rationale**: Gamma and beta have shape [1, W] and are identical for every row. Reading and tilizing them once eliminates redundant DRAM reads. The tilized gamma/beta CBs are marked as program-lifetime, meaning they persist across all row iterations.
- **Alternatives Considered**: Re-reading gamma/beta per row from DRAM. Wasteful.
- **Tradeoffs**: Uses 2 * Wt tiles of persistent L1 space for gamma and beta, but this is a small cost compared to the savings.

### Decision 4: Reduce Scaler = 1/W for Direct Mean Computation
- **Choice**: Set the reduce scaler to 1/W so that `reduce<SUM, REDUCE_ROW>` directly computes the mean (sum * 1/W = mean) in a single step.
- **Rationale**: The hardware automatically multiplies the reduction result by the scaler. By using 1/W, we avoid a separate divide step. This is the same approach suggested in the softmax analysis for layer norm adaptation.
- **Alternatives Considered**: Reduce with scaler=1.0 then multiply by 1/W. Extra step with no benefit.
- **Tradeoffs**: None -- strictly better.

### Decision 5: Reduce Scaler CB in bfloat16 Format
- **Choice**: The CB for the reduce scaler (c_6) is always configured with bfloat16 data format, regardless of whether the operation runs in float32 mode.
- **Rationale**: The reduce hardware expects scaler values in bfloat16 format. The `generate_reduce_scaler` function writes packed bfloat16 values (bf16 << 16 | bf16). Even when fp32_dest_acc_en is true for the rest of the computation, the scaler CB must be bfloat16.
- **Alternatives Considered**: None -- this is a hardware requirement.
- **Tradeoffs**: None.

### Decision 6: No Width Mask Needed
- **Choice**: Do not generate or use a width mask tile.
- **Rationale**: The constraint that W must be a multiple of 32 means the last tile in every row is fully populated -- there are no padding columns that need to be zeroed out. Softmax uses a mask because it supports arbitrary W, but layer_norm_rm does not need this.
- **Alternatives Considered**: Including a mask for safety. Unnecessary overhead.
- **Tradeoffs**: If the constraint is relaxed in the future, a mask will need to be added.

### Decision 7: Epsilon Handling via Scalar Add
- **Choice**: After computing variance, add epsilon using `add<SCALAR>` with a precomputed epsilon tile, then apply rsqrt.
- **Rationale**: The epsilon value needs to be added to the variance before computing the inverse square root. Using a broadcast scalar add is the cleanest approach. The epsilon scalar tile is generated once by the reader and persists for the program lifetime.
- **Alternatives Considered**: Fusing epsilon into the rsqrt operation. Not possible with the available hardware primitives.
- **Tradeoffs**: Requires one extra CB (c_7) for the epsilon scalar tile.

### Decision 8: Separate Tilize Source and Destination CBs
- **Choice**: Use separate CBs for tilize input (row-major) and output (tiled) -- e.g., c_0 -> c_1 for input, c_2 -> c_3 for gamma, c_4 -> c_5 for beta.
- **Rationale**: Tilize is NOT in-place. The hardware unpack unit reads from one CB and writes to another. The source CB contains row-major sticks; the destination CB contains properly formatted tiles. Using the same CB for both would corrupt data.
- **Alternatives Considered**: None -- this is a hardware constraint.
- **Tradeoffs**: Doubles the CB count for the tilize stage, but this is unavoidable.

### Decision 9: Block Width for Tilize = Wt (Full Row)
- **Choice**: Set `num_tiles_per_block = Wt` for the tilize operation, processing an entire tile-row at once.
- **Rationale**: Since we need all Wt tiles in L1 for the normalization compute anyway, we should tilize the full row in one shot. This avoids partial-block complexity and aligns with the WSmall compute pattern. The L1 budget is sufficient for single-core.
- **Alternatives Considered**: Smaller block widths for L1 pressure. Not needed for single-core.
- **Tradeoffs**: Higher L1 usage per block, but simpler logic.

### Decision 10: Per-Row Processing Loop Structure
- **Choice**: Process the tensor one tile-row at a time. For each tile-row:
  1. Reader reads 32 row-major sticks (one tile-row worth of input) into c_0
  2. Compute tilizes c_0 -> c_1 (Wt tiles)
  3. Compute performs layer norm on Wt tiles (mean, center, variance, rstd, normalize, gamma, beta)
  4. Compute untilizes result from c_8 -> c_16
  5. Writer writes 32 row-major sticks from c_16 to DRAM
- **Rationale**: This matches the natural row-wise independence of layer normalization. Each tile-row is fully self-contained (except for gamma/beta which are shared).
- **Alternatives Considered**: Processing multiple tile-rows at once. Increases L1 pressure without benefit since rows are independent.
- **Tradeoffs**: Simple, correct, and memory-efficient.

### Decision 11: Intermediate CB Data Format
- **Choice**: When input dtype is float32 (fp32_dest_acc_en = true), intermediate CBs use float32 format. When input is bfloat16, intermediates use bfloat16 (or float32 if fp32 accumulation is desired for precision).
- **Rationale**: Matching the intermediate format to the accumulation precision avoids data format mismatches and ensures correct pack/unpack behavior.
- **Alternatives Considered**: Always using float32 intermediates. Would waste L1 for bfloat16 inputs with minimal precision benefit in many cases.
- **Tradeoffs**: bfloat16 intermediates may lose precision in variance computation. For bfloat16 inputs, consider enabling fp32_dest_acc_en for better numerical accuracy.

## Work Distribution

### Work Unit Definition
| Attribute | Value |
|-----------|-------|
| **Granularity** | One tile-row (32 rows of the tensor, all Wt tiles across the width) |
| **Unit size** | Wt tiles (one complete tile-row) |
| **Total units** | num_tile_rows = total_num_sticks / 32 |
| **Loop structure** | Single loop over tile-rows. Within each tile-row: tilize -> normalize -> untilize. |

### Parallelization Strategy
- **Grid**: 1 x 1 (single core)
- **Work per core**: All tile-rows (entire tensor)
- **Load balancing**: N/A (single core)

## Data Flow

### High-Level Flow
```
For each tile-row (32 sticks):
  Reader:  DRAM (RM sticks) ──read 32 sticks──> c_0
  Compute: c_0 (RM)  ──tilize──>  c_1 (tiled, Wt tiles)
           c_1        ──reduce SUM * 1/W──>  c_24 (mean, 1 tile)
           c_1, c_24  ──sub<COL>──>  c_25 (centered, Wt tiles)
           c_1        ──pop──  (free input tiles)
           c_25       ──square──>  c_26 (centered^2, Wt tiles)
           c_26       ──reduce SUM * 1/W──>  c_27 (variance, 1 tile)
           c_27, c_7  ──add<SCALAR>──>  c_27 (var+eps, reuse CB)
           c_27       ──rsqrt──>  c_28 (rstd, 1 tile)
           c_25, c_28 ──mul<COL>──>  c_29 (normed, Wt tiles)
           c_25       ──pop──  (free centered tiles)
           c_29, c_3  ──mul<NONE>──>  c_30 (gamma*normed, Wt tiles)
           c_29       ──pop──  (free normed tiles)
           c_30, c_5  ──add<NONE>──>  c_8 (output tiles, Wt tiles)
           c_30       ──pop──  (free gamma*normed tiles)
           c_8  (tiled) ──untilize──>  c_16 (RM sticks)
  Writer:  c_16 (RM sticks) ──write 32 sticks──> DRAM
```

At program start (before the main loop):
```
  Reader:  DRAM (gamma RM sticks) ──read Wt sticks──> c_2
           DRAM (beta RM sticks)  ──read Wt sticks──> c_4
           generate_reduce_scaler(c_6, packed_1_over_W)
           generate_bcast_scalar(c_7, packed_epsilon)
  Compute: c_2 (RM) ──tilize──> c_3 (gamma tiled, Wt tiles, persistent)
           c_4 (RM) ──tilize──> c_5 (beta tiled, Wt tiles, persistent)
```

### Kernel Data Movement

| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| Reader | RISCV_0 (BRISC) | NOC0 | Reads RM sticks from DRAM for input, gamma, beta. Generates reduce scaler and epsilon scalar tiles in L1. |
| Compute | RISCV_2 (TRISC UNPACK+MATH+PACK) | N/A | Tilizes input/gamma/beta, performs layer norm (mean, center, variance, rstd, normalize, gamma, beta), untilizes output. |
| Writer | RISCV_1 (NCRISC) | NOC1 | Writes RM sticks from L1 to DRAM for output. |

### Circular Buffer Requirements

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime | Data Format |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|-------------|
| c_0 | cb_input_rm | Input row-major sticks staging | Wt tiles | Wt tiles | Single | Reader | Compute (tilize) | Row | input dtype |
| c_1 | cb_tilized_input | Tilized input tiles | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (norm) | Row | input dtype |
| c_2 | cb_gamma_rm | Gamma row-major sticks staging | Wt tiles | Wt tiles | Single | Reader | Compute (tilize) | One-shot | input dtype |
| c_3 | cb_gamma_tilized | Gamma tilized tiles (reused every row) | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (affine) | Program | input dtype |
| c_4 | cb_beta_rm | Beta row-major sticks staging | Wt tiles | Wt tiles | Single | Reader | Compute (tilize) | One-shot | input dtype |
| c_5 | cb_beta_tilized | Beta tilized tiles (reused every row) | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (affine) | Program | input dtype |
| c_6 | cb_reduce_scaler | Reduce scaler tile (1/W) | 1 tile | 1 tile | Single | Reader | Compute (reduce) | Program | **bfloat16 always** |
| c_7 | cb_eps_scalar | Epsilon scalar tile | 1 tile | 1 tile | Single | Reader | Compute (add eps) | Program | input dtype |
| c_8 | cb_output_tiles | Final output tiles before untilize | Wt tiles | Wt tiles | Single | Compute (affine) | Compute (untilize) | Row | input dtype |
| c_16 | cb_output_rm | Output row-major sticks for writer | Wt tiles | Wt tiles | Single | Compute (untilize) | Writer | Row | input dtype |
| c_24 | cb_mean | Row-wise mean (reduction output) | 1 tile | 1 tile | Single | Compute | Compute | Row | intermed dtype |
| c_25 | cb_centered | x - mean intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row | intermed dtype |
| c_26 | cb_centered_sq | (x - mean)^2 intermediate | Wt tiles | Wt tiles | Single | Compute | Compute | Row | intermed dtype |
| c_27 | cb_var | Row-wise variance (also reused for var+eps) | 1 tile | 1 tile | Single | Compute | Compute | Row | intermed dtype |
| c_28 | cb_rstd | 1/sqrt(var+eps) | 1 tile | 1 tile | Single | Compute | Compute | Row | intermed dtype |
| c_29 | cb_normed | x_centered * rstd | Wt tiles | Wt tiles | Single | Compute | Compute | Row | intermed dtype |
| c_30 | cb_gamma_applied | gamma * normed | Wt tiles | Wt tiles | Single | Compute | Compute | Row | intermed dtype |

**Data format notes**:
- "input dtype" = bfloat16 or float32 matching the input tensor
- "intermed dtype" = float32 if fp32_dest_acc_en is true, else matches input dtype
- c_6 (reduce scaler) is **always bfloat16** regardless of input dtype -- this is a hardware requirement

**Total L1 usage estimate** (Wt-sized CBs at bfloat16):
- 10 CBs at Wt tiles: c_0, c_1, c_3, c_5, c_8, c_16, c_25, c_26, c_29, c_30 = 10 * Wt * 2048 bytes
- 2 one-shot CBs at Wt tiles: c_2, c_4 (freed after tilize) = 2 * Wt * 2048 bytes (temporary)
- 4 CBs at 1 tile: c_6, c_7, c_24, c_27, c_28 = 5 * 2048 bytes
- For Wt=32 (W=1024): ~10 * 32 * 2048 = 640KB + overhead. Fits in 1.5MB L1.

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**Phase 0 -- Program initialization (once)**:
1. Generate reduce scaler tile into c_6: `generate_reduce_scaler(c_6, packed_1_over_W)`. The scaler value is 1/W packed as `(bf16 << 16 | bf16)`.
2. Generate epsilon scalar tile into c_7: `generate_bcast_scalar(c_7, packed_epsilon)` (or `generate_bcast_scalar_bfloat16` for bf16, `generate_bcast_scalar` for f32).
3. Read gamma sticks: For each of Wt sticks (gamma is 1 row of W elements = Wt sticks of 32 elements each): `noc_async_read` from DRAM into c_2. Push c_2 with Wt tiles.
4. Read beta sticks: Same as gamma but into c_4. Push c_4 with Wt tiles.

**Phase 1 -- Main loop (per tile-row)**:
5. For each tile-row (num_tile_rows iterations):
   - Pre-compute 32 NoC addresses for the 32 sticks in this tile-row using TensorAccessor
   - `cb_reserve_back(c_0, Wt)`
   - For each of 32 sticks: `noc_async_read(base_addr[stick], l1_write_ptr, stick_size)` where stick_size = W * element_size
   - L1 write address advances by stick_size after each stick, packing all 32 sticks contiguously (same split-rows pattern as tilize reference)
   - `noc_async_read_barrier()`
   - `cb_push_back(c_0, Wt)`

**Gamma/Beta read pattern**: Sequential stick reads using TensorAccessor with page_size = W * element_size (one row). Since gamma/beta are 1D tensors of shape [W], they consist of a single stick that spans W elements. However, for the tilize hardware to work, we need to present 32 sticks (one tile-row). If gamma/beta have only 1 row, we read that single stick 32 times (broadcast across 32 rows). Alternative: read Wt sticks worth of data where each "stick" is 32 elements wide (TILE_WIDTH). The exact approach depends on how gamma/beta are padded. Since gamma/beta have shape [1, W] and must be tilized, the reader should read them as Wt "mini-sticks" of 32 elements each, repeated 32 times vertically to fill a tile-row. In practice, the simplest approach is: read the one row of W elements as raw bytes, then let the tilize hardware handle it. The reader writes W * element_size bytes 32 times (repeating the same row) into the CB to create a proper tilize-able block.

**Simplified gamma/beta approach**: Read W * element_size bytes (the full gamma row) into a temporary location in L1, then fill the CB with 32 copies of this row. This creates a 32xW block that tilizes into Wt tiles where every row in every tile has the same gamma values -- which is exactly what we want for per-element affine transform.

### RISCV_1 ("writer" / NCRISC) Access

**Main loop (per tile-row)**:
1. `cb_wait_front(c_16, Wt)` -- wait for untilized output sticks
2. For each of 32 sticks in the tile-row:
   - Compute DRAM destination address using TensorAccessor (output is RM interleaved, page_size = W * element_size)
   - `noc_async_write(l1_read_addr, dst_noc_addr, stick_size)`
   - l1_read_addr += stick_size
3. `noc_async_write_barrier()`
4. `cb_pop_front(c_16, Wt)`

The write pattern matches the untilize reference: after untilize, data in c_16 is already in row-major stick order. Each stick is W * element_size bytes. The writer writes 32 sticks per tile-row.

### Compute Access

**Phase 0 -- Tilize gamma and beta (once)**:
1. `tilize<c_2, c_3>(Wt, 1)` -- tilize gamma from RM to tiles
2. `tilize<c_4, c_5>(Wt, 1)` -- tilize beta from RM to tiles
3. Pop c_2 and c_4 (no longer needed)

**Phase 1 -- Main loop (per tile-row)**:
For each tile-row:

**Step 1: Tilize input**
- `tilize<c_0, c_1>(Wt, 1)` -- tilize current row's input sticks into tiles

**Step 2: Compute row-wise mean**
- `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>(c_1, c_6, c_24, ReduceInputBlockShape::row(Wt))`
- Input tiles in c_1 remain available (WaitUpfrontNoPop policy)
- Output: 1 tile in c_24 containing per-row means

**Step 3: Compute x - mean**
- `sub<COL, WaitUpfrontNoPop, WaitAndPopPerTile>(c_1, c_24, c_25, BinaryInputBlockShape::row(Wt))`
- c_1 tiles: WaitUpfrontNoPop (already waited, keep for reuse -- actually no, we can pop after this step)
- c_24 (mean): single tile broadcast across columns
- Output: Wt tiles in c_25
- After this: pop c_1 (Wt tiles), pop c_24 (1 tile)

**Step 4: Compute (x - mean)^2**
- `square<WaitUpfrontNoPop>(c_25, c_26, BinaryInputBlockShape::row(Wt))`
- c_25 tiles remain available (needed for step 7)
- Output: Wt tiles in c_26

**Step 5: Compute row-wise variance**
- `reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>(c_26, c_6, c_27, ReduceInputBlockShape::row(Wt))`
- Pops c_26 tiles as consumed
- Output: 1 tile in c_27 containing per-row variance

**Step 6: Compute rstd = 1/sqrt(var + eps)**
- `add<SCALAR>(c_27, c_7, c_27, BinaryInputBlockShape::single())`  -- var + eps, reuse c_27
- Apply rsqrt: This can be done as a post-reduce op or a separate step. As a separate step:
  - Wait for c_27, acquire DST, copy tile, rsqrt_tile, pack to c_28, pop c_27
- Output: 1 tile in c_28

**Step 7: Compute normed = centered * rstd**
- `mul<COL, WaitUpfrontNoPop, WaitAndPopPerTile>(c_25, c_28, c_29, BinaryInputBlockShape::row(Wt))`
- c_25 (centered): Wt tiles, already waited from step 3 (WaitUpfrontNoPop in step 4)
- c_28 (rstd): 1 tile broadcast across columns
- Output: Wt tiles in c_29
- After this: pop c_25 (Wt tiles), pop c_28 (1 tile)

**Step 8: Apply gamma**
- `mul<NONE, WaitAndPopPerTile, NoWaitNoPop>(c_29, c_3, c_30, BinaryInputBlockShape::row(Wt))`
- c_29 (normed): Wt tiles, consumed per-tile
- c_3 (gamma_tilized): Wt tiles, persistent (NoWaitNoPop -- already waited at program start, never popped)
- Output: Wt tiles in c_30

**Step 9: Apply beta**
- `add<NONE, WaitAndPopPerTile, NoWaitNoPop>(c_30, c_5, c_8, BinaryInputBlockShape::row(Wt))`
- c_30 (gamma*normed): Wt tiles, consumed per-tile
- c_5 (beta_tilized): Wt tiles, persistent (NoWaitNoPop)
- Output: Wt tiles in c_8

**Step 10: Untilize output**
- `untilize<Wt, c_8, c_16>(1)` -- untilize 1 block of Wt tiles from c_8 to c_16

End of per-row loop.

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one input stick in bytes (W * element_size) |
| 1 | gamma_beta_stick_size | uint32_t | Size of one gamma/beta stick in bytes (same as stick_size since gamma/beta have width W) |
| 2+ | TensorAccessorArgs (input) | uint32_t[] | Bank mapping for input buffer |
| N+ | TensorAccessorArgs (gamma) | uint32_t[] | Bank mapping for gamma buffer |
| M+ | TensorAccessorArgs (beta) | uint32_t[] | Bank mapping for beta buffer |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Number of tiles along W dimension |
| 1 | num_tile_rows | uint32_t | Total number of tile-rows to process |
| 2 | has_gamma | uint32_t (bool) | 1 if gamma is provided, 0 if not |
| 3 | has_beta | uint32_t (bool) | 1 if beta is provided, 0 if not |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_stick_size | uint32_t | Size of one output stick in bytes (W * element_size) |
| 1 | tile_height | uint32_t | Height of a tile (32) |
| 2 | num_tile_rows | uint32_t | Total number of tile-rows (height blocks) |
| 3 | Wt | uint32_t | Tiles along W dimension (for cb_wait_front/pop_front) |
| 4+ | TensorAccessorArgs (output) | uint32_t[] | Bank mapping for output buffer |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer DRAM base address |
| 1 | gamma_addr | uint32_t | Gamma buffer DRAM base address (0 if no gamma) |
| 2 | beta_addr | uint32_t | Beta buffer DRAM base address (0 if no beta) |
| 3 | num_sticks | uint32_t | Total number of input sticks (H * batch_dims) |
| 4 | num_tile_rows | uint32_t | Number of tile-rows (num_sticks / 32) |
| 5 | Wt | uint32_t | Tiles along W |
| 6 | reduce_scaler | uint32_t | Packed bfloat16 value of 1/W: `(bf16(1/W) << 16) | bf16(1/W)` |
| 7 | eps_scalar | uint32_t | Packed scalar epsilon value (bf16 packed for bf16 dtype, f32 bit-cast for f32 dtype) |

### Compute Kernel

No runtime arguments. All parameters are compile-time.

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer DRAM base address |

## Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile (H=32, W=32) | One tile-row, one tile. Reduce produces 1 tile, all operations work on 1 tile. |
| W = 32 (Wt = 1) | Minimal width. Reduce scaler = 1/32. All intermediate CBs hold 1 tile. |
| Large W (e.g., W = 4096, Wt = 128) | Must fit 10 * 128 tiles in L1 (bfloat16: ~2.5MB). May exceed L1 for very large W. Validation should check L1 capacity. |
| H = 32 (single tile-row) | Only one iteration of the main loop. |
| No gamma, no beta | Skip affine transform steps (steps 8 and 9). Output = normed directly. Untilize from c_29 (or c_25 * c_28 output) instead of c_8. |
| Gamma only, no beta | Skip beta add (step 9). Output = gamma * normed. Untilize from c_30. |
| No gamma, beta only | Skip gamma multiply (step 8). Output = normed + beta. Untilize from add(c_29, c_5). |
| Epsilon = 0 | Division by zero possible if variance is 0. Not recommended but should not crash -- rsqrt(0) = inf, which propagates to output. |
| bfloat16 input | Use bfloat16 CBs and tile sizes. Consider enabling fp32_dest_acc_en for precision. |
| float32 input | Use float32 CBs and tile sizes (4 bytes/element). fp32_dest_acc_en = true. L1 usage doubles. |
| High-rank tensor (e.g., 5D) | Flatten all dims except last into H (num_sticks). Normalization is still row-wise. |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-generic-op-builder** | API Specification, Input Tensor Requirements, Output Tensor Specification, Circular Buffer Requirements, Work Distribution, Compile-Time Arguments, Runtime Arguments |
| **ttnn-kernel-designer** | Circular Buffer Requirements, Data Flow, Compute Access, Mathematical Definition, Component Sources |
| **ttnn-kernel-dataflow** | Kernel Data Movement, Memory Access Patterns, Reader/Writer sections |
| **ttnn-kernel-compute** | Compute Access, Mathematical Definition, Component Sources |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Input not on device -> error containing "must be on device"
- Input not ROW_MAJOR -> error containing "must be in ROW_MAJOR layout"
- Input dtype not bfloat16 or float32 -> error containing "unsupported dtype"
- Input last dim not multiple of 32 -> error containing "must be a multiple of 32"
- Input second-to-last dim not multiple of 32 -> error containing "must be a multiple of 32"
- Gamma/beta dtype mismatch -> error containing "dtype must match"
- Gamma/beta shape mismatch -> error containing "last dim must match"

### Shape Behavior
- Output shape = input shape for all valid inputs
- Output layout = ROW_MAJOR
- Output dtype = input dtype

### Functional Behavior
- **Single tile** (32x32): output matches `torch.nn.functional.layer_norm(input, [32], gamma, beta, eps)` with PCC > 0.99
- **Multi-tile width** (32x1024): output matches PyTorch layer_norm with PCC > 0.99
- **Multi-tile height** (128x256): output matches PyTorch layer_norm with PCC > 0.99
- **Batch dimensions** (2x3x64x128): output matches PyTorch layer_norm with PCC > 0.99
- **No gamma/beta**: output matches PyTorch layer_norm with weight=None, bias=None
- **Gamma only**: output matches PyTorch layer_norm with bias=None
- **bfloat16 precision**: PCC > 0.99 vs PyTorch float32 reference
- **float32 precision**: PCC > 0.999 vs PyTorch float32 reference

### Test Parameters (pytest parametrize)
```python
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 32, 1024), (1, 1, 128, 256), (2, 3, 64, 128)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("has_gamma_beta", [True, False])
@pytest.mark.parametrize("epsilon", [1e-5])
```

## Open Questions

1. **L1 capacity validation**: Should the operation raise an error at program creation time if the total CB allocation exceeds L1 capacity? If so, what is the maximum supported W for bfloat16 and float32?
2. **fp32_dest_acc_en for bfloat16**: Should bfloat16 inputs always use fp32 destination accumulation for better precision, or should this be configurable?
3. **Gamma/beta tilize strategy**: The reader must present 32 identical rows of gamma/beta to the tilize hardware. Should this be done by:
   (a) Reading the single gamma row and memcpy-ing it 32 times in L1, or
   (b) Using a specialized gamma reader that fills the CB differently?
   Option (a) is simpler but uses more reader cycles. The kernel-designer and kernel-writer should determine the best approach.

## References
- Reference analyses:
  - `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/agent_logs/tilize_single_core_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/agent_logs/softmax_general_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/ttnn/operations/layer_norm_rm/agent_logs/untilize_single_core_analysis.md`
- DeepWiki queries:
  - "How does the reduce scaler work in REDUCE_ROW operations?" -- Confirmed scaler is applied automatically, must be bfloat16 packed.
  - "How do tilize and untilize work as phases within a larger compute kernel?" -- Confirmed single kernel can sequence tilize, math ops, and untilize with proper init/uninit calls.
- Kernel helper libraries:
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` -- Unified reduce with policies
  - `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` -- Unified binary ops (add, sub, mul, square) with broadcast dimensions
  - `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` -- Unified tilize with init/uninit modes
  - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` -- Unified untilize with init/uninit modes
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` -- generate_reduce_scaler
  - `ttnn/cpp/ttnn/kernel_lib/scalar_helpers.hpp` -- generate_bcast_scalar_bfloat16, generate_bcast_scalar
