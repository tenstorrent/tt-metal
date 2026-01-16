# LayerNorm Fused RM Functional Specification

## Overview
- **Operation Name**: `layernorm_fused_rm`
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operation(s)**: tilize, layernorm, untilize
- **Reference Analysis/Analyses**:
  - `tilize_analysis.md` (role: input_stage)
  - `layernorm_analysis.md` (role: compute_stage)
  - `untilize_analysis.md` (role: output_stage)

## Mathematical Definition

### Formula
```
y[..., h, w] = gamma[w] * ((x[..., h, w] - mean_h) / sqrt(var_h + epsilon)) + beta[w]

where:
  mean_h = (1/W) * sum(x[..., h, :])        // mean computed across last dimension
  var_h  = (1/W) * sum((x[..., h, :] - mean_h)^2)  // variance computed across last dimension
```

### Semantic Description
LayerNorm normalizes each row of the input tensor independently by:
1. Computing the mean across the last (width) dimension for each row
2. Computing the variance across the last dimension for each row
3. Centering each element by subtracting the row mean
4. Scaling by the inverse standard deviation (1/sqrt(variance + epsilon))
5. Applying learnable affine transform: multiply by gamma, add beta

The key difference from standard layernorm is that this operation accepts **row-major** input and produces **row-major** output, internally performing tilize/untilize to enable tiled compute.

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | - | - | Input tensor in ROW_MAJOR layout |
| gamma | Tensor | Yes | - | - | Scale weights, shape [1, ..., 1, W] |
| beta | Tensor | Yes | - | - | Bias weights, shape [1, ..., 1, W] |
| epsilon | float | No | > 0 | 1e-5 | Small constant for numerical stability |
| memory_config | MemoryConfig | No | - | input.memory_config() | Output memory configuration |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Device | Must be on device | "Input must be allocated on device" |
| Layout | ROW_MAJOR | "Input must be in ROW_MAJOR layout" |
| Memory | INTERLEAVED | "Input must be in INTERLEAVED memory" |
| Rank | >= 2 | "Input must have at least 2 dimensions" |
| Dtype | BFLOAT16 | "Input must be BFLOAT16" |
| Width Alignment | W % TILE_WIDTH == 0 | "Width must be multiple of 32" |
| Height Alignment | H % TILE_HEIGHT == 0 | "Height must be multiple of 32" |

### Gamma/Beta Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Device | Must be on device | "Gamma/Beta must be allocated on device" |
| Layout | ROW_MAJOR | "Gamma/Beta must be in ROW_MAJOR layout" |
| Memory | INTERLEAVED | "Gamma/Beta must be in INTERLEAVED memory" |
| Shape | Broadcastable to [..., 1, W] | "Gamma/Beta shape must match last dimension" |
| Dtype | BFLOAT16 | "Gamma/Beta must be BFLOAT16" |

### Output Tensor Specification

| Property | Value |
|----------|-------|
| Shape | Same as input: [..., H, W] |
| Layout | ROW_MAJOR |
| Memory | INTERLEAVED |
| Dtype | Same as input |
| Buffer Type | DRAM |

## Component Sources (Hybrid Mode)

This operation is composed from multiple references:

### Input Stage (from tilize_analysis.md)

| Component | Source | Modifications |
|-----------|--------|---------------|
| Reader kernel | tilize.reader | Extended to also read gamma/beta in RM format, generate scaler and epsilon tiles |
| CB_in (c_0) | tilize.CB_in0 | Same purpose: RM input sticks for tilization |
| CB_gamma_rm (c_4) | New | RM gamma sticks before tilization |
| CB_beta_rm (c_5) | New | RM beta sticks before tilization |
| Tilize pattern | tilize.compute | Used in fused compute kernel for input tilization |

### Compute Stage (from layernorm_analysis.md)

| Component | Source | Modifications |
|-----------|--------|---------------|
| Statistics computation | layernorm.compute | Row-wise mean/variance computation (unchanged) |
| Centering (x - mean) | layernorm.compute | bcast_cols subtraction (unchanged) |
| Normalization | layernorm.compute | rsqrt, mul_bcast_cols (unchanged) |
| Gamma/Beta application | layernorm.compute | mul_bcast_rows, add_bcast_rows (unchanged) |
| CB_scaler (c_2) | layernorm.CB_scaler | 1/W scaler tile |
| CB_eps (c_3) | layernorm.CB_eps | Epsilon scalar tile |
| CB_mean (c_25) | layernorm.CB_ex | Mean result (was c_18) |
| CB_var (c_26) | layernorm.CB_ex2 | Variance result (was c_19) |
| CB_centered (c_24) | layernorm.CB_xmm | Centered values (x - mean) |
| CB_invstd (c_27) | layernorm.CB_ex2pe | 1/sqrt(var+eps) (was c_21) |

### Output Stage (from untilize_analysis.md)

| Component | Source | Modifications |
|-----------|--------|---------------|
| Untilize pattern | untilize.compute | Used in fused compute for output untilization |
| Writer kernel | untilize.writer | Stick-based writes via TensorAccessor |
| CB_out (c_16) | untilize.CB_output | RM output sticks |

### Interface Compatibility

| Interface | Component A | Component B | Format A | Format B | Compatible? |
|-----------|------------|-------------|----------|----------|-------------|
| Reader -> Compute (input) | tilize.reader | fused.compute (tilize phase) | RM sticks | RM sticks | YES |
| Reader -> Compute (gamma) | reader.gamma_read | fused.compute (tilize_gamma) | RM sticks | RM sticks | YES |
| Reader -> Compute (beta) | reader.beta_read | fused.compute (tilize_beta) | RM sticks | RM sticks | YES |
| Compute (tilize) -> Compute (layernorm) | tilize_block | layernorm_compute | Tiled | Tiled | YES |
| Compute (layernorm) -> Compute (untilize) | layernorm_compute | untilize | Tiled | Tiled | YES |
| Compute -> Writer | fused.compute (untilize phase) | untilize.writer | RM sticks | RM sticks | YES |

### CB ID Resolution

| Logical CB | Source Ref | Original ID | Final ID | Notes |
|------------|-----------|-------------|----------|-------|
| CB_in_rm | tilize | c_0 | c_0 | RM input sticks |
| CB_in_tiled | tilize | c_16 | c_1 | Tiled input (double buffered) |
| CB_scaler | layernorm | c_2 | c_2 | 1/W scaler |
| CB_eps | layernorm | c_3 | c_3 | Epsilon scalar |
| CB_gamma_rm | New | - | c_4 | RM gamma sticks |
| CB_beta_rm | New | - | c_5 | RM beta sticks |
| CB_gamma_tiled | layernorm | c_5 | c_6 | Tiled gamma (PERSISTENT) |
| CB_beta_tiled | layernorm | c_6 | c_7 | Tiled beta (PERSISTENT) |
| CB_out_rm | untilize | c_16 | c_16 | RM output sticks |
| CB_centered | layernorm | c_24 | c_24 | x - mean |
| CB_mean | layernorm | c_18 | c_25 | Mean result |
| CB_var | layernorm | c_19 | c_26 | Variance result |
| CB_invstd | layernorm | c_21 | c_27 | 1/sqrt(var+eps) |

## Design Decisions

### Decision 1: Fused Single-Kernel Compute
- **Choice**: Single compute kernel performing tilize -> layernorm -> untilize
- **Rationale**: Eliminates intermediate DRAM writes, maximizes L1 residency
- **Alternatives Considered**: Separate tilize/compute/untilize operations (standard pipeline)
- **Tradeoffs**: More complex kernel logic, but ~3x reduction in DRAM bandwidth

### Decision 2: Persistent Gamma/Beta CBs
- **Choice**: Tilize gamma/beta once at start, keep in CB without popping for entire program
- **Rationale**: Gamma/beta are 1D tensors reused across all rows; no need to re-read
- **Alternatives Considered**: Re-read per row (wastes bandwidth)
- **Tradeoffs**: Requires Wt tiles of CB capacity for each, but avoids repeated DRAM access

### Decision 3: Row-Major Input/Output
- **Choice**: Accept ROW_MAJOR layout directly instead of requiring pre-tilized input
- **Rationale**: Many real-world use cases have RM data; avoids user-side tilize/untilize
- **Alternatives Considered**: Require TILE_LAYOUT input (forces user to tilize)
- **Tradeoffs**: Additional compute work (tilize/untilize) traded for convenience and bandwidth savings

### Decision 4: Double-Buffered Tiled Input CB
- **Choice**: CB c_1 (tiled input) is double-buffered (2*Wt tiles)
- **Rationale**: Allows overlap between tilizing next row and computing current row
- **Alternatives Considered**: Single-buffered (simpler but no overlap)
- **Tradeoffs**: 2x CB memory for input tiles, enables better pipelining

### Decision 5: Single-Core Initial Implementation
- **Choice**: Start with single-core implementation before multi-core
- **Rationale**: Simplifies debugging, establishes correctness baseline
- **Alternatives Considered**: Multi-core from start
- **Tradeoffs**: Lower initial throughput, but faster time to correctness

### Decision 6: Standard Accumulation (No FP32 Mode)
- **Choice**: Use standard bfloat16 accumulation for statistics
- **Rationale**: Sufficient precision for most use cases, simpler CB sizing
- **Alternatives Considered**: FP32 accumulation for higher precision
- **Tradeoffs**: Potential precision loss on large width tensors

## Work Distribution

### Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Complete tile row |
| **Unit size** | Wt tiles (all tiles across width dimension) |
| **Total units** | Ht = total_height / TILE_HEIGHT |
| **Loop structure** | Outer: Ht rows; Inner: Wt tiles per row processed as block |

A single work unit is one tile row. Each row must be processed completely before moving to the next because:
1. LayerNorm computes row-wise statistics (mean, variance)
2. All Wt tiles must be available to compute accurate row statistics
3. The untilize phase produces 32 RM sticks from one tile row

### Parallelization Strategy
- **Grid**: Single core initially; future: 1D grid with `split_work_to_cores(grid_size, Ht)`
- **Work per core**: `num_tile_rows_per_core` complete rows
- **Load balancing**: Equal distribution with cliff core for remainder

### Per-Row Processing (Ht iterations)
```
For each tile row h in [0, Ht):
    1. Tilize input row: c_0 (RM) -> c_1 (tiled)
    2. Compute mean across row
    3. Center values (x - mean)
    4. Square centered values
    5. Compute variance
    6. Compute inverse std: 1/sqrt(var + eps)
    7. Normalize: centered * inv_std
    8. Apply gamma: normalized * gamma (bcast_rows)
    9. Apply beta: scaled + beta (bcast_rows)
    10. Untilize result: tiled -> c_16 (RM)
```

## Data Flow

### High-Level Flow
```
DRAM (RM)           L1 Circular Buffers                    DRAM (RM)
    |                        |                                 ^
    | sticks                 |                                 |
    v                        v                                 |
[input] --NOC0--> [c_0 RM] --tilize--> [c_1 tiled]            |
                                           |                   |
                                           v                   |
                              [LayerNorm compute stages]       |
                                           |                   |
                                           v                   |
                              [c_result tiled] --untilize--> [c_16 RM] --NOC1--> [output]

[gamma] --NOC0--> [c_4 RM] --tilize--> [c_6 tiled PERSISTENT]
[beta]  --NOC0--> [c_5 RM] --tilize--> [c_7 tiled PERSISTENT]
```

### Kernel Data Movement

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (input RM, gamma RM, beta RM) | c_0, c_2, c_3, c_4, c_5 | Read RM sticks, generate scaler/eps tiles |
| compute | RISCV_2 | N/A | c_0, c_2, c_3, c_4, c_5 | c_16 | Tilize, layernorm, untilize |
| writer | RISCV_1 | NOC1 | c_16 | DRAM (output RM) | Write RM sticks |

### Circular Buffer Requirements

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in_rm | Input RM sticks | Wt tiles | Wt tiles | Single | Reader | Compute | Row |
| c_1 | cb_in_tiled | Tilized input | 2*Wt tiles | Wt tiles | Double | Compute | Compute | Row |
| c_2 | cb_scaler | Reduce scaler (1/W) | 2 tiles | 1 tile | Single | Reader | Compute | Program |
| c_3 | cb_eps | Epsilon tile | 1 tile | 1 tile | Single | Reader | Compute | Program |
| c_4 | cb_gamma_rm | Gamma RM sticks | Wt tiles | Wt tiles | Single | Reader | Compute | Program |
| c_5 | cb_beta_rm | Beta RM sticks | Wt tiles | Wt tiles | Single | Reader | Compute | Program |
| c_6 | cb_gamma_tiled | Tilized gamma | Wt tiles | Wt tiles | Single | Compute | Compute | Program (PERSISTENT) |
| c_7 | cb_beta_tiled | Tilized beta | Wt tiles | Wt tiles | Single | Compute | Compute | Program (PERSISTENT) |
| c_16 | cb_out_rm | Output RM sticks | Wt tiles | Wt tiles | Single | Compute | Writer | Row |
| c_24 | cb_centered | x - mean | Wt tiles | Wt tiles | Single | Compute | Compute | Row |
| c_25 | cb_mean | Mean tile | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_26 | cb_var | Variance tile | 1 tile | 1 tile | Single | Compute | Compute | Row |
| c_27 | cb_invstd | 1/sqrt(var+eps) | 1 tile | 1 tile | Single | Compute | Compute | Row |

### CB Sizing Details

```cpp
// Wt = width / TILE_WIDTH (tiles per row)
uint32_t Wt = width / TILE_WIDTH;
uint32_t tile_size = get_tile_size(DataType::BFLOAT16);  // 2048 bytes

// Input/output CBs: one complete row
cb_in_rm_size = Wt * tile_size;      // c_0
cb_in_tiled_size = 2 * Wt * tile_size; // c_1 (double buffered)
cb_out_rm_size = Wt * tile_size;     // c_16

// Gamma/beta CBs: one complete row (persistent)
cb_gamma_rm_size = Wt * tile_size;   // c_4
cb_beta_rm_size = Wt * tile_size;    // c_5
cb_gamma_tiled_size = Wt * tile_size; // c_6
cb_beta_tiled_size = Wt * tile_size;  // c_7

// Scalar CBs
cb_scaler_size = 2 * tile_size;      // c_2
cb_eps_size = tile_size;             // c_3

// Intermediate CBs
cb_centered_size = Wt * tile_size;   // c_24
cb_mean_size = tile_size;            // c_25
cb_var_size = tile_size;             // c_26
cb_invstd_size = tile_size;          // c_27
```

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**Input Tensor (RM sticks):**
- Read Wt tiles worth of sticks per row (32 sticks * stick_size bytes)
- Sequential stick IDs using TensorAccessor
- Pattern: `for stick in [row_start..row_start+32): noc_async_read(stick_addr)`

**Gamma Tensor (RM sticks):**
- Read once at program start, Wt tiles of data
- Sequential read of all gamma sticks
- Never re-read (persistent in CB)

**Beta Tensor (RM sticks):**
- Read once at program start, Wt tiles of data
- Sequential read of all beta sticks
- Never re-read (persistent in CB)

**Scaler Tile:**
- Generate in L1: fill with 1.0/(W) for reduction
- Uses `generate_reduce_scaler` pattern

**Epsilon Tile:**
- Generate in L1: column broadcast scalar
- Uses `generate_bcast_col_scalar` pattern

### RISCV_1 ("writer" / NCRISC) Access

**Output Tensor (RM sticks):**
- Write Wt tiles worth of sticks per row (32 sticks * stick_size bytes)
- Sequential stick IDs using TensorAccessor
- Pattern: `for stick in [row_start..row_start+32): noc_async_write(stick_addr)`

### Compute Access

**Per-Row Processing:**
1. **Tilize phase**: cb_wait_front(c_0, Wt) -> tilize_block -> cb_push_back(c_1, Wt)
2. **Mean computation**: cb_wait_front(c_1, Wt) -> reduce_row -> cb_push_back(c_25, 1)
3. **Center values**: sub_tiles_bcast_cols(c_1, c_25) -> cb_push_back(c_24, Wt)
4. **Square**: mul_tiles(c_24, c_24) -> intermediate
5. **Variance**: reduce_row(squared) -> cb_push_back(c_26, 1)
6. **Inv std**: add_tiles(c_26, c_3) + rsqrt -> cb_push_back(c_27, 1)
7. **Normalize**: mul_tiles_bcast_cols(c_24, c_27) -> intermediate
8. **Apply gamma**: mul_tiles_bcast_rows(normalized, c_6) -> intermediate
9. **Apply beta**: add_tiles_bcast_rows(scaled, c_7) -> final
10. **Untilize**: untilize_block -> cb_push_back(c_16, Wt)

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Size of one RM stick in bytes (W * element_size) |
| 1 | Wt | uint32_t | Tiles per row |
| 2+ | src_args | TensorAccessorArgs | Input tensor accessor config |
| N+ | gamma_args | TensorAccessorArgs | Gamma tensor accessor config |
| M+ | beta_args | TensorAccessorArgs | Beta tensor accessor config |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row |
| 1 | Ht | uint32_t | Tile rows per core |
| 2 | W | uint32_t | Logical width (for partial tile handling) |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output CB ID (c_16) |
| 1 | stick_size | uint32_t | Size of one RM stick in bytes |
| 2 | tile_height | uint32_t | TILE_HEIGHT (32) |
| 3 | Wt | uint32_t | Tiles per row |
| 4+ | dst_args | TensorAccessorArgs | Output tensor accessor config |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input tensor DRAM address |
| 1 | gamma_addr | uint32_t | Gamma tensor DRAM address |
| 2 | beta_addr | uint32_t | Beta tensor DRAM address |
| 3 | num_tile_rows | uint32_t | Number of tile rows for this core |
| 4 | start_stick_id | uint32_t | Starting stick index for this core |
| 5 | packed_scaler | uint32_t | Packed bfloat16 (1.0/W) |
| 6 | epsilon | uint32_t | Epsilon value (bit_cast float to uint32) |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tile_rows | uint32_t | Number of tile rows for this core |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output tensor DRAM address |
| 1 | num_tile_rows | uint32_t | Number of tile rows for this core |
| 2 | start_stick_id | uint32_t | Starting output stick index for this core |

## Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile (W=32, H=32) | Process as single row; statistics computed over 32 elements |
| Wide tensor (W=4096) | Wt=128 tiles per row; larger CB allocation |
| Tall tensor (H=1024) | Ht=32 tile rows; more iterations per core |
| Minimum valid (W=32, H=32) | Single tile row, single iteration |
| Large input (H=8192, W=2048) | May require multi-core distribution |
| Epsilon near zero | Use eps=1e-5 default minimum for stability |
| Uniform input row | Mean equals all elements, variance=0, output=(0*gamma)+beta |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-operation-scaffolder** | API Specification, Input Tensor Requirements, Output Tensor Specification |
| **ttnn-factory-builder** | Circular Buffer Requirements, Work Distribution, Data Flow, Component Sources |
| **ttnn-kernel-designer** | Kernel Data Movement, Memory Access Patterns, Component Sources |
| **ttnn-kernel-writer** | Compute Access, Mathematical Definition, Component Sources |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Wrong layout (not ROW_MAJOR) -> error containing "must be in ROW_MAJOR layout"
- Wrong memory (not INTERLEAVED) -> error containing "must be in INTERLEAVED memory"
- Unsupported dtype (not BFLOAT16) -> error containing "must be BFLOAT16"
- Mismatched gamma/beta shape -> error containing "shape must match last dimension"
- Input not on device -> error containing "must be allocated on device"
- Non-tile-aligned width -> error containing "Width must be multiple of 32"
- Non-tile-aligned height -> error containing "Height must be multiple of 32"

### Shape Behavior
- Output shape equals input shape
- Output layout is ROW_MAJOR
- Output memory is INTERLEAVED

### Functional Behavior
- Single tile (32x32): output matches PyTorch `torch.nn.functional.layer_norm()`
- Multi-tile (64x64, 128x128): output matches PyTorch within tolerance
- Wide tensor (32x1024): verify row-wise statistics are correct
- Tall tensor (1024x32): verify each row normalized independently
- Numerical accuracy: PCC > 0.999 vs PyTorch reference

### Compute Helper Usage
The compute kernel should use these kernel_lib helpers:
- `compute_kernel_lib::tilize()` for input tilization
- `compute_kernel_lib::reduce<SUM, REDUCE_ROW>()` for mean/variance computation
- `compute_kernel_lib::sub<BroadcastDim::COL>()` for centering
- `compute_kernel_lib::mul<BroadcastDim::COL>()` for normalization
- `compute_kernel_lib::mul<BroadcastDim::ROW>()` for gamma application
- `compute_kernel_lib::add<BroadcastDim::ROW>()` for beta application
- `compute_kernel_lib::untilize()` for output untilization

Raw LLK (no helper):
- `square_tile()` or `mul_tiles()` for squaring
- `add_tiles()` for epsilon addition
- `rsqrt_tile()` for inverse square root

## Open Questions

1. **Large tensor mode**: Should we implement reduced CB sizes for very wide tensors that exceed L1 capacity?
   - Current spec assumes Wt tiles fit in L1 with all CBs
   - May need streaming approach for W > 4096

2. **FP32 accumulation**: Should we add optional FP32 accumulation mode for higher precision statistics?
   - Would affect CB sizing (Float32 vs Float16_b for intermediates)
   - Adds complexity but improves accuracy for large W

3. **Partial width handling**: Should we support non-tile-aligned widths with padding?
   - Current spec requires W % 32 == 0
   - Could add partial tile scaler like layernorm does

4. **Multi-core scaling**: What is the minimum tensor size to benefit from multi-core?
   - Single-core simpler for debugging
   - Multi-core needed for production throughput

## References
- Reference analyses:
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/agent_logs/tilize_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/agent_logs/layernorm_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm_fused_rm/agent_logs/untilize_analysis.md`
- DeepWiki queries: (to be consulted by implementation agents)
  - Row-major to tiled conversion patterns
  - Broadcast operation semantics (bcast_cols, bcast_rows)
  - Reduce operation variants
- Documentation consulted:
  - `tech_reports/tensor_layouts/tensor_layouts.md`
  - `tech_reports/tensor_accessor/tensor_accessor.md`
  - `ttnn/cpp/ttnn/kernel_lib/` helper APIs
