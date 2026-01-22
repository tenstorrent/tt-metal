# standardize_w_rm Functional Specification

## Overview
- **Operation Name**: standardize_w_rm
- **Category**: reduction (normalization variant)
- **Planning Mode**: Derivative
- **Reference Operation**: variance_w_rm
- **Reference Analysis**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/references/variance_w_rm_analysis.md`

## Mathematical Definition

### Formula
```
mean[..., 0] = (1/W) * sum(input[..., j] for j in range(W))
centralized[..., j] = input[..., j] - mean[..., 0]  for all j
variance[..., 0] = (1/W) * sum(centralized[..., j]^2 for j in range(W))
rsqrt_var[..., 0] = rsqrt(variance[..., 0] + epsilon)
output[..., j] = centralized[..., j] * rsqrt_var[..., 0]  for all j
```

### Semantic Description
The standardize_w_rm operation performs row-wise standardization (z-score normalization) on row-major interleaved tensors. For each row of width W, it:
1. Computes the mean across the row
2. Subtracts the mean from each element (centralization)
3. Computes the variance (mean of squared deviations)
4. Adds epsilon for numerical stability
5. Computes the reciprocal square root of (variance + epsilon)
6. Multiplies the centralized values by this factor to produce standardized output

The output has the same shape as the input (unlike variance_w_rm which reduces to width 1).

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | at least 2D | - | Input tensor in ROW_MAJOR layout |
| epsilon | float | No | > 0 | 1e-5 | Small constant for numerical stability |
| memory_config | MemoryConfig | No | - | input.memory_config() | Output memory configuration |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | Must be at least 2D | "must be at least 2D" |
| Layout | ROW_MAJOR | "must be in ROW_MAJOR layout" |
| Memory layout | INTERLEAVED | "must have INTERLEAVED memory layout" |
| Buffer type | DRAM | "must be on DRAM" |
| Dtype | BFLOAT16 or FLOAT32 | "unsupported dtype" |
| Device | Must be on device | "must be on device" |
| Width | > 0 | "width must be positive" |

### Output Tensor Specification

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input: `[..., H, W]` |
| **Padded shape** | `[..., ceil(H/32)*32, ceil(W/32)*32]` |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or as specified in memory_config) |
| **Data type** | Same as input |

**Shape formula**: `output_shape = input_shape` (no reduction)

## Comparison with Reference Operation

### What Can Be Reused
| Aspect | Reuse Status | Notes |
|--------|--------------|-------|
| Reader kernel structure | Reuse with modification | Add epsilon tile generation |
| Tilize phase (Phase 1) | Fully reusable | No changes needed |
| Mean reduce (Phase 2) | Fully reusable | PERSISTENT mode preserved |
| Centralize phase (Phase 3) | Reusable with modification | CB_4 must also be PERSISTENT |
| Square phase (Phase 4) | Fully reusable | No changes needed |
| Variance reduce (Phase 5) | Fully reusable | STREAMING mode preserved |
| Untilize phase (Phase 6) | Reusable with modification | Operate on Wt tiles, not 1 |
| Writer kernel structure | Reuse with modification | Write full-width sticks |

### Key Differences

| Aspect | variance_w_rm | standardize_w_rm | Implementation Impact |
|--------|---------------|------------------|----------------------|
| Output shape | `[..., H, 1]` padded to `[..., H, 32]` | `[..., H, W]` (same as input) | Writer outputs W elements per stick, not 32 |
| Output tiles per tile-row | 1 tile | Wt tiles | CB_16 must hold Wt tiles |
| CB_4 lifetime | Block (consumed by square) | PERSISTENT (needed for final multiply) | CB_4 tiles persist through phases 5-8 |
| Compute phases | 6 phases | 9 phases | Add: add_eps, rsqrt, broadcast_mul |
| Epsilon parameter | None | Required (default 1e-5) | New runtime arg, new CB for epsilon |
| CB count | 8 CBs (c_0 to c_6, c_16) | 10 CBs | Add c_7 (epsilon), c_8 (rsqrt result) |

## Design Decisions

### Decision 1: Epsilon Storage
- **Choice**: Generate epsilon tile in reader kernel using `generate_bcast19_scalar` pattern
- **Rationale**: Same proven pattern used for 1/W scaler. Epsilon is a scalar constant that can be packed into a broadcast tile.
- **Alternatives Considered**:
  - Embed epsilon as compile-time arg (rejected: epsilon varies per call)
  - Pass epsilon as raw value and compute in kernel (rejected: more complex)
- **Tradeoffs**: Requires additional CB (c_7) but enables runtime-configurable epsilon

### Decision 2: Centralized Tile Persistence
- **Choice**: CB_4 (centralized tiles) uses PERSISTENT retention through rsqrt computation
- **Rationale**: Centralized values are needed for the final broadcast multiply. Re-reading and re-tilizing would double memory bandwidth.
- **Alternatives Considered**:
  - Re-read input after rsqrt (rejected: doubles memory bandwidth)
  - Duplicate centralized to separate CB (rejected: wastes L1 memory)
- **Tradeoffs**: CB_4 cannot be reclaimed until final multiply is complete, constraining L1 usage

### Decision 3: Output CB Sizing
- **Choice**: CB_16 sized to hold Wt tiles per tile-row (not 1 tile)
- **Rationale**: Standardize outputs full-width tensor. Untilize produces Wt tiles that must be written.
- **Alternatives Considered**:
  - Stream tiles individually (rejected: serializes write, reduces bandwidth)
  - Use separate intermediate CB (rejected: unnecessary complexity)
- **Tradeoffs**: Larger CB_16 footprint (Wt tiles vs 1 tile)

### Decision 4: Add-Epsilon Implementation
- **Choice**: Use `add_binary_tile` to add epsilon tile (from c_7) to variance tile (from c_6)
- **Rationale**: Follows batch_norm pattern for variance + epsilon. Clear and maintainable.
- **Alternatives Considered**:
  - Use helper library add() (acceptable alternative if available for SCALAR broadcast)
  - Inline add using SFPU (rejected: non-standard, harder to maintain)
- **Tradeoffs**: Requires copy_tile to DST, init/uninit overhead

### Decision 5: Rsqrt Implementation
- **Choice**: Use `rsqrt_tile` directly on DST register after add
- **Rationale**: Standard SFPU operation with good accuracy. Used by batch_norm, groupnorm.
- **Alternatives Considered**:
  - sqrt then reciprocal (rejected: two operations, potential precision loss)
  - Fast approximate mode (could be future parameter)
- **Tradeoffs**: None significant, well-tested approach

### Decision 6: Final Multiply Broadcast Dimension
- **Choice**: Use BroadcastDim::COL for rsqrt * centralized multiplication
- **Rationale**: Rsqrt produces 1 tile per tile-row (column-shaped after REDUCE_ROW), same as mean in Phase 3.
- **Alternatives Considered**: None (this is the correct dimension)
- **Tradeoffs**: None

## Work Distribution

### Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile-row |
| **Unit size** | Wt tiles input, Wt tiles output (one tile-row = 32 sticks x W elements) |
| **Total units** | Ht tile-rows |
| **Loop structure** | Outer loop over Ht tile-rows, each iteration processes all 9 phases |

One work unit is a **tile-row**: 32 consecutive row-major sticks that, when tilized, produce Wt tiles. Each tile-row is processed through the complete 9-phase pipeline before moving to the next.

### Parallelization Strategy
- **Grid**: 1 x 1 (single core, same as variance_w_rm)
- **Work per core**: All Ht tile-rows
- **Load balancing**: N/A (single core implementation)

**Note**: Multi-core extension would split Ht tile-rows across cores, with each core processing a subset independently.

## Data Flow

### High-Level Flow
```
DRAM (RM sticks, shape [..., W])
    |
    v  [Reader: read 32 sticks per tile-row, generate scaler and epsilon once]
CB_0 (cb_in_rm): RM sticks (2*Wt pages, double-buffered)
    |
    v  [Compute Phase 1: tilize]
CB_1 (cb_in_tiled): Wt tiled tiles [PERSISTENT for Phase 2-3]
    |
    +---> [Compute Phase 2: reduce PERSISTENT] ---> CB_3 (cb_mean): 1 mean tile
    |                                                       |
    |<------------------------------------------------------+
    |     [Compute Phase 3: broadcast subtract (A=CB_1, B=CB_3)]
    v
CB_4 (cb_centralized): Wt centralized tiles [PERSISTENT through Phase 8]
    |
    v  [Compute Phase 4: square (SQUARE binary op)]
CB_5 (cb_squared): Wt squared tiles
    |
    +---> [Compute Phase 5: reduce STREAMING] ---> CB_6 (cb_variance): 1 variance tile
    |
    |     [Compute Phase 6: add epsilon]
    |     CB_6 (variance) + CB_7 (epsilon) ---> CB_6 (var_eps) [reuse CB_6]
    |
    |     [Compute Phase 7: rsqrt]
    |     rsqrt(CB_6) ---> CB_8 (cb_rsqrt): 1 rsqrt tile
    |
    |<--------------------------------------------------+
    |     [Compute Phase 8: broadcast multiply (A=CB_4, B=CB_8)]
    v
CB_16 (cb_out_rm): Wt RM output tiles
    |
    v  [Compute Phase 9: untilize]
CB_16 (cb_out_rm): 32 RM output sticks per tile (Wt tiles total)
    |
    v  [Writer: write 32 sticks (width W) per tile-row]
DRAM (RM sticks, shape [..., W])
```

### Phase-by-Phase Data Flow

| Phase | Operation | Input CB(s) | Output CB | Description |
|-------|-----------|-------------|-----------|-------------|
| 1 | Tilize | CB_0 (Wt pages) | CB_1 (Wt tiles) | Convert 32 RM sticks to Wt tiles |
| 2 | Reduce (Mean) | CB_1 (Wt tiles), CB_2 (scaler) | CB_3 (1 tile) | REDUCE_ROW with PERSISTENT mode |
| 3 | Broadcast Sub | CB_1 (Wt tiles), CB_3 (1 tile) | CB_4 (Wt tiles) | Centralize: input - mean [CB_4 PERSISTENT] |
| 4 | Square | CB_4 (Wt tiles) | CB_5 (Wt tiles) | Element-wise (x-mean)^2 |
| 5 | Reduce (Variance) | CB_5 (Wt tiles), CB_2 (scaler) | CB_6 (1 tile) | REDUCE_ROW with STREAMING mode |
| 6 | Add Epsilon | CB_6 (1 tile), CB_7 (epsilon) | CB_6 (1 tile) | variance + epsilon [in-place or via DST] |
| 7 | Rsqrt | CB_6 (1 tile) | CB_8 (1 tile) | rsqrt(variance + epsilon) |
| 8 | Broadcast Mul | CB_4 (Wt tiles), CB_8 (1 tile) | CB_16 (Wt tiles) | standardized = centralized * rsqrt |
| 9 | Untilize | CB_16 (Wt tiles) | CB_16 (Wt tiles worth of sticks) | Convert Wt tiles to 32*Wt RM sticks |

### Kernel Data Movement

| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| reader_standardize_w_rm | RISCV_0 (BRISC) | NOC0 | Read RM sticks from DRAM, generate 1/W scaler and epsilon tiles |
| standardize_w_rm_compute | RISCV_2,3,4 | N/A | 9-phase pipeline: tilize, reduce, sub, square, reduce, add, rsqrt, mul, untilize |
| writer_standardize_w_rm | RISCV_1 (NCRISC) | NOC1 | Write full-width RM sticks to DRAM |

## Circular Buffer Requirements

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in_rm | Input RM sticks | 2*Wt tiles | Wt tiles | Double | Reader | Compute (tilize) | Block |
| c_1 | cb_in_tiled | Tiled input (persists for bcast_sub) | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (reduce, sub) | Block |
| c_2 | cb_scaler | Scaler (1/W) for both reduces | 1 tile | 1 tile | Single | Reader | Compute (reduce x2) | Program |
| c_3 | cb_mean_tiled | Mean tile output from first reduce | 1 tile | 1 tile | Single | Compute (reduce1) | Compute (sub) | Block |
| c_4 | cb_centralized | Centralized tiles (input - mean) | Wt tiles | Wt tiles | Single | Compute (sub) | Compute (square, mul) | **PERSISTENT** |
| c_5 | cb_squared | Squared tiles ((x-mean)^2) | Wt tiles | Wt tiles | Single | Compute (square) | Compute (reduce2) | Block |
| c_6 | cb_variance | Variance tile from second reduce | 1 tile | 1 tile | Single | Compute (reduce2) | Compute (add_eps) | Block |
| c_7 | cb_epsilon | Epsilon scalar tile | 1 tile | 1 tile | Single | Reader | Compute (add_eps) | Program |
| c_8 | cb_rsqrt | Rsqrt result tile | 1 tile | 1 tile | Single | Compute (rsqrt) | Compute (mul) | Block |
| c_16 | cb_out_rm | Output RM sticks | 2*Wt tiles | Wt tiles | Double | Compute (mul, untilize) | Writer | Block |

### CB Sizing Rationale

- **CB_0 (2*Wt)**: Double-buffered for reader/compute overlap. Unchanged from variance_w_rm.
- **CB_1 (Wt)**: PERSISTENT mode for reduce + subsequent subtract. Unchanged from variance_w_rm.
- **CB_2 (1)**: Single scaler tile (1/W), program-lifetime. Unchanged from variance_w_rm.
- **CB_3 (1)**: Single mean tile per tile-row. Unchanged from variance_w_rm.
- **CB_4 (Wt)**: **PERSISTENT** - must hold centralized tiles from Phase 3 through Phase 8 for final multiply.
- **CB_5 (Wt)**: Full tile-row of squared data. Unchanged from variance_w_rm.
- **CB_6 (1)**: Single variance tile per tile-row. Reused for variance+epsilon intermediate.
- **CB_7 (1)**: **NEW** - Single epsilon tile, program-lifetime. Generated once by reader.
- **CB_8 (1)**: **NEW** - Single rsqrt result tile per tile-row.
- **CB_16 (2*Wt)**: **RESIZED** - Double-buffered, Wt tiles output per tile-row (was 2 tiles for variance).

### CB Lifetime Summary

| Lifetime | CBs |
|----------|-----|
| Program | c_2 (scaler), c_7 (epsilon) |
| Block | c_0, c_1, c_3, c_5, c_6, c_8, c_16 |
| PERSISTENT (across phases) | c_4 (centralized, phases 3-8) |

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**At program start (once)**:
1. Generate scaler tile (1/W) into CB_2 using `generate_reduce_scaler`
2. Generate epsilon tile into CB_7 using `generate_bcast19_scalar`
3. Push both tiles (they persist for program lifetime)

**Per tile-row loop**:
1. Reserve Wt pages in CB_0
2. Read 32 sticks sequentially from DRAM using TensorAccessor
3. Each stick is W elements wide
4. After all 32 sticks read, push Wt pages to CB_0
5. Wait barrier before push

### RISCV_1 ("writer" / NCRISC) Access

**Per tile-row loop**:
1. Wait for Wt tiles in CB_16
2. Write 32 sticks sequentially to DRAM using TensorAccessor
3. Each stick is W elements wide (same as input, not reduced)
4. After all 32 sticks written, pop Wt tiles from CB_16
5. Wait barrier between stick writes

### Compute Access

**Per tile-row loop**:

| Phase | CB Operations |
|-------|---------------|
| Phase 1 (Tilize) | wait CB_0 Wt pages, push CB_1 Wt tiles |
| Phase 2 (Reduce Mean) | wait CB_1 Wt tiles [PERSISTENT], push CB_3 1 tile |
| Phase 3 (Subtract) | [CB_1 preloaded], pop CB_3, push CB_4 Wt tiles [PERSISTENT] |
| Phase 4 (Square) | [CB_4 preloaded, don't pop], push CB_5 Wt tiles |
| Phase 5 (Reduce Var) | wait/pop CB_5 Wt tiles [STREAMING], push CB_6 1 tile |
| Phase 6 (Add Eps) | wait CB_6, wait CB_7, result to DST |
| Phase 7 (Rsqrt) | result in DST, push CB_8 1 tile |
| Phase 8 (Multiply) | [CB_4 preloaded], pop CB_4 Wt, pop CB_8, push CB_16 Wt tiles |
| Phase 9 (Untilize) | wait CB_16 Wt tiles (for mul output), push CB_16 (untilized sticks) |

**Critical: CB_4 Persistence**
- CB_4 tiles are written in Phase 3 and must NOT be popped until Phase 8
- Phase 4 reads CB_4 without popping (both input/output to square are CB_4->CB_5)
- Phase 8 pops CB_4 after final multiply

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_stick_size_aligned | uint32_t | NoC-aligned stick size in bytes (W * element_size, rounded up) |
| 1 | packed_scaler_value | uint32_t | Two bfloat16 (1/W) packed into uint32 for reduce scaler |
| 2 | packed_epsilon_value | uint32_t | Two bfloat16 (epsilon) packed into uint32 for epsilon tile |
| 3 | Ht | uint32_t | Height in tiles (number of tile-rows to process) |
| 4 | Wt | uint32_t | Width in tiles (tiles per row) |
| 5+ | TensorAccessorArgs | multiple | Input buffer address mode, page size, etc. |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Height in tiles (outer loop count) |
| 1 | Wt | uint32_t | Width in tiles (tiles per row) |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_stick_size_aligned | uint32_t | NoC-aligned output stick size (W * element_size, rounded up) |
| 1 | Ht | uint32_t | Height in tiles (number of tile-rows) |
| 2+ | TensorAccessorArgs | multiple | Output buffer address mode, page size, etc. |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input DRAM buffer address |

### Compute Kernel

None (all parameters are compile-time).

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output DRAM buffer address |

## Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile (W=32, H=32) | Process 1 tile-row with 1 tile |
| Single element per row (W=1) | Centralized = 0, variance = 0, rsqrt(eps), output = 0 |
| Very small epsilon | May cause numerical issues; epsilon > 0 required |
| Large input (H > 10000 sticks) | Sequential tile-row processing; limited by DRAM bandwidth |
| Input with zeros | Mean computed, variance computed, rsqrt applied normally |
| Input all same value | Centralized = 0, variance = 0, rsqrt(eps), output = 0 |
| Width not tile-aligned (W % 32 != 0) | Padded internally to tile boundary; results valid for W elements |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-operation-scaffolder** | API Specification, Input Tensor Requirements, Output Tensor Specification |
| **ttnn-factory-builder** | Circular Buffer Requirements, Work Distribution, Data Flow |
| **ttnn-kernel-dataflow** | Kernel Data Movement, Memory Access Patterns, Compile-Time/Runtime Arguments |
| **ttnn-kernel-compute** | Phase-by-Phase Data Flow, CB Operations, Mathematical Definition |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Wrong tensor rank (< 2D) -> error containing "must be at least 2D"
- Wrong layout (not ROW_MAJOR) -> error containing "must be in ROW_MAJOR layout"
- Wrong memory layout (not INTERLEAVED) -> error containing "must have INTERLEAVED memory layout"
- Unsupported dtype -> error containing "unsupported dtype"
- Tensor not on device -> error containing "must be on device"
- Invalid epsilon (<= 0) -> error containing "epsilon must be positive"

### Shape Behavior
- Output shape == Input shape (no reduction)
- Output dtype == Input dtype
- Output layout == ROW_MAJOR

### Functional Behavior
- Single tile: output matches expected computation (verify mean, variance, standardized)
- Multi-tile: output matches expected computation
- Numerical accuracy vs PyTorch: `output = (input - input.mean(dim=-1, keepdim=True)) / torch.sqrt(input.var(dim=-1, keepdim=True, unbiased=False) + epsilon)`

### Numerical Properties
- Standardized output should have approximately zero mean per row
- Standardized output should have approximately unit variance per row (when original variance > 0)
- Different epsilon values should produce different outputs when variance is small

## Open Questions

None - all design decisions documented above with rationale.

## References

- Reference analysis: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/references/variance_w_rm_analysis.md`
- DeepWiki queries:
  - "rsqrt_tile API in compute kernels" - Found: rsqrt_tile(idst) operates on DST, requires rsqrt_tile_init()
  - "epsilon + rsqrt pattern in normalization" - Found: batch_norm uses add_binary_tile(var, eps) then rsqrt_tile
- Documentation consulted:
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp` - PERSISTENT mode for reduce
  - `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` - BroadcastDim::COL for multiply
  - `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp` - epsilon+rsqrt pattern
