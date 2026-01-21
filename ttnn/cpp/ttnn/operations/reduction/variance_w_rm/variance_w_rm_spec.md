# variance_w_rm Functional Specification

## Overview
- **Operation Name**: variance_w_rm
- **Category**: reduction
- **Planning Mode**: Derivative
- **Reference Operation**: centralize_w_rm
- **Reference Analysis**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/variance_w_rm/references/centralize_w_rm_analysis.md`

## Mathematical Definition

### Formula
```
mean[..., 0] = (1/W) * sum(input[..., j] for j in range(W))
centralized[..., j] = input[..., j] - mean[..., 0]  for all j in range(W)
squared[..., j] = centralized[..., j]^2  for all j in range(W)
variance[..., 0] = (1/W) * sum(squared[..., j] for j in range(W))
```

### Semantic Description
Computes the variance along the W (width) dimension of a row-major tensor. The operation:
1. Computes the mean of each row
2. Subtracts the mean from each element (centralization)
3. Squares each centralized value
4. Computes the mean of squared values (variance)

The output has the last dimension reduced to 1 (logical) but padded to 32 (tile boundary).

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | - | - | Input tensor to compute variance over |
| memory_config | MemoryConfig | No | Any valid | Input's config | Output memory configuration |
| dtype | DataType | No | BFLOAT16, FLOAT32 | Input dtype | Output data type |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | At least 2D | "must have at least 2 dimensions" |
| Layout | ROW_MAJOR | "must be ROW_MAJOR layout" |
| Memory | INTERLEAVED | "must be interleaved" |
| Buffer type | DRAM | "must be on DRAM" |
| Device | Must be on device | "must be on device" |
| Dtype | BFLOAT16 or FLOAT32 | "unsupported dtype, must be BFLOAT16 or FLOAT32" |
| Width | >= 1 | "width must be at least 1" |
| Width alignment | Padded to multiple of 32 | Handled internally via padding |

### Output Tensor Specification

| Property | Specification |
|----------|---------------|
| **Logical shape** | `[..., 1]` (last dimension reduced to 1) |
| **Padded shape** | `[..., 32]` (padded to tile boundary) |
| **Layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or as specified in memory_config) |
| **Dtype** | Same as input (or as specified) |

**Shape calculation**:
- If input shape is `[N, C, H, W]`, output logical shape is `[N, C, H, 1]`
- Output padded shape is `[N, C, H, 32]`
- General: `output_shape[:-1] + [1]` (logical), `output_shape[:-1] + [32]` (padded)

## Comparison with Reference Operation (centralize_w_rm)

### What Can Be Reused
- Work unit definition (tile-row)
- Reader kernel structure (read sticks, generate scaler)
- Writer kernel structure (write reduced output sticks)
- CB_0 (input RM sticks) configuration
- CB_1 (tiled input) PERSISTENT pattern
- CB_2 (scaler) for reduce operations
- Tilize phase (Phase 1)
- First reduce phase (Phase 2 - compute mean for centralization)
- Broadcast subtract phase (Phase 3 - centralization)
- Core distribution strategy (single-core initial implementation)

### Key Differences

| Aspect | centralize_w_rm (Reference) | variance_w_rm (This Operation) | Implementation Impact |
|--------|----------------------------|--------------------------------|----------------------|
| **Output shape** | Same as input `[..., W]` | Reduced `[..., 1]` (padded to 32) | Different output tensor creation, smaller output CB |
| **Number of phases** | 4 phases | 6 phases | Additional compute phases, more CBs needed |
| **Compute operations** | tilize->reduce->sub->untilize | tilize->reduce->sub->**square->reduce**->untilize | Add square and second reduce phases |
| **CB_4 usage** | Centralized tiles (Wt) | Centralized tiles (Wt) - same, but consumed by square | Same sizing, different consumer |
| **CB_5 (NEW)** | N/A | Squared tiles (Wt) | NEW CB for squared values |
| **CB_6 (NEW)** | N/A | Variance tile (1) | NEW CB for variance result |
| **CB_16 sizing** | 2*Wt tiles (full row output) | 2 tiles (single reduced tile output) | Significantly smaller output CB |
| **Output stick size** | W elements | 32 elements (1 tile width, padded) | Different writer stick size |
| **Writer iterations** | 32 sticks per tile-row | 32 sticks per tile-row | Same, but narrower sticks |

## Design Decisions

### Decision 1: 6-Phase Pipeline Architecture
- **Choice**: Extend centralize_w_rm's 4-phase pipeline to 6 phases: tilize -> reduce (mean) -> sub (centralize) -> square -> reduce (variance) -> untilize
- **Rationale**: This naturally builds on the centralize_w_rm pattern. The additional square and reduce phases are straightforward extensions.
- **Alternatives Considered**:
  - Fusing square and reduce into a single phase (rejected: would require custom compute helper)
  - Computing variance directly without centralization (rejected: numerically less stable)
- **Tradeoffs**: More phases means more CB transitions, but each phase uses proven patterns.

### Decision 2: CB_4 Retention Pattern for Square Input
- **Choice**: CB_4 (centralized tiles) uses the same PERSISTENT pattern as CB_1 in centralize_w_rm, retained across square phase for the second reduce
- **Rationale**: The squared tiles in CB_5 need to be consumed by the second reduce. However, we can actually pop CB_4 after squaring since we don't need centralized values after that.
- **Alternatives Considered**: Re-reading centralized tiles from DRAM (rejected: inefficient)
- **Tradeoffs**: Requires Wt tiles in CB_5 to hold squared values, but this is necessary for the reduce.

### Decision 3: Output CB Sizing for Reduced Output
- **Choice**: CB_16 sized for 2 tiles (double-buffered single-tile output) instead of 2*Wt tiles
- **Rationale**: Output is reduced to 1 tile per tile-row (padded to 32 elements), so we only need space for 1 tile at a time per untilize call, double-buffered for compute/writer overlap.
- **Alternatives Considered**: Keeping full Wt sizing (rejected: wasteful)
- **Tradeoffs**: Reduces memory usage significantly for wide tensors.

### Decision 4: Square Operation Implementation
- **Choice**: Use element-wise `mul` (A*A) for squaring
- **Rationale**: No dedicated square helper exists in the kernel library. Element-wise multiply with same input achieves the same result.
- **Alternatives Considered**:
  - Custom square kernel (rejected: unnecessary complexity)
  - Using power function (rejected: mul is more efficient)
- **Tradeoffs**: Requires loading same CB twice or using a self-multiply pattern.

### Decision 5: Variance Reduce Uses Same Scaler as Mean
- **Choice**: Reuse CB_2 scaler tile (1/W) for both reduce operations
- **Rationale**: Both operations compute an average (mean) - first of the raw values, second of squared deviations. The scaler is 1/W in both cases.
- **Alternatives Considered**: Separate scaler CBs (rejected: wasteful duplication)
- **Tradeoffs**: None - this is purely beneficial.

## Work Distribution

### Work Unit Definition
| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile-row |
| **Unit size** | Wt tiles (one tile-row = 32 sticks x W elements = Wt tiles) |
| **Total units** | Ht tile-rows |
| **Loop structure** | Outer loop over Ht tile-rows, each iteration processes all 6 phases |

One work unit is a **tile-row**: 32 consecutive row-major sticks that, when tilized, produce Wt tiles. Each tile-row is processed through the complete 6-phase pipeline before moving to the next.

### Parallelization Strategy
- **Grid**: 1 x 1 (single core, initial implementation)
- **Work per core**: All Ht tile-rows
- **Load balancing**: N/A (single core)

Note: Multi-core distribution would split Ht tile-rows across cores, same as centralize_w_rm.

## Data Flow

### High-Level Flow
```
DRAM (RM sticks, shape [..., W])
    |
    v  [Reader: read 32 sticks per tile-row, generate scaler once]
CB_0 (cb_in_rm): RM sticks (2*Wt pages)
    |
    v  [Compute: Phase 1 - tilize]
CB_1 (cb_in_tiled): Wt tiled tiles [RETAINED for Phase 3]
    |
    +---> [Compute: Phase 2 - reduce PERSISTENT] ---> CB_3 (cb_mean): 1 mean tile
    |                                                       |
    |<------------------------------------------------------+
    |     [Compute: Phase 3 - broadcast subtract (A=CB_1, B=CB_3)]
    v
CB_4 (cb_centralized): Wt centralized tiles
    |
    v  [Compute: Phase 4 - square (mul A*A)]
CB_5 (cb_squared): Wt squared tiles
    |
    +---> [Compute: Phase 5 - reduce STREAMING] ---> CB_6 (cb_variance): 1 variance tile
    |
    v  [Compute: Phase 6 - untilize (1 tile -> 32 sticks)]
CB_16 (cb_out_rm): 32 RM output sticks (width=32, padded)
    |
    v  [Writer: write 32 sticks to reduced output]
DRAM (RM sticks, shape [..., 32] padded)
```

### Kernel Data Movement

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_variance_w_rm | RISCV_0 (BRISC) | NOC0 | DRAM (RM sticks) | CB_0, CB_2 | Read 32 sticks per tile-row, generate 1/W scaler once |
| variance_w_rm_compute | RISCV_2,3,4 | N/A | CB_0,1,2,3,4,5 | CB_1,3,4,5,6,16 | 6-phase: tilize, reduce, sub, square, reduce, untilize |
| writer_variance_w_rm | RISCV_1 (NCRISC) | NOC1 | CB_16 | DRAM (reduced RM sticks) | Write 32 sticks (width=32) per tile-row |

### Circular Buffer Requirements

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in_rm | Input RM sticks | 2*Wt tiles | Wt tiles | Double | Reader | Compute (tilize) | Block |
| c_1 | cb_in_tiled | Tiled input | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (reduce, sub) | Block |
| c_2 | cb_scaler | Scaler (1/W) | 1 tile | 1 tile | Single | Reader | Compute (reduce x2) | Program |
| c_3 | cb_mean | Mean tile | 1 tile | 1 tile | Single | Compute (reduce1) | Compute (sub) | Block |
| c_4 | cb_centralized | Centralized tiles | Wt tiles | Wt tiles | Single | Compute (sub) | Compute (square) | Block |
| c_5 | cb_squared | Squared tiles | Wt tiles | Wt tiles | Single | Compute (square) | Compute (reduce2) | Block |
| c_6 | cb_variance | Variance tile | 1 tile | 1 tile | Single | Compute (reduce2) | Compute (untilize) | Block |
| c_16 | cb_out_rm | Output RM sticks | 2 tiles | 1 tile | Double | Compute (untilize) | Writer | Block |

### CB Sizing Rationale

- **CB_0 (2*Wt)**: Double-buffered for reader/compute overlap, same as centralize_w_rm
- **CB_1 (Wt)**: Single-buffered, must hold full tile-row for PERSISTENT reduce mode
- **CB_2 (1)**: Single scaler tile, reused for both reduce operations
- **CB_3 (1)**: Single mean tile per tile-row
- **CB_4 (Wt)**: Full tile-row of centralized data
- **CB_5 (Wt)**: Full tile-row of squared data
- **CB_6 (1)**: Single variance tile per tile-row
- **CB_16 (2)**: Double-buffered, but only 1 tile output per tile-row (reduced width)

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**Phase 1 (once at start)**: Generate scaler tile
- Write 1/W value to CB_2 using `generate_reduce_scaler(cb_scaler, packed_scaler_value)`
- CB_2 pushed once, consumed repeatedly by both reduce phases

**Phase 2 (per tile-row)**: Read 32 sticks into CB_0
- Reserve Wt pages in CB_0
- For each of 32 sticks:
  - `noc_async_read(accessor.get_noc_addr(stick_id), l1_write_addr, input_stick_size)`
  - Increment l1_write_addr by input_stick_size
  - Increment stick_id
- Barrier and push Wt pages

### RISCV_1 ("writer" / NCRISC) Access

**Per tile-row**: Write 32 output sticks
- Wait for 1 tile in CB_16 (representing 32 sticks of width 32)
- For each of 32 sticks:
  - `noc_async_write(l1_read_addr, accessor.get_noc_addr(stick_id), output_stick_size)`
  - Increment l1_read_addr by output_stick_size
  - Increment stick_id
- Barrier and pop 1 tile

Note: output_stick_size is 32 elements (tile width) * element_size, NOT W * element_size

### Compute Access

**Phase 1 - Tilize**:
- Wait CB_0 (Wt pages of RM data)
- Reserve CB_1 (Wt tiles)
- `tilize_block(CB_0, Wt, CB_1)`
- Push CB_1 (Wt tiles)
- Pop CB_0 (Wt pages)

**Phase 2 - Reduce (Mean)**:
- Wait CB_1 (Wt tiles) - already there
- Wait CB_2 (1 scaler tile) - already there from reader
- Reserve CB_3 (1 tile)
- `reduce<SUM, REDUCE_ROW, PERSISTENT>(CB_1, CB_2, CB_3, TileShape::row(Wt))`
- Push CB_3 (1 tile)
- CB_1 NOT popped (PERSISTENT mode)

**Phase 3 - Broadcast Subtract (Centralize)**:
- CB_1 still has Wt tiles (from PERSISTENT)
- Wait CB_3 (1 mean tile)
- Reserve CB_4 (Wt tiles)
- `sub<COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(CB_1, CB_3, CB_4, BinaryTileShape::row(Wt))`
- Push CB_4 (Wt tiles)
- Pop CB_1 (Wt tiles) - via PopAtEnd policy
- Pop CB_3 (1 tile) - via PopAtEnd policy

**Phase 4 - Square (Element-wise Multiply)**:
- Wait CB_4 (Wt centralized tiles)
- Reserve CB_5 (Wt tiles)
- `mul<NONE, WaitUpfrontPopAtEnd, PreloadedNoWaitNoPop>(CB_4, CB_4, CB_5, BinaryTileShape::row(Wt))`
  - Input A and B are the same CB (self-multiply for square)
  - Or use eltwise pattern: process tile-by-tile with mul_tiles
- Push CB_5 (Wt tiles)
- Pop CB_4 (Wt tiles)

**Phase 5 - Reduce (Variance)**:
- Wait CB_5 (Wt squared tiles)
- Wait CB_2 (1 scaler tile) - still there
- Reserve CB_6 (1 tile)
- `reduce<SUM, REDUCE_ROW, STREAMING>(CB_5, CB_2, CB_6, TileShape::row(Wt))`
  - Use STREAMING mode (pops as it processes) since we don't need CB_5 after
- Push CB_6 (1 tile)
- CB_5 popped by STREAMING mode

**Phase 6 - Untilize**:
- Wait CB_6 (1 variance tile)
- Reserve CB_16 (1 tile = 32 sticks of width 32)
- `untilize<1, CB_6, CB_16>(1)` - untilize 1 tile
- Push CB_16 (1 tile)
- Pop CB_6 (1 tile)

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_stick_size_aligned | uint32_t | NoC-aligned stick size in bytes (W * element_size, rounded up) |
| 1 | packed_scaler_value | uint32_t | Two bfloat16 (1/W) packed into uint32 |
| 2 | Ht | uint32_t | Height in tiles (number of tile-rows) |
| 3 | Wt | uint32_t | Width in tiles (tiles per row) |
| 4+ | TensorAccessorArgs | multiple | Input buffer address mode, page size, etc. |

### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Height in tiles (outer loop count) |
| 1 | Wt | uint32_t | Width in tiles (tiles per row) |

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_stick_size_aligned | uint32_t | NoC-aligned output stick size (32 * element_size, rounded up) |
| 1 | Ht | uint32_t | Height in tiles |
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
|-----------|------------------|
| Single element width (W=1) | Variance is 0 for all rows (no variation) |
| Single row (H=1) | Normal operation, single tile-row processed |
| Large W (many tiles per row) | Normal operation, Wt tiles per tile-row |
| W not multiple of 32 | Input must be padded; padding handled by tilize |
| All identical values in row | Variance is 0 for that row |
| Mixed positive/negative values | Normal operation, squaring makes all positive |
| Very small values | Potential underflow in squared values (float precision) |
| Very large values | Potential overflow in squared values |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-operation-scaffolder** | API Specification, Input Tensor Requirements, Output Tensor Specification |
| **ttnn-factory-builder** | Circular Buffer Requirements, Work Distribution, Data Flow |
| **ttnn-kernel-dataflow** | Kernel Data Movement, Memory Access Patterns (Reader/Writer) |
| **ttnn-kernel-compute** | Compute Access, Mathematical Definition |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Input tensor with rank < 2 -> error containing "must have at least 2 dimensions"
- Input tensor not ROW_MAJOR -> error containing "must be ROW_MAJOR layout"
- Input tensor not INTERLEAVED -> error containing "must be interleaved"
- Input tensor dtype not BFLOAT16/FLOAT32 -> error containing "unsupported dtype"
- Input tensor not on device -> error containing "must be on device"

### Shape Behavior
- Input `[2, 3, 4, 64]` -> Output logical `[2, 3, 4, 1]`, padded `[2, 3, 4, 32]`
- Input `[10, 100]` -> Output logical `[10, 1]`, padded `[10, 32]`
- Input `[32, 32]` (single tile-row) -> Output `[32, 1]` logical, `[32, 32]` padded

### Functional Behavior
- Single tile (32x32): variance matches PyTorch `torch.var(input, dim=-1, keepdim=True, unbiased=False)`
- Multi-tile row: variance matches PyTorch reference
- Constant row (all same value): variance should be 0 (or very close due to float precision)
- Known test case: `input = [[1, 2, 3, 4, 5]]` -> variance = 2.0 (using population variance formula)

### Numerical Accuracy
- Use `unbiased=False` in PyTorch comparison (population variance, N denominator not N-1)
- Tolerance for BFLOAT16: rtol=1e-2, atol=1e-2 (bfloat16 has limited precision)
- Tolerance for FLOAT32: rtol=1e-5, atol=1e-5

## Open Questions

None - proceeding with reasonable assumptions:

1. **Assumption**: Using population variance (divide by N, not N-1) - this matches typical neural network variance calculations (e.g., batch normalization)
2. **Assumption**: Initial implementation is single-core - multi-core can be added later
3. **Assumption**: Square operation uses self-multiply pattern (A*A) - verified this is efficient

## References
- Reference analysis: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/variance_w_rm/references/centralize_w_rm_analysis.md`
- Additional references consulted:
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/reduce_w_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/tilize_analysis.md`
  - `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/reduce_mean_w_rm/references/untilize_analysis.md`
