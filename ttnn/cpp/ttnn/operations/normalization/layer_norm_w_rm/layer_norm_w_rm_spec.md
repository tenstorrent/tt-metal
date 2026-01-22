# layer_norm_w_rm Functional Specification

## Overview
- **Operation Name**: layer_norm_w_rm
- **Category**: normalization
- **Planning Mode**: Derivative
- **Reference Operation**: standardize_w_rm
- **Reference Analysis**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/layer_norm_w_rm/references/standardize_w_rm_analysis.md`

## Mathematical Definition

### Formula
```
mean[..., 0] = (1/W) * sum(input[..., j] for j in range(W))
centralized[..., j] = input[..., j] - mean[..., 0]
variance[..., 0] = (1/W) * sum(centralized[..., j]^2 for j in range(W))
rsqrt_var[..., 0] = rsqrt(variance[..., 0] + epsilon)
standardized[..., j] = centralized[..., j] * rsqrt_var[..., 0]
output[..., j] = standardized[..., j] * gamma[j] + beta[j]
```

### Semantic Description
Layer normalization across the last dimension (W). For each row, compute the mean and variance, standardize the values, then apply learnable affine transformation with gamma (scale) and beta (shift). This extends `standardize_w_rm` by adding the final gamma multiplication and beta addition.

## API Specification

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | At least 2D | - | Input tensor in ROW_MAJOR layout |
| gamma | Tensor | Yes | Shape [1, ..., 1, W] | - | Scale factor tensor, ROW_MAJOR in DRAM |
| beta | Tensor | Yes | Shape [1, ..., 1, W] | - | Bias factor tensor, ROW_MAJOR in DRAM |
| epsilon | float | No | > 0 | 1e-5 | Numerical stability constant |
| output_mem_config | MemoryConfig | No | - | input_tensor.memory_config() | Output memory configuration |

### Input Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Rank | Must be at least 2D | "input tensor must have at least 2 dimensions" |
| Layout | ROW_MAJOR | "input tensor must be in ROW_MAJOR layout" |
| Memory | INTERLEAVED | "input tensor must be interleaved" |
| Device | Must be on device | "input tensor must be on device" |
| Dtype | BFLOAT16 or FLOAT32 | "unsupported dtype: only BFLOAT16 and FLOAT32 are supported" |

### Gamma Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Shape | [1, ..., 1, W] matching input width | "gamma shape must match input width" |
| Layout | ROW_MAJOR | "gamma must be in ROW_MAJOR layout" |
| Memory | INTERLEAVED in DRAM | "gamma must be interleaved in DRAM" |
| Device | Same device as input | "gamma must be on same device as input" |
| Dtype | Same as input | "gamma dtype must match input dtype" |

### Beta Tensor Requirements

| Property | Requirement | Error Message Hint |
|----------|-------------|-------------------|
| Shape | [1, ..., 1, W] matching input width | "beta shape must match input width" |
| Layout | ROW_MAJOR | "beta must be in ROW_MAJOR layout" |
| Memory | INTERLEAVED in DRAM | "beta must be interleaved in DRAM" |
| Device | Same device as input | "beta must be on same device as input" |
| Dtype | Same as input | "beta dtype must match input dtype" |

### Output Tensor Specification
- **Shape**: Same as input: `[..., H, W]`
- **Padded Shape**: `[..., ceil(H/32)*32, ceil(W/32)*32]`
- **Dtype**: Same as input
- **Layout**: ROW_MAJOR
- **Memory**: INTERLEAVED (as specified by output_mem_config)

## Comparison with Reference Operation

### What Can Be Reused
- Work unit granularity (tile-row processing)
- Data flow pattern (9-phase pipeline for standardization)
- Reader kernel structure for input tensor
- Writer kernel structure
- Core distribution strategy (single-core for initial implementation)
- CB indices c_0 through c_9 and c_16
- Phases 1-9 of compute kernel (tilize, mean, subtract, square, variance, add_epsilon+rsqrt, multiply, untilize)

### Key Differences

| Aspect | Reference (standardize_w_rm) | This Operation (layer_norm_w_rm) | Implementation Impact |
|--------|------------------------------|----------------------------------|----------------------|
| **Input tensors** | 1 (input) | 3 (input, gamma, beta) | Reader must read gamma/beta once |
| **CB count** | 11 CBs | 16 CBs | Add c_10, c_11, c_12, c_13, c_14 |
| **Phases** | 9 | 11 | Add Phase 10 (gamma mul) and Phase 11 (beta add) |
| **Gamma/beta handling** | N/A | ROW broadcast, tilize once, reuse | Compute kernel tilizes gamma/beta before main loop |
| **Compute arguments** | Ht, Wt | Ht, Wt | Same (gamma/beta CB indices are constants) |
| **Reader arguments** | input access only | input + gamma + beta access | Add gamma_addr, beta_addr as runtime args |
| **Output** | standardized values | normalized + affine transformed | Phase 11 feeds untilize instead of Phase 8 |

## Design Decisions

### Decision 1: Gamma/Beta Tilization Location
- **Choice**: Tilize gamma and beta in compute kernel, not in reader
- **Rationale**: The reader kernel handles data movement, not format transformation. Tilize is a compute operation and belongs in the compute kernel. This also allows single read + tilize, then reuse across all tile-rows.
- **Alternatives Considered**:
  - Tilize in reader (rejected: tilize is compute, not data movement)
  - Pre-tilize on host (rejected: adds preprocessing burden, gamma/beta may change per call)
- **Tradeoffs**: Compute kernel is slightly more complex but maintains clean separation of concerns.

### Decision 2: Gamma/Beta CB Persistence
- **Choice**: Tilized gamma/beta CBs (c_11, c_13) persist for entire program lifetime
- **Rationale**: Gamma and beta are applied to every tile-row. Reading and tilizing once, then reusing, is more efficient than re-reading per tile-row.
- **Alternatives Considered**: Re-read gamma/beta per tile-row (rejected: wasteful DRAM bandwidth)
- **Tradeoffs**: Uses more L1 space (Wt tiles each for gamma and beta) but saves DRAM bandwidth.

### Decision 3: Broadcast Dimension for Gamma/Beta
- **Choice**: BroadcastDim::ROW for both gamma multiply and beta add
- **Rationale**: Gamma and beta have shape [1, ..., 1, W]. After tilizing, they produce Wt tiles where each tile contains valid data in the top row. ROW broadcast replicates this top row down the column (height) dimension, which is correct for applying gamma/beta to each row of the input.
- **Alternatives Considered**: None - this is the semantically correct choice per DeepWiki and binary_op_helpers documentation.
- **Tradeoffs**: None.

### Decision 4: Separate CBs for Gamma and Beta (RM and Tiled)
- **Choice**: Use 4 separate CBs: c_10 (gamma_rm), c_11 (gamma_tiled), c_12 (beta_rm), c_13 (beta_tiled)
- **Rationale**: Tilize cannot read and write the same CB. Separate RM and tiled CBs are required. Separating gamma from beta allows independent lifetimes and cleaner code.
- **Alternatives Considered**: Share CBs between gamma and beta (rejected: complicates lifetime management)
- **Tradeoffs**: More CB resources used but simpler code.

### Decision 5: Intermediate CB for Scaled Output
- **Choice**: Use c_14 (cb_scaled) for Phase 10 output, distinct from c_9
- **Rationale**: Phase 10 (gamma mul) consumes c_9 (standardized) and produces c_14 (scaled). Phase 11 (beta add) consumes c_14 and produces back to c_9 (reused for final output before untilize). This allows the untilize phase to remain unchanged from the reference.
- **Alternatives Considered**:
  - Chain: c_9 -> gamma -> c_14 -> beta -> c_15 -> untilize (rejected: adds another CB)
  - Reuse c_9 for both phases (rejected: cannot read and write same CB simultaneously)
- **Tradeoffs**: CB c_9 is reused (cleared between Phase 8 and Phase 11), which is valid since untilize consumes it immediately.

### Decision 6: Phase 11 Output Goes to c_9
- **Choice**: Beta add (Phase 11) writes to c_9, which untilize then reads
- **Rationale**: The reference untilize reads from c_9 (cb_standardized). By having Phase 11 output to c_9, we can reuse the untilize code unchanged.
- **Alternatives Considered**: Create new c_15 for final tiled output (rejected: unnecessary CB)
- **Tradeoffs**: Requires c_9 to be cleared/reset between Phase 8 and Phase 11. This is implicitly handled since Phase 9 untilize pops from c_9.

## Work Distribution

### Work Unit Definition
**Granularity**: Tile-row (same as reference)
**Unit size**: Wt tiles input produces Wt tiles output (one tile-row = 32 sticks x W elements)
**Total units**: Ht tile-rows

### Parallelization Strategy
- **Grid**: 1 x 1 (single core for initial implementation)
- **Work per core**: All Ht tile-rows
- **Load balancing**: N/A (single core)

**Note**: Multi-core extension would split Ht tile-rows across cores. Each core would need its own copy of gamma/beta or would read from shared L1.

## Data Flow

### High-Level Flow

```
DRAM (RM sticks, shape [..., W])
    |
    v  [Reader: read 32 sticks per tile-row, generate scaler and epsilon once]
    |  [Reader: read gamma (32 sticks) ONCE at program start]
    |  [Reader: read beta (32 sticks) ONCE at program start]
CB_0 (cb_in_rm): RM sticks (2*Wt pages, double-buffered)
CB_10 (cb_gamma_rm): Gamma RM sticks (Wt pages, read once)
CB_12 (cb_beta_rm): Beta RM sticks (Wt pages, read once)
    |
    +---> [Compute: tilize gamma ONCE] ---> CB_11 (cb_gamma_tiled): Wt tiles [PROGRAM LIFETIME]
    +---> [Compute: tilize beta ONCE] ---> CB_13 (cb_beta_tiled): Wt tiles [PROGRAM LIFETIME]
    |
    v  [Compute Phase 1: tilize input]
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
    |     [Compute Phases 6-7: add epsilon + rsqrt (combined in DST)]
    |     CB_6 (variance) + CB_7 (epsilon) ---> CB_8 (cb_rsqrt): 1 rsqrt tile
    |
    |<--------------------------------------------------+
    |     [Compute Phase 8: broadcast multiply (A=CB_4, B=CB_8)]
    v
CB_9 (cb_standardized): Wt standardized tiles
    |
    v  [Compute Phase 10: broadcast multiply gamma (A=CB_9, B=CB_11)]
CB_14 (cb_scaled): Wt scaled tiles
    |
    v  [Compute Phase 11: broadcast add beta (A=CB_14, B=CB_13)]
CB_9 (cb_output_tiled): Wt output tiles (reuses c_9)
    |
    v  [Compute Phase 9: untilize]
CB_16 (cb_out_rm): RM output sticks
    |
    v  [Writer: write 32 sticks (width W) per tile-row]
DRAM (RM sticks, shape [..., W])
```

### Phase-by-Phase Data Flow

| Phase | Operation | Input CB(s) | Output CB | Description |
|-------|-----------|-------------|-----------|-------------|
| Pre-0 | Tilize Gamma | CB_10 (Wt pages) | CB_11 (Wt tiles) | Convert gamma RM to tiles (ONCE) |
| Pre-0 | Tilize Beta | CB_12 (Wt pages) | CB_13 (Wt tiles) | Convert beta RM to tiles (ONCE) |
| 1 | Tilize | CB_0 (Wt pages) | CB_1 (Wt tiles) | Convert 32 RM sticks to Wt tiles |
| 2 | Reduce (Mean) | CB_1 (Wt tiles), CB_2 (scaler) | CB_3 (1 tile) | REDUCE_ROW with PERSISTENT mode |
| 3 | Broadcast Sub | CB_1 (Wt tiles), CB_3 (1 tile) | CB_4 (Wt tiles) | Centralize: input - mean [CB_4 PERSISTENT] |
| 4 | Square | CB_4 (Wt tiles) | CB_5 (Wt tiles) | Element-wise (x-mean)^2 |
| 5 | Reduce (Variance) | CB_5 (Wt tiles), CB_2 (scaler) | CB_6 (1 tile) | REDUCE_ROW with STREAMING mode |
| 6-7 | Add Epsilon + Rsqrt | CB_6 (1 tile), CB_7 (epsilon) | CB_8 (1 tile) | Combined: rsqrt(variance + epsilon) |
| 8 | Broadcast Mul | CB_4 (Wt tiles), CB_8 (1 tile) | CB_9 (Wt tiles) | standardized = centralized * rsqrt |
| 10 | Broadcast Mul (Gamma) | CB_9 (Wt tiles), CB_11 (Wt tiles) | CB_14 (Wt tiles) | scaled = standardized * gamma, ROW broadcast |
| 11 | Broadcast Add (Beta) | CB_14 (Wt tiles), CB_13 (Wt tiles) | CB_9 (Wt tiles) | output = scaled + beta, ROW broadcast |
| 9 | Untilize | CB_9 (Wt tiles) | CB_16 (Wt pages) | Convert Wt tiles to 32 RM sticks |

### Kernel Data Movement

| Kernel | Core | NOC | Actual Function |
|--------|------|-----|-----------------|
| reader_layer_norm_w_rm | RISCV_0 (BRISC) | NOC0 | Read input sticks, gamma sticks, beta sticks, generate scaler/epsilon |
| layer_norm_w_rm_compute | RISCV_2,3,4 (TRISC) | N/A | Tilize gamma/beta once, then 11-phase pipeline per tile-row |
| writer_layer_norm_w_rm | RISCV_1 (NCRISC) | NOC1 | Write output RM sticks |

### Circular Buffer Requirements

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in_rm | Input RM sticks | 2*Wt tiles | Wt tiles | Double | Reader | Compute (tilize) | Block |
| c_1 | cb_in_tiled | Tiled input | Wt tiles | Wt tiles | Single | Compute (tilize) | Compute (reduce, sub) | PERSISTENT (Phases 1-3) |
| c_2 | cb_scaler | Scaler (1/W) | 1 tile | 1 tile | Single | Reader | Compute (reduce x2) | Program |
| c_3 | cb_mean_tiled | Mean tile | 1 tile | 1 tile | Single | Compute (reduce1) | Compute (sub) | Block |
| c_4 | cb_centralized | Centralized tiles | Wt tiles | Wt tiles | Single | Compute (sub) | Compute (square, mul) | PERSISTENT (Phases 3-8) |
| c_5 | cb_squared | Squared tiles | Wt tiles | Wt tiles | Single | Compute (square) | Compute (reduce2) | Block |
| c_6 | cb_variance | Variance tile | 1 tile | 1 tile | Single | Compute (reduce2) | Compute (add_eps) | Block |
| c_7 | cb_epsilon | Epsilon scalar | 1 tile | 1 tile | Single | Reader | Compute (add_eps) | Program |
| c_8 | cb_rsqrt | Rsqrt result | 1 tile | 1 tile | Single | Compute (rsqrt) | Compute (mul) | Block |
| c_9 | cb_standardized / cb_output_tiled | Standardized / Final tiled output | Wt tiles | Wt tiles | Single | Compute (mul/add) | Compute (gamma mul / untilize) | Block |
| c_10 | cb_gamma_rm | Gamma RM sticks | Wt tiles | Wt tiles | Single | Reader | Compute (tilize gamma) | Program (read once) |
| c_11 | cb_gamma_tiled | Gamma tiled | Wt tiles | Wt tiles | Single | Compute (tilize gamma) | Compute (gamma mul) | Program (persist) |
| c_12 | cb_beta_rm | Beta RM sticks | Wt tiles | Wt tiles | Single | Reader | Compute (tilize beta) | Program (read once) |
| c_13 | cb_beta_tiled | Beta tiled | Wt tiles | Wt tiles | Single | Compute (tilize beta) | Compute (beta add) | Program (persist) |
| c_14 | cb_scaled | Scaled output | Wt tiles | Wt tiles | Single | Compute (gamma mul) | Compute (beta add) | Block |
| c_16 | cb_out_rm | Output RM sticks | 2*Wt tiles | Wt tiles | Double | Compute (untilize) | Writer | Block |

### CB Lifetime Summary

| Lifetime | CBs | Description |
|----------|-----|-------------|
| Program | c_2 (scaler), c_7 (epsilon), c_10 (gamma_rm), c_11 (gamma_tiled), c_12 (beta_rm), c_13 (beta_tiled) | Generated/read once, persist for all tile-rows |
| PERSISTENT | c_1 (phases 1-3), c_4 (phases 3-8) | Multi-phase persistence within tile-row |
| Block | c_0, c_3, c_5, c_6, c_8, c_9, c_14, c_16 | Standard per-tile-row lifetime |

## Memory Access Patterns

### RISCV_0 ("reader" / BRISC) Access

**Input sticks (per tile-row)**:
- Reserve Wt pages in CB_0
- Read 32 sticks sequentially via TensorAccessor
- NoC barrier, push Wt pages

**Gamma sticks (once at program start)**:
- Reserve Wt pages in CB_10
- Read 32 sticks (gamma tensor has 32 rows to form one tile-row)
- NoC barrier, push Wt pages

**Beta sticks (once at program start)**:
- Reserve Wt pages in CB_12
- Read 32 sticks (beta tensor has 32 rows to form one tile-row)
- NoC barrier, push Wt pages

**Scaler and epsilon (once at program start)**:
- Generate scaler tile (1/W) into CB_2
- Generate epsilon tile into CB_7

### RISCV_1 ("writer" / NCRISC) Access

**Output sticks (per tile-row)**:
- Wait for Wt tiles in CB_16
- Write 32 sticks sequentially via TensorAccessor
- NoC barrier, pop Wt tiles

### Compute Access

**Pre-loop (once)**:
- Wait CB_10 (gamma_rm, Wt pages), tilize to CB_11 (gamma_tiled)
- Wait CB_12 (beta_rm, Wt pages), tilize to CB_13 (beta_tiled)
- Do NOT pop CB_11 or CB_13 (program lifetime)

**Per tile-row loop**:
1. Phase 1: Wait CB_0 (Wt pages), tilize to CB_1, pop CB_0
2. Phase 2: Reduce CB_1 (PERSISTENT, no pop) with CB_2, output to CB_3
3. Phase 3: Subtract CB_1 (pop at end) - CB_3 (pop), output to CB_4
4. Phase 4: Square CB_4 (PERSISTENT, no pop), output to CB_5
5. Phase 5: Reduce CB_5 (STREAMING, pop per tile) with CB_2, output to CB_6
6-7. Phases 6-7: Add CB_6 + CB_7 (no pop, program lifetime), rsqrt, output to CB_8, pop CB_6
8. Phase 8: Multiply CB_4 (pop at end) * CB_8 (pop), output to CB_9
9. Phase 10: Multiply CB_9 (pop) * CB_11 (no pop, program lifetime), output to CB_14, BroadcastDim::ROW
10. Phase 11: Add CB_14 (pop) + CB_13 (no pop, program lifetime), output to CB_9 (reused), BroadcastDim::ROW
11. Phase 9: Wait CB_9 (Wt tiles), untilize to CB_16, pop CB_9

## Compile-Time Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_stick_size_aligned | uint32_t | NoC-aligned stick size in bytes (W * element_size, rounded up) |
| 1 | packed_scaler_value | uint32_t | Two bfloat16 (1/W) packed into uint32 for reduce scaler |
| 2 | packed_epsilon_value | uint32_t | Two bfloat16 (epsilon) packed into uint32 for epsilon tile |
| 3 | Ht | uint32_t | Height in tiles (number of tile-rows to process) |
| 4 | Wt | uint32_t | Width in tiles (tiles per row) |
| 5 | gamma_stick_size_aligned | uint32_t | NoC-aligned gamma stick size (W * element_size, rounded up) |
| 6 | beta_stick_size_aligned | uint32_t | NoC-aligned beta stick size (W * element_size, rounded up) |
| 7+ | TensorAccessorArgs (input) | multiple | Input buffer address mode, page size, etc. |
| X+ | TensorAccessorArgs (gamma) | multiple | Gamma buffer address mode, page size, etc. |
| Y+ | TensorAccessorArgs (beta) | multiple | Beta buffer address mode, page size, etc. |

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
| 2 | Wt | uint32_t | Width in tiles |
| 3+ | TensorAccessorArgs | multiple | Output buffer address mode, page size, etc. |

## Runtime Arguments

### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input DRAM buffer address |
| 1 | gamma_addr | uint32_t | Gamma DRAM buffer address |
| 2 | beta_addr | uint32_t | Beta DRAM buffer address |

### Compute Kernel

None (all parameters are compile-time).

### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output DRAM buffer address |

## Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| Single tile (H=32, W=32) | Process one tile-row, output matches layer norm formula |
| W not multiple of 32 | Pad W to next multiple of 32, process padded tensor |
| H not multiple of 32 | Pad H to next multiple of 32, process padded tensor |
| Large input (many tile-rows) | Process tile-rows sequentially, gamma/beta reused |
| epsilon = 0 | Undefined behavior (potential division by zero), should validate epsilon > 0 |
| All elements equal (zero variance) | Output depends on epsilon, should produce valid result |
| Gamma all zeros | Output all beta values after normalization |
| Beta all zeros | Output gamma-scaled normalized values |

## Agent Handoff

This spec will be consumed by implementation agents. Each agent reads specific sections:

| Agent | Reads These Sections |
|-------|---------------------|
| **ttnn-operation-scaffolder** | API Specification, Input Tensor Requirements, Output Tensor Specification |
| **ttnn-factory-builder** | Circular Buffer Requirements, Work Distribution, Data Flow |
| **ttnn-kernel-dataflow** | Kernel Data Movement, Memory Access Patterns |
| **ttnn-kernel-compute** | Compute Access, Mathematical Definition |

The agents know HOW to implement; this spec defines WHAT to implement.

## Test Criteria

What behavior should be verified (agents decide how/when to test):

### Validation Behavior
- Wrong input tensor rank (< 2D) -> error containing "must have at least 2 dimensions"
- Wrong input layout (not ROW_MAJOR) -> error containing "must be in ROW_MAJOR layout"
- Wrong gamma shape -> error containing "gamma shape must match input width"
- Wrong beta shape -> error containing "beta shape must match input width"
- Gamma not on DRAM -> error containing "gamma must be interleaved in DRAM"
- Beta not on DRAM -> error containing "beta must be interleaved in DRAM"
- Unsupported dtype -> error containing "unsupported dtype"
- Mismatched dtypes -> error containing "dtype must match"

### Shape Behavior
- Output shape equals input shape
- Output is ROW_MAJOR layout

### Functional Behavior
- Single tile (32x32): output matches PyTorch layer_norm reference
- Multi-tile (64x64, 128x32, etc.): output matches PyTorch layer_norm reference
- Numerical accuracy: rtol=1e-3, atol=1e-3 for BFLOAT16

### Performance Behavior (informational, not blocking)
- Gamma and beta should be read once, not per tile-row (verify via profiling if needed)

## Open Questions

None - all design decisions have been made with reasonable assumptions.

## References
- Reference analysis: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/normalization/layer_norm_w_rm/references/standardize_w_rm_analysis.md`
- DeepWiki query: "How does layer normalization handle gamma and beta parameters?"
  - Finding: BroadcastType::ROW for gamma/beta, mul_bcast_rows and add_bcast_rows primitives
- Documentation: `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`
  - Finding: BroadcastDim::ROW means B shape [1, Wt], replicates down (height)
- Documentation: `ttnn/cpp/ttnn/kernel_lib/cb_policies.hpp`
  - Finding: PreloadedNoPop = InputPolicy<WaitCallerManaged, PopNever> for persistent tiles
