# standardize_w_rm Implementation Analysis

## Overview

The `standardize_w_rm` operation performs row-wise standardization (z-score normalization) on row-major interleaved tensors. For each row of width W, it computes:

```
output = (input - mean) / sqrt(variance + epsilon)
```

This operation is the foundation for extending to `layer_norm_w_rm`, which adds:
1. Multiplication by gamma (scale factor) with shape [1, ..., 1, W]
2. Addition of beta (bias factor) with shape [1, ..., 1, W]

**Program Factory Path**: `/localdev/mstaletovic/tt-metal/ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/device/standardize_w_rm_program_factory.cpp`

## Mathematical Definition

```
mean[..., 0] = (1/W) * sum(input[..., j] for j in range(W))
centralized[..., j] = input[..., j] - mean[..., 0]  for all j
variance[..., 0] = (1/W) * sum(centralized[..., j]^2 for j in range(W))
rsqrt_var[..., 0] = rsqrt(variance[..., 0] + epsilon)
output[..., j] = centralized[..., j] * rsqrt_var[..., 0]  for all j
```

For `layer_norm_w_rm`, two additional operations are appended:
```
scaled[..., j] = output[..., j] * gamma[j]  for all j
final[..., j] = scaled[..., j] + beta[j]  for all j
```

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile-row |
| **Unit size** | Wt tiles input, Wt tiles output (one tile-row = 32 sticks x W elements) |
| **Total units** | Ht tile-rows |
| **Loop structure** | Outer loop over Ht tile-rows, each iteration processes all 9 phases |

One work unit is a **tile-row**: 32 consecutive row-major sticks that, when tilized, produce Wt tiles. Each tile-row is processed through the complete 9-phase pipeline before moving to the next.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [..., H, W] |
| **Dimension convention** | Last dim is W (width to reduce over) |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 (FLOAT32 supported) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input: [..., H, W] |
| **Padded shape** | [..., ceil(H/32)*32, ceil(W/32)*32] |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input |

### Layout Transformations

1. **Phase 1 (Tilize)**: RM sticks (CB c_0) -> Tiled (CB c_1)
2. **Phase 9 (Untilize)**: Tiled (CB c_9) -> RM sticks (CB c_16)

For `layer_norm_w_rm` extension:
- **Gamma/Beta**: Start as RM in DRAM, need tilize in compute kernel (one-time, reused across all tile-rows)

## Data Flow Pattern

### High-Level Flow Diagram

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
    |     [Compute Phases 6-7: add epsilon + rsqrt (combined in DST)]
    |     CB_6 (variance) + CB_7 (epsilon) ---> CB_8 (cb_rsqrt): 1 rsqrt tile
    |
    |<--------------------------------------------------+
    |     [Compute Phase 8: broadcast multiply (A=CB_4, B=CB_8)]
    v
CB_9 (cb_standardized): Wt standardized tiles
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
| 1 | Tilize | CB_0 (Wt pages) | CB_1 (Wt tiles) | Convert 32 RM sticks to Wt tiles |
| 2 | Reduce (Mean) | CB_1 (Wt tiles), CB_2 (scaler) | CB_3 (1 tile) | REDUCE_ROW with PERSISTENT mode |
| 3 | Broadcast Sub | CB_1 (Wt tiles), CB_3 (1 tile) | CB_4 (Wt tiles) | Centralize: input - mean [CB_4 PERSISTENT] |
| 4 | Square | CB_4 (Wt tiles) | CB_5 (Wt tiles) | Element-wise (x-mean)^2 |
| 5 | Reduce (Variance) | CB_5 (Wt tiles), CB_2 (scaler) | CB_6 (1 tile) | REDUCE_ROW with STREAMING mode |
| 6-7 | Add Epsilon + Rsqrt | CB_6 (1 tile), CB_7 (epsilon) | CB_8 (1 tile) | Combined: rsqrt(variance + epsilon) |
| 8 | Broadcast Mul | CB_4 (Wt tiles), CB_8 (1 tile) | CB_9 (Wt tiles) | standardized = centralized * rsqrt |
| 9 | Untilize | CB_9 (Wt tiles) | CB_16 (Wt pages) | Convert Wt tiles to 32 RM sticks |

## Circular Buffer Configuration

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
| c_9 | cb_standardized | Standardized tiles | Wt tiles | Wt tiles | Single | Compute (mul) | Compute (untilize) | Block |
| c_16 | cb_out_rm | Output RM sticks | 2*Wt tiles | Wt tiles | Double | Compute (untilize) | Writer | Block |

### CB Lifetime Summary

| Lifetime | CBs | Description |
|----------|-----|-------------|
| Program | c_2 (scaler), c_7 (epsilon) | Generated once by reader, persist for all tile-rows |
| PERSISTENT | c_1 (phases 1-3), c_4 (phases 3-8) | Multi-phase persistence within tile-row |
| Block | c_0, c_3, c_5, c_6, c_8, c_9, c_16 | Standard per-tile-row lifetime |

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering Type | Overlap Potential |
|----|----------|------------|----------------|-------------------|
| c_0 | 2*Wt | Wt | Double | Reader/Compute overlap |
| c_1 | Wt | Wt | Single | No overlap (PERSISTENT) |
| c_2 | 1 | 1 | Single | N/A (Program lifetime) |
| c_3 | 1 | 1 | Single | No overlap |
| c_4 | Wt | Wt | Single | No overlap (PERSISTENT) |
| c_5 | Wt | Wt | Single | No overlap |
| c_6 | 1 | 1 | Single | No overlap |
| c_7 | 1 | 1 | Single | N/A (Program lifetime) |
| c_8 | 1 | 1 | Single | No overlap |
| c_9 | Wt | Wt | Single | No overlap |
| c_16 | 2*Wt | Wt | Double | Compute/Writer overlap |

## Index Calculations

### Stick-to-Tile Mapping

- Each tile-row contains 32 sticks (TILE_HEIGHT)
- Each tile-row produces Wt tiles after tilize
- Stick index within tile-row: `s` (0 to 31)
- Global stick index: `ht * 32 + s` where `ht` is tile-row index

### CB Page Addressing

The tilize helper expects Wt pages in the input CB, where each "page" corresponds to the portion of 32 sticks that contributes to one tile. The mapping is:
- `input_stick_size = W * element_size` (aligned to buffer alignment)
- L1 write address advances by `input_stick_size` per stick

### Tensor Accessor Usage

```cpp
const auto accessor = TensorAccessor(tensor_args, src_addr, input_stick_size);
uint64_t noc_addr = accessor.get_noc_addr(stick_id);
```

The TensorAccessor handles interleaved memory layout, computing the correct NoC address for each stick based on its global index.

## Memory Access Patterns

### Read Pattern (Reader Kernel)

| Aspect | Pattern |
|--------|---------|
| **Data ordering** | Sequential sticks within tile-row |
| **Access type** | DRAM -> L1 via NoC0 |
| **Granularity** | 32 sticks per tile-row |
| **Barriers** | `noc_async_read_barrier()` after all 32 sticks |

**Pseudocode**:
```cpp
for (ht = 0; ht < Ht; ht++) {
    cb_reserve_back(cb_in_rm, Wt);
    for (s = 0; s < 32; s++) {
        noc_async_read(accessor.get_noc_addr(stick_id++), l1_addr, input_stick_size);
        l1_addr += input_stick_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_in_rm, Wt);
}
```

### Write Pattern (Writer Kernel)

| Aspect | Pattern |
|--------|---------|
| **Data ordering** | Sequential sticks within tile-row |
| **Access type** | L1 -> DRAM via NoC1 |
| **Granularity** | 32 sticks per tile-row |
| **Barriers** | `noc_async_write_barrier()` after all 32 sticks |

**Pseudocode**:
```cpp
for (ht = 0; ht < Ht; ht++) {
    cb_wait_front(cb_out_rm, Wt);
    for (s = 0; s < 32; s++) {
        noc_async_write(l1_addr, accessor.get_noc_addr(stick_id++), output_stick_size);
        l1_addr += output_stick_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out_rm, Wt);
}
```

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D |
| **Grid dimensions** | 1 x 1 |
| **Total cores** | 1 (single core implementation) |
| **Work per core** | All Ht tile-rows |
| **Load balancing** | N/A (single core) |

**Note**: Multi-core extension would split Ht tile-rows across cores, with each core processing a subset independently.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_stick_size_aligned | uint32_t | NoC-aligned stick size in bytes (W * element_size, rounded up) |
| 1 | packed_scaler_value | uint32_t | Two bfloat16 (1/W) packed into uint32 for reduce scaler |
| 2 | packed_epsilon_value | uint32_t | Two bfloat16 (epsilon) packed into uint32 for epsilon tile |
| 3 | Ht | uint32_t | Height in tiles (number of tile-rows to process) |
| 4 | Wt | uint32_t | Width in tiles (tiles per row) |
| 5+ | TensorAccessorArgs | multiple | Input buffer address mode, page size, etc. |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Height in tiles (outer loop count) |
| 1 | Wt | uint32_t | Width in tiles (tiles per row) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_stick_size_aligned | uint32_t | NoC-aligned output stick size (W * element_size, rounded up) |
| 1 | Ht | uint32_t | Height in tiles (number of tile-rows) |
| 2 | Wt | uint32_t | Width in tiles |
| 3+ | TensorAccessorArgs | multiple | Output buffer address mode, page size, etc. |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input DRAM buffer address |

#### Compute Kernel

None (all parameters are compile-time).

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output DRAM buffer address |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_standardize_w_rm | RISCV_0 (BRISC) | NOC0 | DRAM | CB_0, CB_2, CB_7 | Read RM sticks, generate scalers |

**File**: `ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/device/kernels/dataflow/reader_standardize_w_rm.cpp`

**Key Logic**:
- **Phase 0 (Once)**: Generate scaler tile (1/W) using `generate_reduce_scaler()`, generate epsilon tile using `generate_reduce_scaler()`
- **Per tile-row**: Reserve Wt pages, read 32 sticks via TensorAccessor, barrier, push Wt pages

**Includes**:
```cpp
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
```

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| standardize_w_rm_compute | RISCV_2,3,4 (TRISC) | N/A | CB_0-8 | CB_9, CB_16 | 9-phase pipeline |

**File**: `ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/device/kernels/compute/standardize_w_rm_compute.cpp`

**Key Logic**:

1. **Custom CB Policies** (for persistence and no-pop semantics):
```cpp
using PreloadedPopAtEnd = cb_policies::InputPolicy<cb_policies::WaitCallerManaged, cb_policies::PopAtEnd>;
using PreloadedNoPop = cb_policies::InputPolicy<cb_policies::WaitCallerManaged, cb_policies::PopNever>;
using WaitUpfrontPopAtEnd = cb_policies::InputPolicy<cb_policies::WaitUpfront, cb_policies::PopAtEnd>;
```

2. **Phase-by-Phase Implementation**:

| Phase | Helper/Raw | Function Call |
|-------|------------|---------------|
| 1 | Helper | `compute_kernel_lib::tilize(cb_in_rm, Wt, cb_in_tiled, 1)` |
| 2 | Helper | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, PERSISTENT>(cb_in_tiled, cb_scaler, cb_mean_tiled, TileShape::row(Wt))` |
| 3 | Helper | `compute_kernel_lib::sub<COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(cb_in_tiled, cb_mean_tiled, cb_centralized_tiled, BinaryTileShape::row(Wt))` |
| 4 | Helper | `compute_kernel_lib::binary_op<SQUARE, NONE, PreloadedNoPop>(cb_centralized_tiled, cb_centralized_tiled, cb_squared_tiled, BinaryTileShape::row(Wt))` |
| 5 | Helper | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, STREAMING>(cb_squared_tiled, cb_scaler, cb_variance_tiled, TileShape::row(Wt))` |
| 6-7 | Raw | copy_tile + add_binary_tile + rsqrt_tile (DST-based, no intermediate CB) |
| 8 | Helper | `compute_kernel_lib::mul<COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(cb_centralized_tiled, cb_rsqrt_tiled, cb_standardized_tiled, BinaryTileShape::row(Wt))` |
| 9 | Helper | `compute_kernel_lib::untilize<Wt, cb_standardized_tiled, cb_out_rm>(1)` |

3. **Phases 6-7 Raw Implementation** (add epsilon + rsqrt combined):
```cpp
cb_wait_front(cb_variance_tiled, 1);
cb_wait_front(cb_epsilon, 1);
cb_reserve_back(cb_rsqrt_tiled, 1);

tile_regs_acquire();

// Copy variance to DST[0]
copy_tile_to_dst_init_short_with_dt(cb_epsilon, cb_variance_tiled);
copy_tile(cb_variance_tiled, 0, 0);

// Add epsilon
add_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_variance_tiled, cb_epsilon);
copy_tile(cb_epsilon, 0, 1);
add_binary_tile(0, 1, 0);  // DST[0] = DST[0] + DST[1]

// Rsqrt
rsqrt_tile_init();
rsqrt_tile(0);

// Pack result
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_rsqrt_tiled);
tile_regs_release();

cb_push_back(cb_rsqrt_tiled, 1);
cb_pop_front(cb_variance_tiled, 1);
// Note: cb_epsilon NOT popped (program lifetime)
```

**Includes**:
```cpp
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/cb_policies.hpp"
```

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_standardize_w_rm | RISCV_1 (NCRISC) | NOC1 | CB_16 | DRAM | Write RM sticks |

**File**: `ttnn/cpp/ttnn/operations/reduction/standardize_w_rm/device/kernels/dataflow/writer_standardize_w_rm.cpp`

**Key Logic**:
- Per tile-row: Wait for Wt tiles, write 32 sticks via TensorAccessor, barrier, pop Wt tiles

**Includes**:
```cpp
#include "api/dataflow/dataflow_api.h"
```

## Implementation Notes

### Critical Design Decisions

1. **CB_4 Persistence**: The centralized tiles in CB_4 must persist from Phase 3 through Phase 8. This is achieved using `PreloadedNoPop` policy in Phase 4 (square) and `PreloadedPopAtEnd` policy in Phase 8 (multiply).

2. **Phases 6-7 Combined**: Add epsilon and rsqrt are combined into a single DST-based sequence to avoid an intermediate CB write/read cycle. This matches the batch_norm pattern.

3. **Separate CB for Multiply Output**: CB_9 (cb_standardized) is used for Phase 8 multiply output, separate from CB_16 (cb_out_rm), because untilize cannot read and write the same CB.

4. **Scaler Generation Pattern**: Both 1/W scaler and epsilon tile are generated using `generate_reduce_scaler()` which creates a tile with the scalar value in the bcast19 pattern (optimized for reduce operations).

### Binary Operation Broadcast Verification

| Phase | Op | CB_A | CB_A Valid | CB_B | CB_B Valid | Broadcast |
|-------|-----|------|------------|------|------------|-----------|
| 3 | sub | c_1 | All | c_3 | Col0 | **COL** (mean broadcasts across columns) |
| 4 | square | c_4 | All | c_4 | All | NONE (self-multiply) |
| 8 | mul | c_4 | All | c_8 | Col0 | **COL** (rsqrt broadcasts across columns) |

**Key insight**: REDUCE_ROW produces column-shaped output (Col0 valid), so subsequent broadcast operations use `BroadcastDim::COL` to replicate the reduced value across the width dimension.

### CB Index Constants

```cpp
constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;              // Input RM sticks
constexpr uint32_t cb_in_tiled = tt::CBIndex::c_1;           // Tiled input (PERSISTENT phases 1-3)
constexpr uint32_t cb_scaler = tt::CBIndex::c_2;             // Scaler tile (1/W)
constexpr uint32_t cb_mean_tiled = tt::CBIndex::c_3;         // Mean tile
constexpr uint32_t cb_centralized_tiled = tt::CBIndex::c_4;  // Centralized tiles (PERSISTENT phases 3-8)
constexpr uint32_t cb_squared_tiled = tt::CBIndex::c_5;      // Squared tiles
constexpr uint32_t cb_variance_tiled = tt::CBIndex::c_6;     // Variance tile
constexpr uint32_t cb_epsilon = tt::CBIndex::c_7;            // Epsilon scalar tile
constexpr uint32_t cb_rsqrt_tiled = tt::CBIndex::c_8;        // Rsqrt result tile
constexpr uint32_t cb_standardized_tiled = tt::CBIndex::c_9; // Standardized tiles (Phase 8 output)
constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;            // Output RM sticks
```

## Extension Points for layer_norm_w_rm

### Additional CBs Required

| CB ID | Name | Purpose | Capacity | Lifetime |
|-------|------|---------|----------|----------|
| c_10 | cb_gamma_rm | Gamma RM sticks (from DRAM) | 2*Wt tiles | Program (read once) |
| c_11 | cb_gamma_tiled | Gamma tiled (tilized once) | Wt tiles | Program (persist all tile-rows) |
| c_12 | cb_beta_rm | Beta RM sticks (from DRAM) | 2*Wt tiles | Program (read once) |
| c_13 | cb_beta_tiled | Beta tiled (tilized once) | Wt tiles | Program (persist all tile-rows) |
| c_14 | cb_scaled | Scaled output (after gamma multiply) | Wt tiles | Block |

### Additional Phases

| Phase | Operation | Input CB(s) | Output CB | Description |
|-------|-----------|-------------|-----------|-------------|
| 10 | Broadcast Mul (Gamma) | CB_9 (Wt tiles), CB_11 (Wt tiles) | CB_14 (Wt tiles) | scaled = standardized * gamma, ROW broadcast |
| 11 | Broadcast Add (Beta) | CB_14 (Wt tiles), CB_13 (Wt tiles) | CB_9 (Wt tiles) | output = scaled + beta, ROW broadcast |

**Important**: Gamma and beta have shape [1, ..., 1, W], meaning they have valid data in all Wt tiles (not just Col0 like mean/rsqrt). This requires `BroadcastDim::ROW` for the multiply and add operations, where the gamma/beta values are replicated down the column (height) dimension.

### Reader Kernel Modifications

1. **Once at program start**:
   - Read gamma tensor (32 sticks x W elements) into cb_gamma_rm
   - Read beta tensor (32 sticks x W elements) into cb_beta_rm

2. **Compile-time args to add**:
   - `gamma_stick_size_aligned`
   - `beta_stick_size_aligned`
   - `gamma_addr`, `beta_addr` as runtime args

### Compute Kernel Modifications

1. **Once at program start (before tile-row loop)**:
   - Tilize gamma: `compute_kernel_lib::tilize(cb_gamma_rm, Wt, cb_gamma_tiled, 1)`
   - Tilize beta: `compute_kernel_lib::tilize(cb_beta_rm, Wt, cb_beta_tiled, 1)`

2. **Per tile-row (after Phase 9)**:
   - Phase 10: `compute_kernel_lib::mul<BroadcastDim::ROW>(...)`
   - Phase 11: `compute_kernel_lib::add<BroadcastDim::ROW>(...)`

### Persistence Strategy for Gamma/Beta

- CB_11 (gamma_tiled) and CB_13 (beta_tiled) must use `PopNever` policy
- They are tilized once and reused for all Ht tile-rows
- Alternative: Keep them in L1 and use indexed access with `WaitCallerManaged` policy

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the tilize operation work for row-major tensors in compute kernels?"
   **Reason**: Understanding tilize mechanics for gamma/beta one-time tilization
   **Key Findings**:
   - `tilize_init` configures unpacker and math for tilize operation
   - `tilize_block` converts "sticks" (rows) to 32x32 tiles
   - Process: llk_unpack_tilize_block -> llk_math_eltwise_unary_datacopy -> llk_pack

2. **Query**: "How does layer normalization handle gamma and beta parameters in TTNN operations?"
   **Reason**: Finding existing patterns for gamma/beta handling
   **Key Findings**:
   - Gamma and beta are read from DRAM by reader kernels
   - Stored in ROW_MAJOR_LAYOUT to avoid padding
   - Applied in compute via `mul_bcast_rows` (gamma) and `add_bcast_rows` (beta)
   - Conditional execution based on `do_gamma` and `do_beta` flags

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
   **Reason**: Understanding tilize helper API for gamma/beta tilization
   **Key Information**: `tilize(icb, block_w, ocb, num_blocks)` handles wait/reserve/push internally

2. **Source**: `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`
   **Reason**: Understanding broadcast dimension semantics
   **Key Information**:
   - `BroadcastDim::ROW` replicates top row down (for width-shaped inputs)
   - `BroadcastDim::COL` replicates left column right (for height-shaped inputs after REDUCE_ROW)

3. **Source**: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp`
   **Reason**: Understanding PERSISTENT vs STREAMING modes
   **Key Information**:
   - PERSISTENT: waits for all tiles upfront, does NOT pop (tiles persist)
   - STREAMING: waits/pops one tile at a time

4. **Source**: `ttnn/cpp/ttnn/kernel_lib/cb_policies.hpp`
   **Reason**: Understanding custom CB management policies
   **Key Information**:
   - `PreloadedNoPop`: `InputPolicy<WaitCallerManaged, PopNever>` - tiles present, never pop
   - `PreloadedPopAtEnd`: `InputPolicy<WaitCallerManaged, PopAtEnd>` - tiles present, pop at end

5. **Source**: `standardize_w_rm_spec.md` and `kernel_design.md`
   **Reason**: Reference design documentation
   **Key Information**: Complete phase-by-phase design with CB policies and data flow
