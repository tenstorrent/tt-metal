# LERP Implementation Analysis

## Overview

LERP (Linear Interpolation) computes `output = input + weight * (end - input)`, which is equivalent to `output = (1 - weight) * input + weight * end`. It is implemented as a ternary SFPU operation within the shared ternary device operation framework, alongside WHERE, ADDCMUL, and ADDCDIV.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

LERP supports two variants:
- **TTT** (tensor-tensor-tensor): `lerp(input, end, weight)` where all three operands are tensors
- **TTS** (tensor-tensor-scalar): `lerp(input, end, scalar_weight)` where weight is a scalar float

The operation is parameterized through preprocessor defines (`TERNARY_SFPU_OP_INIT` = `lerp_tile_init`, `TERNARY_SFPU_OP_FUNC` = `lerp_tile<DataFormat>`) injected by the program factory, reusing the same kernel source files as the WHERE operation.

## Work Unit Definition

One work unit is **one 32x32 tile**. Each tile is processed independently through the read-compute-write pipeline. The compute kernel processes `num_tiles_per_cycle = 1` tile per iteration: it copies 3 input tiles (or 2 tiles + 1 scalar fill) into destination registers, executes the LERP SFPU operation, and packs 1 output tile.

## Tensor Format and Layout

### Input Tensors

| Property | Input A (predicate/input) | Input B (end / true) | Input C (weight / false) |
|---|---|---|---|
| Semantic Role | Base value (`input`) | End value (`end`) | Weight (`weight`) |
| CB Assignment | c_0 | c_1 | c_2 (TTT only) |
| Dimension Convention | Up to rank 6+ (collapsed beyond 5) | Same | Same |
| Tensor Layout | TILE (32x32) | TILE (32x32) | TILE (32x32) or scalar |
| Memory Layout | Interleaved or Sharded (L1) | Interleaved or Sharded (L1) | Interleaved or Sharded (L1) |
| Buffer Type | DRAM or L1 | DRAM or L1 | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32 | BFLOAT16, FLOAT32 | BFLOAT16, FLOAT32 (or scalar float for TTS) |

### Output Tensor

| Property | Output |
|---|---|
| CB Assignment | c_3 |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded (L1) |
| Buffer Type | DRAM or L1 |
| Data Type | Same as input dtype (or explicitly specified) |

### Layout Transformations

No explicit tilize/untilize is performed in the kernel. All inputs must already be in tiled format. For the TTS variant, the scalar weight is filled into a destination register using `fill_tile` (float) or `fill_tile_int` (int) rather than being read from memory. For column/scalar broadcast cases, the reader kernel fills tiles with first-column/first-row/first-element values using dataflow helper functions.

## Data Flow Pattern

### TTT Variant (No Broadcast)
1. **Reader** reads tile `i` from each of the 3 input tensors via TensorAccessor (NoC read from DRAM/L1 interleaved) into CBs c_0, c_1, c_2
2. **Compute** waits for 1 tile on each of c_0, c_1, c_2; copies them to DST registers 0, 1, 2; calls `lerp_tile_init()` then `lerp_tile(0, 1, 2, 0)` on SFPU; packs result from DST[0] to c_3
3. **Writer** waits for 1 tile on c_3; writes it to output via TensorAccessor (NoC write to DRAM/L1)

### TTS Variant (No Broadcast)
1. **Reader** reads tile `i` from input A (predicate) into c_0 and tile `i` from input B (end/true) into c_1
2. **Compute** waits for tiles on c_0 and c_1; copies predicate to DST[0], true tensor to DST[1]; fills scalar weight into DST[2] via `fill_tile(2, scalar_val)`; calls `lerp_tile(0, 1, 2, 0)`; packs to c_3
3. **Writer** writes output tile from c_3 to DRAM/L1

### Sharded Path
When all tensors are sharded in L1 on the same core grid, the reader simply does `cb_reserve_back` + `cb_push_back` to expose the already-present L1 data through the CB. No NoC reads occur. The writer similarly skips writes since output is already in L1.

## Circular Buffer Configuration

### Base Configuration (TTT, No Broadcast)

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| c_0 | predicate_tensor_cb | Input A (input/base value) | 2 (or shard volume) | 1 | Double-buffered | Reader | Compute |
| c_1 | value_true_tensor_cb | Input B (end value) | 2 (or shard volume) | 1 | Double-buffered | Reader | Compute |
| c_2 | value_false_tensor_cb | Input C (weight) | 2 (or shard volume) | 1 | Double-buffered | Reader | Compute |
| c_3 | output_tensor_cb | Output | 2 (or shard volume) | 1 | Double-buffered | Compute | Writer |

### Additional CBs for ROW_BCAST (TTT, bfloat16 only)

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| c_4 | cb_bcast_a | Broadcast-expanded A | 2 | 1 | Double-buffered | Compute (bcast stage) | Compute (SFPU stage) |
| c_5 | cb_bcast_b | Broadcast-expanded B | 2 | 1 | Double-buffered | Compute (bcast stage) | Compute (SFPU stage) |
| c_6 | cb_bcast_c | Broadcast-expanded C | 2 | 1 | Double-buffered | Compute (bcast stage) | Compute (SFPU stage) |

### TTS Variant

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| c_0 | predicate_tensor_cb | Input A (input/base value) | 2 (or shard volume) | 1 | Double-buffered | Reader | Compute |
| c_1 | value_true_tensor_cb | Input B (end value) | 2 (or shard volume) | 1 | Double-buffered | Reader | Compute |
| c_3 | output_tensor_cb | Output | 2 (or shard volume) | 1 | Double-buffered | Compute | Writer |

Note: c_2 is not allocated for TTS since the weight is a scalar filled directly into a DST register.

## Pipeline Pattern Summary

All CBs are allocated with capacity = 2 tiles and consumed/produced 1 tile at a time, enabling **double-buffering**. This allows the reader to fill the next tile while compute processes the current tile, and compute to produce the next output while the writer drains the current one. When sharded, CB capacity equals the shard volume, enabling the entire shard to be processed without per-tile NoC transfers.

## Index Calculations

### Interleaved Mode

The reader and writer use `TensorAccessor` for address resolution. Logical tile indices are computed from a linear `start_tile_id` decomposed into multi-dimensional coordinates:

```
tiles_per_n = C * Ht * Wt
tiles_per_d = N * tiles_per_n
tiles_per_nd = D * tiles_per_d

start_nd = start_tile_id / tiles_per_nd
start_d  = (start_tile_id % tiles_per_nd) / tiles_per_d
start_n  = ... / tiles_per_n
start_c  = ... / HtWt
start_th = ... / Wt
start_tw = ... % Wt
```

For broadcast cases, each input tensor maintains its own stride structure (`nD_stride`, `d_stride`, `n_stride`, `c_stride`), which are set to 0 for dimensions that should broadcast (dimension size = 1). This causes the tile offset to not advance for broadcast dimensions.

### Sharded Mode

For sharded tensors, `c_start_id` is computed based on the shard position in the grid:

```
c_start_id = (core_index / num_shards_per_width) * (c_shard_height * output_Wt)
           + (core_index % num_shards_per_width) * c_shard_width
```

The `ShardShapeGenerator` class handles edge cases where the last core(s) may have fewer tiles than the nominal shard shape.

## Memory Access Patterns

### Read Pattern

- **Interleaved (no broadcast)**: Sequential tile reads with stride-based offset calculation for multi-dimensional iteration. Tiles are read in the order: nD -> D -> N -> C -> Ht -> Wt (innermost). Each of the 3 inputs is read at its own computed offset.
- **Interleaved (broadcast)**: Same iteration order, but broadcast inputs have zero strides for broadcast dimensions, causing repeated reads of the same tile.
- **Sharded**: No NoC reads; L1 data is exposed through CB reserve/push.

### Write Pattern

- **Interleaved**: Sequential tile writes following the same nD -> D -> N -> C -> Ht -> Wt order, with `noc_async_write_page` per tile.
- **Width-sharded**: Writer adjusts `dst_tile_offset` by `(Wt - dst_shard_width)` after each tile row to skip tiles belonging to other shards.
- **Fully sharded output**: Writer is a no-op (DST_SHARDED=1 disables the writer body).

## Core Distribution Strategy

| Property | Interleaved Mode | Sharded Mode |
|---|---|---|
| Grid Topology | Rectangular grid from `worker_grid` | Shard grid from tensor shard spec |
| Work Splitting | `split_work_to_cores()` divides total output tiles | Each core processes its shard volume |
| Core Group 1 | Cores with `num_tiles_per_core_group_1` tiles | All shard cores (equal work) |
| Core Group 2 | Cores with `num_tiles_per_core_group_1 - 1` tiles (remainder) | N/A |
| Load Balancing | Remainder tiles distributed to first N cores | Edge shards may have fewer tiles (handled by ShardShapeGenerator) |
| No-op Cores | Cores outside both groups get zero args and skip | Cores outside shard grid get zero args |
| Row-major preference | Grid traversed row-major unless shard orientation says otherwise | Follows shard orientation |

The factory supports two grid strategies:
1. **zero_start_grid**: When the grid starts at (0,0) with a single rectangular range, uses `grid_to_cores` with compute grid dimensions for efficient enumeration.
2. **General grid**: Uses `corerange_to_cores` or `grid_to_cores_with_noop` for arbitrary core ranges.

## Arguments

### Compile-Time Arguments

#### Reader (TTT variant)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_src0 | uint32_t | CB index for predicate tensor (c_0) |
| 1 | cb_id_src1 | uint32_t | CB index for true/end tensor (c_1) |
| 2 | cb_id_src2 | uint32_t | CB index for false/weight tensor (c_2) |
| 3+ | TensorAccessorArgs (src0) | varies | Compile-time tensor accessor config for input A |
| N+ | TensorAccessorArgs (src1) | varies | Compile-time tensor accessor config for input B |
| M+ | TensorAccessorArgs (src2) | varies | Compile-time tensor accessor config for input C |
| last | has_sharding | uint32_t | 1 if sharded path is used, 0 otherwise |

#### Reader (TTS variant)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_in0 | uint32_t | CB index for predicate tensor (c_0) |
| 1 | cb_id_in1 | uint32_t | CB index for true/end tensor (c_1) |
| 2+ | TensorAccessorArgs (src0) | varies | Compile-time tensor accessor config for input A |
| N+ | TensorAccessorArgs (src1) | varies | Compile-time tensor accessor config for input B |

#### Writer

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | CB index for output tensor (c_3) |
| 1+ | TensorAccessorArgs (dst) | varies | Compile-time tensor accessor config for output |
| last | has_sharding | uint32_t | 1 if sharded output, 0 otherwise |

#### Compute

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 - tiles processed per iteration |
| 1 | scalar_is_true_value | uint32_t | 0 for TTS (scalar is false/weight), 1 for TST |

### Runtime Arguments

#### Reader (27 args)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | DRAM address of predicate/input tensor |
| 1 | src1_addr | uint32_t | DRAM address of end/true tensor |
| 2 | src2_addr | uint32_t | DRAM address of weight/false tensor (0 for TTS) |
| 3 | num_tiles | uint32_t | Number of output tiles for this core |
| 4 | start_id | uint32_t | Starting tile ID for this core |
| 5 | nD_stride | uint32_t | Predicate stride for >5D dims |
| 6 | d_stride | uint32_t | Predicate stride for dim -5 |
| 7 | n_stride | uint32_t | Predicate stride for dim -4 |
| 8 | c_stride | uint32_t | Predicate stride for dim -3 |
| 9 | D | uint32_t | Output dim -5 size |
| 10 | N | uint32_t | Output dim -4 size |
| 11 | C | uint32_t | Output dim -3 size |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Collapsed >5D dimension size |
| 15-18 | true/B strides | uint32_t | nD/d/n/c strides for true/end tensor |
| 19 | b_num_tiles | uint32_t | True tensor shard tile count (sharded) |
| 20-23 | false/C strides | uint32_t | nD/d/n/c strides for weight/false tensor |
| 24 | f_num_tiles | uint32_t | False tensor shard tile count (sharded) |
| 25 | dst_shard_width | uint32_t | Output shard width in tiles (0 if not sharded) |
| 26 | src0_num_tiles | uint32_t | Predicate tensor shard tile count (sharded) |

#### Writer (11 args)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | DRAM address of output tensor |
| 1 | num_tiles | uint32_t | Number of tiles to write for this core |
| 2 | start_id | uint32_t | Starting tile ID for output |
| 3 | dst_shard_width | uint32_t | Shard width in tiles (0 if not sharded) |
| 4-8 | D, N, C, Ht, Wt | uint32_t | Output tensor dimensions |
| 9 | cND | uint32_t | Collapsed >5D dimension |
| 10 | padding | uint32_t | Unused (always 0) |

#### Compute (4 args)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles | uint32_t | Number of tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (Wt for COL_BCAST, HtWt for SCALAR_BCAST, 0 for NONE) |
| 2 | counter | uint32_t | Starting position within broadcast cycle |
| 3 | scalar_arg | uint32_t | Packed scalar value (bit-cast float for TTS, 0 for TTT) |

## Kernel Implementations

### Reader Kernels

LERP uses several reader kernels depending on variant and broadcast type:

#### TTT No-Broadcast / Outer-Broadcast Reader
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_reader_nosubtilebcast_ttt.cpp`
- **Key Logic**: Reads tiles from 3 input tensors using nested 6D loops (nD, D, N, C, Ht, Wt). Each tensor has independent stride calculations for broadcast support. Supports conditional sharding via `SRC_SHARDED_A/B/C` defines -- sharded tensors skip NoC reads entirely. Uses `TensorAccessor` for page-level address resolution.

#### TTS/TST No-Broadcast Reader
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_reader_nobcast_tst_tts.cpp`
- **Key Logic**: Simple linear tile loop reading 2 tensors (predicate + one operand) using the same tile ID for both. No broadcast logic needed.

#### Other Reader Variants
- **Col Broadcast TTT**: `ternary_reader_colbcast_ttt.cpp` - fills tiles with first column values for broadcast inputs
- **Row Broadcast TTT**: `ternary_reader_rowbcast_ttt.cpp` - fills tiles with first row values or passes through for LLK broadcast
- **Scalar Broadcast TTT**: `ternary_reader_scalar_ttt.cpp` - fills tiles with first element for (1,1) shaped inputs
- **TTS/TST broadcast variants**: `tts_tst_reader_col_bcast.cpp`, `tts_tst_reader_row_bcast.cpp`, `tst_tts_reader_scalar_bcast.cpp`, `tst_tts_reader_outer_bcast.cpp`

### Compute Kernels

#### TTT No-Broadcast Compute
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_ttt.cpp`
- **Key Logic**: Per-tile loop: waits for 3 input tiles on c_0/c_1/c_2, copies each to DST[0]/DST[1]/DST[2] via `copy_tile`, calls `lerp_tile_init()` then `lerp_tile<DataFormat>(0, 1, 2, 0)`, packs DST[0] to c_3. Pure SFPU execution path.

#### TTS/TST No-Broadcast Compute
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_tts_tst.cpp`
- **Key Logic**: Same as TTT but only 2 CBs. Copies predicate to DST[0], tensor operand to DST[1] (TTS) or DST[2] (TST), fills scalar into the remaining register using `fill_tile`. Compile-time arg `scalar_is_true` selects register assignment.

#### TTT Col/Scalar Broadcast Compute
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_col_scalar_bcast_ttt.cpp`
- **Key Logic**: Uses `freq` and `tile_start` runtime args to control broadcast tile reuse. Broadcast CBs are waited/popped outside the inner loop; non-broadcast CBs inside. `BCAST_A/B/C` defines control which inputs are broadcast.

#### TTT Row Broadcast Compute (bfloat16 only)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_row_bcast_ttt.cpp`
- **Key Logic**: Uses `unary_bcast<BroadcastType::ROW>` LLK to expand row-broadcast tiles into intermediate CBs (c_4/c_5/c_6), then performs the standard SFPU ternary op on the expanded tiles. Only used when all inputs are bfloat16 (`is_llk_bcast` check).

#### TTS/TST Col/Scalar Broadcast Compute
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_col_scalar_bcast_tts_tst.cpp`
- **Key Logic**: Combines broadcast-aware CB management with scalar fill. Uses `BCAST_A/B/C` defines similar to TTT broadcast but handles scalar fill for the missing tensor operand.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_writer_nobcast.cpp`
- **Key Logic**: Single writer kernel used for all variants and broadcast types. Nested 6D loop matching reader structure. For width-sharded outputs, adjusts tile offset by `(Wt - dst_shard_width)` after each tile row. Entire body is compiled out when `DST_SHARDED=1`.

## Implementation Notes

1. **Shared Kernel Framework**: LERP reuses the exact same kernel source files as WHERE and other ternary ops. The operation-specific behavior is injected via `TERNARY_SFPU_OP_INIT` and `TERNARY_SFPU_OP_FUNC` preprocessor defines, which resolve to `lerp_tile_init` and `lerp_tile<DataFormat>`.

2. **No TST Variant for LERP**: Unlike WHERE which supports TTT/TTS/TST, the kernel config map for LERP only registers TTT and TTS variants. This is because LERP semantically has `(input, end, weight)` -- making the weight a scalar (TTS) is natural, but making `end` a scalar while `weight` is a tensor (TST) is uncommon.

3. **Broadcast Type Detection**: The program factory determines broadcast type at the host level by comparing tensor shapes. For example, if `input` has shape `(1,1,32,1)` and `end` has shape `(1,1,32,32)`, this triggers `COL_BCAST`. The broadcast type controls kernel selection and runtime argument setup.

4. **LLK Row Broadcast Optimization**: For the TTT variant with ROW_BCAST and all-bfloat16 inputs, a special optimization uses the hardware `unary_bcast<BroadcastType::ROW>` LLK instruction instead of software tile filling. This requires 3 additional intermediate CBs (c_4, c_5, c_6).

5. **Program Caching**: The `override_runtime_arguments` method enables program reuse across calls with different data but same shapes/configs. It updates buffer addresses and tile counts without rebuilding kernels.

6. **FP32 Destination Accumulation**: When output dtype is FLOAT32, INT32, or UINT32, `fp32_dest_acc_en` is set and `UnpackToDestMode::UnpackToDestFp32` is configured for relevant CBs, ensuring full precision in the destination register file.

7. **Uneven Shard Fallback**: When any input or output tensor has uneven sharding (shard shape does not evenly divide the tensor), the factory falls back to interleaved mode (treating all tensors as if they were interleaved) to avoid deadlocks from cores having different shard sizes.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the ternary eltwise operation work in TTNN? What is the program factory structure for ternary operations like lerp? What kernels are used?"
   **Reason**: Needed initial architectural overview of ternary operation framework before diving into source code.
   **Key Findings**: Confirmed that LERP is one of four ternary op types (WHERE, LERP, ADDCMUL, ADDCDIV), shares the same program factory with operation-specific behavior via defines, supports TTT/TTS/TST variants, and uses SFPU functions `lerp_tile_init`/`lerp_tile<DataFormat>`.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.cpp` (kernel config map, lines 164-254)
   **Reason**: Needed to understand which kernel files map to which (op_type, variant, broadcast_type) combinations.
   **Key Information**: LERP registers 11 kernel configurations (5 TTT + 6 TTS), sharing the same kernel files as WHERE but with different compute defines.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp`
   **Reason**: Needed enum definitions for operation types, variants, and broadcast types.
   **Key Information**: LERP formula is documented as `out = input + weight * (end - input)`. Seven broadcast types are supported including NONE, OUTER, COL, ROW, SCALAR, SCALAR_A, SCALAR_B.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_device_operation.hpp`
   **Reason**: Needed to understand the operation attributes and tensor args structure.
   **Key Information**: Operation attributes include `ternary_op_type`, `ternary_variant`, `broadcast_type`, `scalar_input_a`, `scalar_input_b`, and `worker_grid`. The program factory caches reader/writer/compute kernel handles.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/lerp.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_lerp.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_lerp.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_ternary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `TERNARY_SFPU_OP_INIT()` which resolves to `lerp_tile_init()`, and `TERNARY_SFPU_OP_FUNC(0, 1, 2, 0)` which resolves to `lerp_tile<DataFormat>(0, 1, 2, 0)` -- both defined in the API header `lerp.h`.
2. `lerp_tile<DataFormat>` wraps `MATH((llk_math_eltwise_ternary_sfpu_lerp<APPROX, DST_ACCUM_MODE, data_format>(idst0, idst1, idst2, odst)))`, which calls the LLK dispatch function.
3. `llk_math_eltwise_ternary_sfpu_lerp` calls `_llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>()`, passing `sfpu::calculate_lerp<...>` as the callable and the four DST indices.
4. `_llk_math_eltwise_ternary_sfpu_params_` handles sync/stall, then iterates over 4 tile faces (in RC vector mode), calling `calculate_lerp(...)` for each face, with `TTI_SETRWC` between faces to advance the DEST pointer by 16 rows.
5. `calculate_lerp` is the core SFPU function -- it loops 8 iterations (one per pair of rows within a face), performing the lerp formula `in0 + in2 * (in1 - in0)` using SFPI vector operations on DEST registers.
6. `lerp_tile_init()` calls `llk_math_eltwise_ternary_sfpu_lerp_init<APPROX>()`, which calls `_llk_math_eltwise_ternary_sfpu_init_<SfpuType::lerp>()` to configure SFPU config registers and address modes.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_lerp.h
// (Blackhole and Wormhole versions are identical)

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_lerp( // APPROXIMATION_MODE is passed through but unused; ITERATIONS=8
    const uint dst_index_in0,  // input (start)
    const uint dst_index_in1,  // end
    const uint dst_index_in2,  // weight
    const uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");

    // Each tile in DEST occupies 64 rows; SFPI accesses even rows only (stride=2), so 32 SFPI-addressable rows per tile
    constexpr uint dst_tile_size_sfpi = 32;
    // lerp: out = input + weight * (end - input)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DEST tile 0
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DEST tile 1
        sfpi::vFloat in2 = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi]; // SFPLOAD from DEST tile 2
        sfpi::vFloat result = in0 + in2 * (in1 - in0); // SFPMUL + SFPADD sequence (see instruction breakdown below)
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result); // Round-to-nearest-even truncation for bf16 output
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE to DEST tile 0
        sfpi::dst_reg++; // INCRWC: advance DEST row pointer by SFP_DESTREG_STRIDE (2)
    }
}

}  // namespace ckernel::sfpu
```

The `float32_to_bf16_rne` helper used when `is_fp32_dest_acc_en == false`:

```cpp
// File: tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);          // No-op reinterpret (same register)
    sfpi::vUInt lsb = (bits >> 16) & 1;                             // SFPSHFT + SFPAND: extract bit 16
    bits = bits + 0x7fffU + lsb;                                    // SFPXIADD: add rounding bias
    bits = bits & 0xFFFF0000U;                                      // SFPAND: clear lower 16 bits
    return sfpi::reinterpret<sfpi::vFloat>(bits);                   // No-op reinterpret back
}

}  // namespace ckernel::sfpu
```

The parameters dispatch function (identical for Blackhole and Wormhole):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_ternary_sfpu_params.h

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_in2,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    LLK_ASSERT((dst_index_in0 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in0 exceeds max dest tiles");
    LLK_ASSERT((dst_index_in1 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in1 exceeds max dest tiles");
    LLK_ASSERT((dst_index_in2 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_in2 exceeds max dest tiles");
    LLK_ASSERT((dst_index_out < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "dst_index_out exceeds max dest tiles");

    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(0); // Set DEST write address, stall until SFPU ready

    // VectorMode::RC: process all 4 faces of the 32x32 tile (each face = 16x16)
    for (int face = 0; face < 4; face++)
    {
        std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, std::forward<Args>(args)...);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // Advance DEST pointer by 8
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // Advance DEST pointer by 8 (total +16 rows per face)
    }

    _llk_math_eltwise_ternary_sfpu_done_(); // Clear DEST addr, stall until SFPU done, clear addr mod base
}
```

The init function (identical for Blackhole and Wormhole):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_ternary_sfpu.h

template <SfpuType sfpu_op>
inline void eltwise_ternary_sfpu_configure_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    // ADDR_MOD_6 with dest.incr=2 is only configured for SfpuType::where, NOT for lerp
    if (sfpu_op == SfpuType::where)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_6);
    }
}

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_ternary_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();                          // SFPCONFIG: reset SFPU config register
    eltwise_ternary_sfpu_configure_addrmod<sfpu_op>();      // Configure ADDR_MOD_7 with all-zero increments
    math::reset_counters(p_setrwc::SET_ABD_F);              // SETRWC: reset all RWC counters (A, B, D, Fidelity)
}
```

### SFPU Instructions Used

| Instruction | SFPI Intrinsic | Usage in LERP |
|---|---|---|
| **SFPLOAD** | `__builtin_rvtt_sfpload` | Load 32 elements from a DEST register row into an LREG. Used 3 times per iteration to load `in0`, `in1`, `in2` from DST tiles 0, 1, 2. |
| **SFPSTORE** | `__builtin_rvtt_sfpstore` | Store 32 elements from an LREG back to a DEST register row. Used once per iteration to write the result to DST tile 0. |
| **SFPADD** | `__builtin_rvtt_sfpadd` | Lanewise FP32 addition: `VD = VB + VC`. Used for `in1 - in0` (subtraction via negated operand) and for the final `in0 + (in2 * diff)` addition. |
| **SFPMUL** | `__builtin_rvtt_sfpmul` | Lanewise FP32 multiplication: `VD = VA * VB`. Used for `in2 * (in1 - in0)`. |
| **SFPSHFT** | `__builtin_rvtt_sfpshft_i` | Logical right shift by immediate. Used in `float32_to_bf16_rne` to extract bit 16 (`bits >> 16`). |
| **SFPAND** | `__builtin_rvtt_sfpand` | Bitwise AND. Used in `float32_to_bf16_rne` twice: to isolate LSB (`& 1`) and to clear lower 16 bits (`& 0xFFFF0000`). |
| **SFPXIADD** | `__builtin_rvtt_sfpxiadd_i` | Unsigned integer addition with immediate. Used in `float32_to_bf16_rne` to add the rounding bias (`bits + 0x7fffU + lsb`). |
| **INCRWC** | `__builtin_rvtt_ttincrwc` | Increment DEST RWC pointer by `SFP_DESTREG_STRIDE` (2). Used via `dst_reg++` at the end of each iteration to advance to the next row pair. |
| **SETRWC** | `TTI_SETRWC` | Set RWC pointers directly. Used between faces to advance DEST pointer by 16 rows (2 x `SETRWC(CR_D, 8)`), and in init to reset all counters. |
| **STALLWAIT** | `TTI_STALLWAIT` | Stall until SFPU/MATH pipeline is ready. Used in `_start_` (stall SFPU on MATH) and `_done_` (stall CFG on SFPU completion). |
| **SFPCONFIG** | `TTI_SFPCONFIG` | Configure SFPU control register. Used in `_init_sfpu_config_reg()` to reset the SFPU configuration state. |

Note: When `is_fp32_dest_acc_en == true` (FP32 output mode), the `float32_to_bf16_rne` conversion is skipped entirely, eliminating SFPSHFT, SFPAND, and SFPXIADD from the instruction stream. In that case, only SFPLOAD, SFPSTORE, SFPADD, SFPMUL, and INCRWC are used per iteration.

### SFPU Register Usage

**DEST Register File (shared between FPU and SFPU):**
- **DST[0]** (rows 0-63): Holds the `input` (start) tile. Loaded from CB c_0 via `copy_tile`. Also serves as the output tile -- `calculate_lerp` writes results back here since `dst_index_out == 0`.
- **DST[1]** (rows 64-127): Holds the `end` tile. Loaded from CB c_1 via `copy_tile`.
- **DST[2]** (rows 128-191): Holds the `weight` tile. Loaded from CB c_2 via `copy_tile` (TTT) or filled with scalar via `fill_tile` (TTS).

**LREG (SFPU Local Registers):**
The SFPU has 8 local vector registers (LREG0-LREG7), each holding 32 FP32 elements. Within `calculate_lerp`, the compiler allocates LREGs for:
- `in0`: Loaded from DST[0] via SFPLOAD
- `in1`: Loaded from DST[1] via SFPLOAD
- `in2`: Loaded from DST[2] via SFPLOAD
- `result` / temporaries: Intermediate values from subtraction (`in1 - in0`), multiplication (`in2 * diff`), and addition (`in0 + product`)
- When `float32_to_bf16_rne` is inlined, additional LREG usage for `bits`, `lsb`, and intermediate rounding values

**RWC (Read/Write Counter) Registers:**
- **RWC.Dst**: The DEST row pointer. Starts at 0 (set by `_llk_math_eltwise_ternary_sfpu_start_`), auto-incremented by 2 via `dst_reg++` (INCRWC) within each iteration, and advanced by 16 via `SETRWC(CR_D, 8)` x2 between faces. After all 4 faces: 4 * (8*2 + 16) = 128 rows total traversed, but RWC wraps modulo the active DEST region.

### Address Mode Configuration

LERP uses `SfpuType::lerp` when calling `_llk_math_eltwise_ternary_sfpu_init_`, which configures:

**ADDR_MOD_7** (the only address mode set for lerp):
| Field | Value | Description |
|---|---|---|
| `srca.incr` | 0 | No SrcA auto-increment (SrcA not used by SFPU) |
| `srcb.incr` | 0 | No SrcB auto-increment (SrcB not used by SFPU) |
| `dest.incr` | 0 | No DEST auto-increment from address mode (manual control via INCRWC/SETRWC) |

Unlike WHERE (which also sets `ADDR_MOD_6` with `dest.incr = 2` for its conditional store pattern), LERP only needs `ADDR_MOD_7` with all-zero increments. This is because LERP performs unconditional load/store on every row -- the DEST pointer advancement is handled entirely by explicit `dst_reg++` (INCRWC) calls within the loop and `SETRWC` calls between faces.

The address mode configuration is **identical** for Wormhole and Blackhole. The only difference between the two architectures in the ternary SFPU flow is that Wormhole's `_llk_math_eltwise_ternary_sfpu_start_` calls `math::set_addr_mod_base()` (which selects ADDR_MOD_4..7 as the active bank), while Blackhole skips this call (and instead uses `TTI_SETC16(2, 0)` in `_done_` to clear the addr mod base). The net effect is the same: ADDR_MOD_7 is active during SFPU execution.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the ternary eltwise operation work in TTNN? What is the program factory structure for ternary operations like lerp? What kernels are used?"
   **Reason**: Needed initial architectural overview of ternary operation framework before diving into source code.
   **Key Findings**: Confirmed that LERP is one of four ternary op types (WHERE, LERP, ADDCMUL, ADDCDIV), shares the same program factory with operation-specific behavior via defines, supports TTT/TTS/TST variants, and uses SFPU functions `lerp_tile_init`/`lerp_tile<DataFormat>`.

2. **Query**: "How does dst_reg indexing work in SFPI? What SFPU instructions are generated by vFloat loads from dst_reg, vFloat arithmetic (add, subtract, multiply), and stores back to dst_reg? What does the dst_reg++ increment do?"
   **Reason**: Needed to understand the SFPI-to-instruction mapping for the core LERP kernel operations.
   **Key Findings**: `dst_reg[index]` loads use `SFPLOAD`, stores use `SFPSTORE`, addition maps to `SFPADD`, multiplication maps to `SFPMUL`, subtraction is `SFPADD` with negated operand. `dst_reg++` emits `INCRWC` which advances the DEST pointer by `SFP_DESTREG_STRIDE` (2).

3. **Query**: "What SFPU instructions correspond to dst_reg loads (LOADDST), dst_reg stores (STOREDST), vFloat addition/subtraction/multiplication in SFPI? How does SETRWC work for advancing the dest register pointer between faces?"
   **Reason**: Needed ISA-level confirmation of instruction opcodes and SETRWC semantics.
   **Key Findings**: Confirmed SFPLOAD (opcode 0x84-range), SFPSTORE, SFPADD (0x85), SFPMUL (0x86), SFPMAD (0x84). SETRWC (opcode 0x37) directly sets RWC pointer values including Dst.

4. **Query**: "How do vUInt bitwise operations (shift right, AND, addition) map to SFPU instructions? What instructions are used for reinterpret between vFloat and vUInt?"
   **Reason**: Needed to understand instruction mapping for the `float32_to_bf16_rne` helper function.
   **Key Findings**: Shift right maps to `SFPSHFT` (`sfpshft_i`), AND maps to `SFPAND`, integer addition maps to `SFPXIADD` (`sfpxiadd_i`/`sfpxiadd_v`). `reinterpret<>` is a zero-cost cast -- no instruction emitted.

### Confluence References

Not consulted for this analysis. The SFPU instructions used in LERP (SFPLOAD, SFPSTORE, SFPADD, SFPMUL, SFPSHFT, SFPAND, SFPXIADD) are well-documented in DeepWiki and the source code.

### Glean References

Not consulted for this analysis. No confidential hardware specifications were needed beyond what was available in the open-source codebase and DeepWiki.
