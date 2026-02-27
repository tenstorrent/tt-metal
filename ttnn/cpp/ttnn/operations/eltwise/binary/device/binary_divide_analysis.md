# Binary Divide Operation Implementation Analysis

## Overview

The binary divide operation (`ttnn.div`) computes element-wise division of two tensors: `output = input_a / input_b`. This operation is part of the broader binary elementwise framework in TTNN, which provides a unified infrastructure for operations like ADD, SUB, MUL, and DIV.

**Program Factory Path**: `/localdev/vignjatijevic/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_program_factory.cpp`

**Key Insight**: The divide operation has two implementation paths:
1. **Legacy Path (FPU)**: Transforms divide into `a * recip(b)` using the FPU multiply unit
2. **Modern Path (SFPU)**: Uses the SFPU's dedicated `div_binary_tile` function directly

The routing decision is made in `binary.cpp` based on tensor properties (sharding, data format, broadcasting requirements).

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `tensor_volume / TILE_HW` where `TILE_HW = 1024` |
| **Loop structure** | Block-based: outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_size` tiles |

Each core processes a contiguous range of tiles. The total work is divided evenly across cores when possible, with remainder tiles distributed to the first cores (core_group_1 gets more tiles than core_group_2).

## Tensor Format and Layout

### Input Tensors

| Property | Input A | Input B |
|----------|---------|---------|
| **Logical shape** | [N, C, H, W] (up to 4D) or [D, N, C, H, W] (up to 5D+) | [N, C, H, W] or broadcastable shape |
| **Dimension convention** | NHWC-style (innermost is W) | NHWC-style |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, UINT16 | Same as A (or compatible) |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | Broadcast result of A and B shapes |
| **Dimension convention** | NHWC-style |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Typically same as input A |

### Layout Transformations

For the divide operation specifically:
- **Legacy FPU path**: Applies `RECIP` (reciprocal) to input B before the multiply operation
- **SFPU path**: No pre-transformation; direct division in hardware
- **Integer division (INT32)**: Uses dedicated `div_int32_tile` which outputs FLOAT32

## Data Flow Pattern

### Legacy Path (FPU with RECIP preprocessing)

```
DRAM (A, B) --> Reader Kernel --> CB_in0, CB_in1
                                       |
                                       v
                                 Compute Kernel:
                                 1. Wait for tiles in CB_in0, CB_in1
                                 2. Apply RECIP to B tiles (SFPU_OP_INIT_PRE_IN1_0)
                                    - Copy B tile to DST
                                    - Execute recip SFPU operation
                                    - Pack to intermediate CB_c4
                                 3. Wait for A tiles in CB_inp0 and reciprocated B tiles in CB_inp1
                                 4. Execute mul_tiles(A, 1/B) using FPU
                                 5. Pack result to CB_out0
                                       |
                                       v
                                 Writer Kernel --> DRAM (Output)
```

### Modern SFPU Path

```
DRAM (A, B) --> Reader Kernel --> CB_c0, CB_c1
                                       |
                                       v
                                 Compute Kernel:
                                 1. Wait for tiles in CB_c0, CB_c1
                                 2. Copy tiles to DST registers:
                                    - A tiles go to even DST indices (i*2)
                                    - B tiles go to odd DST indices (i*2+1)
                                 3. Execute div_binary_tile(dst_a, dst_b, dst_out)
                                 4. Pack result to CB_c2
                                       |
                                       v
                                 Writer Kernel --> DRAM (Output)
```

## Circular Buffer Configuration

### Legacy Path Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_src1 | Input B staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile | Double | Reader | Compute | Block |
| c_2 | cb_output | Output staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile | Double | Compute | Writer | Block |
| c_3 | cb_interm0 | A pre-processing intermediate | max_block_size tiles | 1 tile | Single | Compute | Compute | Block |
| c_4 | cb_interm1 | B pre-processing (RECIP result) | max_block_size tiles | 1 tile | Single | Compute | Compute | Block |

**Note**: For DIV operation, only `c_4` is created (for RECIP pre-processing of input B). The `c_3` buffer is used when input A requires pre-processing (e.g., LOGADDEXP operations).

### Modern SFPU Path Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_src_b | Input B | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double | Reader | Compute | Block |
| c_2 | cb_out | Output | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double | Compute | Writer | Block |
| c_3 | cb_post_lhs | LHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_4 | cb_post_rhs | RHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |

## Pipeline Pattern Summary

- **Input CBs (c_0, c_1)**: Double-buffered (capacity = 2 * block_size) allows overlap of reader and compute
- **Output CB (c_2)**: Double-buffered allows overlap of compute and writer
- **Intermediate CBs (c_3, c_4)**: Single-buffered as they are internal to the compute kernel

## Index Calculations

### Tile Index Mapping (Legacy Path)

For interleaved tensors, tiles are indexed linearly:
```
tile_id = start_id + local_tile_index
```

For sharded tensors (block or width sharded):
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```

### Tile Index Mapping (Modern binary_ng Path)

The modern path uses a 5D+ index calculation with strides:
```cpp
tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride
            + start_c * c_stride + start_th * Wt + tw
```

Where dimensions are decomposed as:
- `cND`: Collapsed dimensions > 5
- `D`: 5th dimension
- `N`: 4th dimension (batch)
- `C`: 3rd dimension (channel)
- `Ht`: Height in tiles
- `Wt`: Width in tiles

## Memory Access Patterns

### Read Pattern

**Reader Kernel** (`reader_binary_interleaved_start_id.cpp`):
- **Non-sharded path**: Sequential tile reads using `noc_async_read_tile` with TensorAccessor
- **Block/width sharded path**: Row-by-row traversal with stride calculation based on `num_cores_y * block_width`
- **Sharded input**: Uses `cb_reserve_back` + `cb_push_back` with globally allocated buffer (no actual read)

```cpp
// Non-sharded sequential read
for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
    cb_reserve_back(cb_id_in0, onetile);
    noc_async_read_tile(tile_id, s0, l1_write_addr_in0);
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, onetile);
}
```

### Write Pattern

**Writer Kernel** (`writer_unary_interleaved_start_id.cpp`):
- Sequential tile writes using `noc_async_write_page`
- One tile at a time with explicit barrier after each write

```cpp
for (uint32_t i = start_id; i < end_id; ++i) {
    cb_wait_front(cb_id_out, onepage);
    noc_async_write_page(i, s, l1_read_addr);
    noc_async_writes_flushed();
    cb_pop_front(cb_id_out, onepage);
}
```

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D) |
| **Grid dimensions** | Determined by `worker_grid` parameter or device compute grid |
| **Total cores** | `compute_with_storage_grid_size.x * compute_with_storage_grid_size.y` |
| **Work per core** | `num_tiles / num_cores` (balanced) or `num_tiles_per_core_group_1/2` (with remainder) |
| **Load balancing** | Two-tier: core_group_1 gets `ceil(num_tiles/num_cores)`, core_group_2 gets `floor(num_tiles/num_cores)` |

The work splitting uses `tt::tt_metal::split_work_to_cores()`:
```cpp
std::tie(num_cores, all_cores, core_group_1, core_group_2,
         num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
    split_work_to_cores(compute_with_storage_grid_size, num_tiles, row_major);
```

## Arguments

### Compile-Time Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block/width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs | varies | Memory layout info for src0 (if not sharded) |
| N+ | TensorAccessorArgs | varies | Memory layout info for src1 (if not sharded) |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Address of input A buffer |
| 1 | src1_addr | uint32_t | Address of input B buffer |
| 2 | num_tiles | uint32_t | Total tiles for this core to read |
| 3 | start_id | uint32_t | Starting tile index |
| 4 | block_height | uint32_t | Shard height in tiles (sharded only) |
| 5 | block_width | uint32_t | Shard width in tiles (sharded only) |
| 6 | num_cores_y | uint32_t | Number of cores in Y dimension (block sharded) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Tiles per block |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile index |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_binary_interleaved_start_id | RISCV_0 | NOC0 | DRAM (src0, src1) | CB_c0, CB_c1 | Read tiles from both inputs using TensorAccessor |

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`

**Key Logic**:
- Handles both sharded and interleaved inputs via `IN0_SHARDED`/`IN1_SHARDED` defines
- For sharded inputs: Simply pushes the pre-loaded shard to CB
- For interleaved: Uses TensorAccessor for index-to-address mapping
- Supports block/width sharded traversal pattern

### Compute Kernel (Legacy FPU Path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| eltwise_binary_kernel | RISCV_2 (Unpack/Math/Pack) | N/A | CB_inp0, CB_inp1 | CB_out0 | RECIP(B), MUL(A, 1/B) |

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp`

**Key Logic for DIV**:
1. When `SFPU_OP_INIT_PRE_IN1_0` is defined (set to RECIP for DIV):
   - Copy B tile from CB_in1 to DST
   - Execute `SFPU_OP_FUNC_PRE_IN1_0` (recip operation)
   - Pack reciprocated tile to CB_c4 (intermediate)
2. Wait for A tiles in CB_inp0 and 1/B tiles in CB_inp1
3. Execute `ELTWISE_OP` (mul_tiles) via FPU
4. Pack result to output CB

### Compute Kernel (Modern SFPU Path)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| eltwise_binary_sfpu_no_bcast | RISCV_2 (Unpack/Math/Pack) | N/A | CB_c0, CB_c1 | CB_c2 | div_binary_tile |

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

**Key Logic**:
```cpp
// Copy both input tiles to DST
copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
copy_tile(cb_post_lhs, i, i * 2);     // A -> DST[0]
copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
copy_tile(cb_post_rhs, i, i * 2 + 1); // B -> DST[1]

// Execute SFPU binary operation
BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);  // div_binary_tile(0, 1, 0)

// Pack result
pack_tile(i * 2, cb_out);
```

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB_out | DRAM | Write output tiles |

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

**Key Logic**:
- Sequential tile-by-tile writes using TensorAccessor
- Explicit barrier (`noc_async_writes_flushed`) after each write
- Supports sharded output via `OUT_SHARDED` define

## Implementation Notes

### Division Implementation Strategy

The divide operation uses two fundamentally different approaches:

1. **Legacy FPU Path** (used for sharded block-format tensors):
   - Transform: `A / B = A * (1/B)`
   - Uses FPU's multiply unit (faster for certain data formats)
   - Requires intermediate buffer for reciprocated B values
   - Implemented via compile-time defines that inject RECIP preprocessing

2. **Modern SFPU Path** (default for most cases):
   - Direct division using `div_binary_tile` SFPU function
   - Supports INT32 division with `div_int32_tile`
   - More straightforward data flow without intermediate buffers

### Integer Division Special Case

When both inputs are INT32:
```cpp
if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
    new_defines.insert({"BINOP_INIT", "div_int32_tile_init();"});
    op_name = "div_int32_tile";
}
```
The output dtype for integer division is forced to FLOAT32 for accuracy.

### Routing Decision Logic

The choice between legacy and modern paths is made in `binary.cpp`:
```cpp
if (use_legacy ? *use_legacy :
    binary::is_legacy_only(lhs, rhs, memory_config, output, lhs_activations, rhs_activations)
    and (not detail::is_binary_ng_only(lhs, rhs))) {
    // Use legacy path with FPU
    return ttnn::prim::binary(...);
}
// Otherwise use modern SFPU path
return ttnn::prim::binary_ng(...);
```

Legacy-only conditions include:
- Any sharded tensor with block format (BFLOAT4_B, BFLOAT8_B)
- Subtile broadcasting with block format inputs

### Preprocessing Defines for DIV (Legacy Path)

From `binary_op_utils.cpp`:
```cpp
case BinaryOpType::DIV:
    // Divide by a non-zero tensor
    defines.merge(get_defines(UnaryOpType::RECIP, std::nullopt, "PRE_IN1_0"));
    op_name = "mul_tiles";
    op_binary_type = "EltwiseBinaryType::ELWMUL";
    break;
```

This sets up:
- `SFPU_OP_INIT_PRE_IN1_0` = reciprocal initialization
- `SFPU_OP_FUNC_PRE_IN1_0` = reciprocal function call
- `ELTWISE_OP` = `mul_tiles`

### Fused Activations Support

The divide operation supports fused post-activations (e.g., RELU) via:
- `PACK_RELU` define for hardware-accelerated ReLU during pack
- `SFPU_OP_CHAIN_0` for chained SFPU operations

## External Knowledge Sources

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding the three-kernel architecture (reader/compute/writer) and circular buffer synchronization patterns
   **Key Information**: Kernels coordinate via circular buffers with `cb_wait_front`/`cb_pop_front` (consumer) and `cb_reserve_back`/`cb_push_back` (producer) primitives

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding tensor representation in memory and tile layout
   **Key Information**: Tiles are 32x32 elements organized into 16x16 faces; tensors stored as 2D buffers with outer dimensions collapsed

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Understanding how DIV operation is transformed into RECIP + MUL
   **Key Information**: The `get_defines()` function sets up preprocessing macros that inject RECIP on input B

4. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: Understanding the modern SFPU path and OpConfig system
   **Key Information**: DIV maps directly to `SfpuBinaryOp::DIV` with `div_binary_tile` function, or uses FPU MUL with RECIP preprocessing for non-SFPU path
