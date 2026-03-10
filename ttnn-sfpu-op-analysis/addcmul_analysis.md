# ADDCMUL Operation Analysis

## Operation Overview

**Operation**: ADDCMUL (fused multiply-add)
**Mathematical Formula**: `output = input_a + (value * input_b * input_c)`
**Category**: Ternary element-wise SFPU operation
**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

ADDCMUL performs a fused element-wise operation that multiplies a scalar constant (`value`) with two input tensors (`input_b` and `input_c`), then adds the result to a third input tensor (`input_a`). This is the PyTorch-equivalent `torch.addcmul` operation. It is implemented as a single SFPU kernel dispatch to avoid the overhead of three separate operations (multiply, multiply, add).

### Supported Data Types
- **Float32**: Full 32-bit floating point, uses `DataFormat::Float32` in SFPU kernel
- **BFloat16 / Float16_b**: 16-bit brain floating point, uses `DataFormat::Float16_b` in SFPU kernel
- **Bfp8_b**: 8-bit block floating point (supported in SFPU static assert, but typically not surfaced to user)
- **INT32**: Special integer path with a separate compute kernel (`ternary_addcmul_int_sfpu.cpp`) that uses `fill_tile_int`, `mul_int_tile`, and `add_int_tile` instead of the SFPU addcmul instruction sequence

### Supported Variants
- **TTT (Tensor-Tensor-Tensor)**: All three inputs are tensors, plus a scalar `value`
- Other variants (TTS, TST, TSS) are not currently mapped in the kernel config for ADDCMUL

### Supported Broadcast Types (TTT only)
| Broadcast Type | Compute Kernel | Description |
|---|---|---|
| `NONE` | `ComputeNoBcastAddcOp` | No broadcasting, all tensors same shape |
| `OUTER_BCAST` | `ComputeNoBcastAddcOp` | Outer dimensions differ, last two dims match |
| `ROW_BCAST` | `ComputeNoBcastAddcOp` (or `ComputeRowBcastAddcOp` for bfloat16) | Height dimension broadcast |
| `SCALAR_BCAST` | `ComputeBcastAddcOp` | One or more tensors are scalar (1,1) in last two dims |
| `COL_BCAST` | `ComputeBcastAddcOp` | Width dimension broadcast |

---

## Program Factory Structure

### File: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

The ternary program factory is a shared infrastructure used by all ternary operations (WHERE, LERP, ADDCMUL, ADDCDIV). It is not ADDCMUL-specific but is parameterized via operation type and compile-time defines.

### Operation Registration Flow

1. `AddcmulOperation::invoke()` is the Python-facing entry point
2. It determines broadcast type and calls `ttnn::prim::ternary(TernaryOpType::ADDCMUL, ...)`
3. This routes to `TernaryDeviceOperation` which uses `TernaryProgramFactory`
4. The factory creates a `Program` with reader, compute, and writer kernels

### Kernel Selection

Kernel selection uses a hash map (`kernel_config_map`) with composite keys of `(TernaryOpType, TernaryVariant, TernaryBroadcastType)`. For ADDCMUL:

```
{ADDCMUL, TTT, NONE}         -> {ReaderNoBcastTTT, ComputeNoBcastAddcOp, WriterNoBcastTernary}
{ADDCMUL, TTT, OUTER_BCAST}  -> {ReaderOuterBcastTTT, ComputeNoBcastAddcOp, WriterNoBcastTernary}
{ADDCMUL, TTT, ROW_BCAST}    -> {ReaderRowBcastTTT, ComputeNoBcastAddcOp, WriterNoBcastTernary}
{ADDCMUL, TTT, SCALAR_BCAST} -> {ReaderScalarBcastTTT, ComputeBcastAddcOp, WriterNoBcastTernary}
{ADDCMUL, TTT, COL_BCAST}    -> {ReaderColBcastTTT, ComputeBcastAddcOp, WriterNoBcastTernary}
```

### FPU vs SFPU Path Selection

The program factory has a special `is_fpu` flag for ADDCMUL/ADDCDIV that determines whether to use FPU-based kernels. The FPU path is selected when:
- All three input tensors have the same data type
- The data type is NOT Float32, INT32, or UINT32 (i.e., it is BFloat16 or similar)

When `is_fpu` is true and broadcast type is `ROW_BCAST`, the kernel name is overridden to `ComputeRowBcastAddcOp`, which maps to `ternary_addc_ops_fpu_rowbcast.cpp`.

### INT32 Override

When the operation is ADDCMUL and the output dtype is INT32, the compute kernel path is overridden via `override_addcmul_compute_kernel()`:
- `ComputeNoBcastAddcOp` -> `ternary_addcmul_int_sfpu.cpp`
- `ComputeBcastAddcOp` / `ComputeRowBcastAddcOp` -> `ternary_addcmul_int_sfpu_bcast.cpp`

### Compute Defines

For ADDCMUL, `get_compute_defines()` sets:
```cpp
defines["TERNARY_SFPU_OP_INIT"] = "addcmul_tile_init";
defines["TERNARY_SFPU_OP_FUNC"] = "addcmul_tile<DataFormat::Float32>";  // or Float16_b
```

These defines are used by the compute kernel to dispatch the correct SFPU operation at compile time.

---

## Circular Buffer Configuration

| CB Index | Name | Purpose | Size |
|---|---|---|---|
| `c_0` | `predicate_tensor_cb` | Input A (`input_a`) | `num_tiles_per_shard` or 2 tiles |
| `c_1` | `value_true_tensor_cb` | Input B (`input_b`) | `num_tiles_per_shard` or 2 tiles |
| `c_2` | `value_false_tensor_cb` | Input C (`input_c`) | `num_tiles_per_shard` or 2 tiles |
| `c_3` | `output_tensor_cb` | Output | `num_tiles_per_shard` or 2 tiles |
| `c_4` | (row bcast scratch) | Used only for TTT ROW_BCAST | 2 tiles |
| `c_5` | (row bcast scratch) | Used only for TTT ROW_BCAST | 2 tiles |
| `c_6` | (row bcast scratch) | Used only for TTT ROW_BCAST | 2 tiles |

CB sizing uses 2 tiles per CB for interleaved mode (double-buffering) or the shard volume for sharded mode.

### Unpack to Dest Mode

Each input CB's unpack mode is set based on the tensor data type:
- `DataType::FLOAT32` -> `UnpackToDestMode::UnpackToDestFp32`
- All others -> `UnpackToDestMode::Default`

`fp32_dest_acc_en` is set to `true` when the output format is UInt32, Int32, or Float32, enabling 32-bit accumulation in the destination register.

---

## Runtime Arguments

### Compute Kernel Runtime Args (4 args per core)

| Index | Name | Description |
|---|---|---|
| 0 | `num_tiles` | Number of tiles this core processes |
| 1 | `tile_freq` | Broadcast frequency (0 for NONE/OUTER/ROW) |
| 2 | `tile_start` | Start offset within broadcast cycle |
| 3 | `scalar_arg` | Packed scalar value (bit-cast float/int to uint32_t) |

The scalar `value` parameter is packed via `pack_scalar_runtime_arg()`:
- For float dtype: bit-cast float to uint32_t
- For INT32 dtype: cast float to int32_t, then bit-cast to uint32_t
- For int32_t/uint32_t scalars: bit-cast directly or cast to float first depending on output dtype

### Reader Runtime Args (27 args per core, TTT variant)

| Index | Name | Description |
|---|---|---|
| 0 | `src0_addr` | Predicate tensor buffer address |
| 1 | `src1_addr` | True tensor buffer address |
| 2 | `src2_addr` | False tensor buffer address |
| 3 | `dst_num_tiles` | Number of tiles per core |
| 4 | `start_tile_id` | Starting tile index for this core |
| 5-8 | `nD/d/n/c_stride` | Predicate tensor dimension strides |
| 9-14 | `D/N/C/Ht/Wt/cND` | Output tensor dimensions |
| 15-18 | `nD/d/n/c_stride_b` | True tensor dimension strides |
| 19 | `src1_num_tiles` | True tensor shard tile count |
| 20-23 | `nD/d/n/c_stride_c` | False tensor dimension strides |
| 24 | `src2_num_tiles` | False tensor shard tile count |
| 25 | `dst_shard_width` | Shard width in tiles |
| 26 | `src0_num_tiles` | Predicate tensor shard tile count |

### Writer Runtime Args (11 args per core)

| Index | Name | Description |
|---|---|---|
| 0 | `dst_addr` | Output buffer address |
| 1 | `dst_num_tiles` | Number of tiles to write |
| 2 | `start_tile_id` | Starting tile index |
| 3 | `dst_shard_width` | Shard width in tiles |
| 4-9 | `D/N/C/Ht/Wt/cND` | Output dimensions |

---

## Kernel Implementations

### Reader Kernel (No Broadcast, TTT)

**File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_reader_nosubtilebcast_ttt.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);
    const uint32_t src2_addr = get_arg_val<uint32_t>(2);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(4);
    // Strides and dimensions for up to 5D tensor traversal
    const uint32_t nD_stride = get_arg_val<uint32_t>(5);
    const uint32_t d_stride = get_arg_val<uint32_t>(6);
    const uint32_t n_stride = get_arg_val<uint32_t>(7);
    const uint32_t c_stride = get_arg_val<uint32_t>(8);
    const uint32_t D = get_arg_val<uint32_t>(9);
    const uint32_t N = get_arg_val<uint32_t>(10);
    const uint32_t C = get_arg_val<uint32_t>(11);
    const uint32_t Ht = get_arg_val<uint32_t>(12);
    const uint32_t Wt = get_arg_val<uint32_t>(13);
    const uint32_t cND = get_arg_val<uint32_t>(14);
    const uint32_t nD_stride_b = get_arg_val<uint32_t>(15);
    const uint32_t d_stride_b = get_arg_val<uint32_t>(16);
    const uint32_t n_stride_b = get_arg_val<uint32_t>(17);
    const uint32_t c_stride_b = get_arg_val<uint32_t>(18);
    const uint32_t src1_num_tiles = get_arg_val<uint32_t>(19);
    const uint32_t nD_stride_c = get_arg_val<uint32_t>(20);
    const uint32_t d_stride_c = get_arg_val<uint32_t>(21);
    const uint32_t n_stride_c = get_arg_val<uint32_t>(22);
    const uint32_t c_stride_c = get_arg_val<uint32_t>(23);
    const uint32_t src2_num_tiles = get_arg_val<uint32_t>(24);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(25);
    const uint32_t src0_num_tiles = get_arg_val<uint32_t>(26);

    // CB IDs from compile-time args
    constexpr auto cb_id_src0 = get_compile_time_arg_val(0);   // input_a (predicate)
    constexpr auto cb_id_src1 = get_compile_time_arg_val(1);   // input_b (true)
    constexpr auto cb_id_src2 = get_compile_time_arg_val(2);   // input_c (false)

    // TensorAccessorArgs for each input tensor, chained in compile-time arg layout
    constexpr auto src0_args = TensorAccessorArgs<3, 0>();
    constexpr auto src1_args =
        TensorAccessorArgs<src0_args.next_compile_time_args_offset(), src0_args.next_common_runtime_args_offset()>();
    constexpr auto src2_args =
        TensorAccessorArgs<src1_args.next_compile_time_args_offset(), src1_args.next_common_runtime_args_offset()>();

    // For sharded inputs, reserve and push back the entire shard at once
    // (data is already in L1, just make it visible to CBs)
#if SRC_SHARDED_A
    cb_reserve_back(cb_id_src0, src0_num_tiles);
    cb_push_back(cb_id_src0, src0_num_tiles);
#else
    const uint32_t src0_tile_bytes = get_tile_size(cb_id_src0);
    const auto src0 = TensorAccessor(src0_args, src0_addr, src0_tile_bytes);
#endif
#if SRC_SHARDED_B
    cb_reserve_back(cb_id_src1, src1_num_tiles);
    cb_push_back(cb_id_src1, src1_num_tiles);
#else
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_src1);
    const auto src1 = TensorAccessor(src1_args, src1_addr, src1_tile_bytes);
#endif
#if SRC_SHARDED_C
    cb_reserve_back(cb_id_src2, src2_num_tiles);
    cb_push_back(cb_id_src2, src2_num_tiles);
#else
    const uint32_t src2_tile_bytes = get_tile_size(cb_id_src2);
    const auto src2 = TensorAccessor(src2_args, src2_addr, src2_tile_bytes);
#endif

    // For non-sharded inputs, perform nested loop tile reading via NoC
#if !SRC_SHARDED_A || !SRC_SHARDED_B || !SRC_SHARDED_C
    constexpr uint32_t onetile = 1;
    constexpr bool has_sharding = get_compile_time_arg_val(src2_args.next_compile_time_args_offset()) == 1;
    const uint32_t HtWt = Ht * Wt;

    // Decompose start_tile_id into multi-dimensional offsets
    const uint32_t tiles_per_n = C * HtWt;
    const uint32_t tiles_per_d = N * tiles_per_n;
    const uint32_t tiles_per_nd = D * tiles_per_d;
    const uint32_t offset_nd = start_tile_id % tiles_per_nd;
    const uint32_t offset_d = offset_nd % tiles_per_d;
    const uint32_t offset_n = offset_d % tiles_per_n;
    const uint32_t offset_c = offset_n % HtWt;
    uint32_t start_nd = start_tile_id / tiles_per_nd;
    uint32_t start_d = offset_nd / tiles_per_d;
    uint32_t start_n = offset_d / tiles_per_n;
    uint32_t start_c = offset_n / HtWt;
    uint32_t start_th = offset_c / Wt;
    uint32_t start_tw = offset_c % Wt;
    uint32_t end_tw = has_sharding ? start_tw + dst_shard_width : Wt;

    // Per-tensor tile offsets with independent strides (enables broadcast)
    uint32_t tile_offset =
        start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt;
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t tile_offset_b =
        start_nd * nD_stride_b + start_d * d_stride_b + start_n * n_stride_b + start_c * c_stride_b + start_th * Wt;
    uint32_t next_c_shift_b = c_stride_b - HtWt;
    uint32_t next_n_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_d_shift_b = d_stride_b - n_stride_b * N;
    uint32_t next_nd_shift_b = nD_stride_b - d_stride_b * D;

    uint32_t tile_offset_c =
        start_nd * nD_stride_c + start_d * d_stride_c + start_n * n_stride_c + start_c * c_stride_c + start_th * Wt;
    uint32_t next_c_shift_c = c_stride_c - HtWt;
    uint32_t next_n_shift_c = n_stride_c - c_stride_c * C;
    uint32_t next_d_shift_c = d_stride_c - n_stride_c * N;
    uint32_t next_nd_shift_c = nD_stride_c - d_stride_c * D;

    // 6-nested-loop tile traversal over nD, D, N, C, Ht, Wt dimensions
    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
                            // Async read each non-sharded input tile from DRAM to L1 CB
#if !SRC_SHARDED_A
                            cb_reserve_back(cb_id_src0, onetile);
                            uint32_t l1_write_addr_src0 = get_write_ptr(cb_id_src0);
                            noc_async_read_page(tile_offset + tw, src0, l1_write_addr_src0);
#endif
#if !SRC_SHARDED_B
                            cb_reserve_back(cb_id_src1, onetile);
                            uint32_t l1_write_addr_src1 = get_write_ptr(cb_id_src1);
                            noc_async_read_page(tile_offset_b + tw, src1, l1_write_addr_src1);
#endif
#if !SRC_SHARDED_C
                            cb_reserve_back(cb_id_src2, onetile);
                            uint32_t l1_write_addr_src2 = get_write_ptr(cb_id_src2);
                            noc_async_read_page(tile_offset_c + tw, src2, l1_write_addr_src2);
#endif
#if !SRC_SHARDED_A || !SRC_SHARDED_B || !SRC_SHARDED_C
                            noc_async_read_barrier();  // wait for all reads to complete
#endif
#if !SRC_SHARDED_A
                            cb_push_back(cb_id_src0, onetile);
#endif
#if !SRC_SHARDED_B
                            cb_push_back(cb_id_src1, onetile);
#endif
#if !SRC_SHARDED_C
                            cb_push_back(cb_id_src2, onetile);
#endif
                        }
                        if constexpr (!has_sharding) {
                            start_tw = 0;  // next row resets to column 0
                        }
                        tile_offset += Wt;
                        tile_offset_b += Wt;
                        tile_offset_c += Wt;
                    }
                    tile_offset += next_c_shift;
                    tile_offset_b += next_c_shift_b;
                    tile_offset_c += next_c_shift_c;
                }
                tile_offset += next_n_shift;
                tile_offset_b += next_n_shift_b;
                tile_offset_c += next_n_shift_c;
            }
            tile_offset += next_d_shift;
            tile_offset_b += next_d_shift_b;
            tile_offset_c += next_d_shift_c;
        }
        tile_offset += next_nd_shift;
        tile_offset_b += next_nd_shift_b;
        tile_offset_c += next_nd_shift_c;
    }
#endif
}
```

**Key design aspects**:
- Each input tensor has independent strides, allowing different broadcast patterns per tensor
- Sharded tensors skip the NoC read entirely -- the data is already in L1 and just needs CB reservation
- The 6-level nested loop supports up to rank-6 tensors (higher ranks are collapsed into `nD`)

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_writer_nobcast.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(3);
    const uint32_t D = get_arg_val<uint32_t>(4);
    const uint32_t N = get_arg_val<uint32_t>(5);
    const uint32_t C = get_arg_val<uint32_t>(6);
    const uint32_t Ht = get_arg_val<uint32_t>(7);
    const uint32_t Wt = get_arg_val<uint32_t>(8);
    const uint32_t cND = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1, 0>();

#if !DST_SHARDED
    // For non-sharded output: write tiles to DRAM via NoC
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    constexpr bool has_sharding = get_compile_time_arg_val(dst_args.next_compile_time_args_offset()) == 1;
    const uint32_t HtWt = Ht * Wt;

    // Same 6-nested-loop structure as reader, but writes output tiles
    const uint32_t tiles_per_n = C * HtWt;
    const uint32_t tiles_per_d = N * tiles_per_n;
    const uint32_t tiles_per_nd = D * tiles_per_d;
    // ... (dimension decomposition identical to reader)

    uint32_t num_tiles_written = 0;
    uint32_t dst_tile_offset = start_tile_id;

    for (uint32_t nd = start_nd; nd < cND && num_tiles_written < dst_num_tiles; ++nd, start_d = 0) {
        // ... nested loops ...
        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_written < dst_num_tiles;
             ++tw, ++num_tiles_written) {
            cb_wait_front(cb_id_out, onetile);       // wait for compute to produce a tile
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            noc_async_write_page(dst_tile_offset + num_tiles_written, s, l1_read_addr);
            noc_async_write_barrier();               // wait for write to complete
            cb_pop_front(cb_id_out, onetile);        // free the CB slot
        }
        // ...
    }
#endif
    // When DST_SHARDED, output is already in L1 -- nothing to do
}
```

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File (No Broadcast)
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu.cpp`

#### Annotated Compute Kernel Source (No Broadcast)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/addcmul.h"    // provides addcmul_tile<> and addcmul_tile_init()
#include "api/compute/eltwise_unary/addcdiv.h"    // also included since kernel is shared with ADDCDIV

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);       // total tiles to process on this core
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);       // packed scalar "value" parameter
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // always 1

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a: the addend
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b: first multiplicand
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c: second multiplicand
    constexpr auto cb_out = tt::CBIndex::c_3;   // output

    // Initialize unpack/pack hardware for the input->output CB pair
    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Wait for reader to produce one tile in each input CB
        cb_wait_front(cb_in0, num_tiles_per_cycle);
        cb_wait_front(cb_in1, num_tiles_per_cycle);
        cb_wait_front(cb_in2, num_tiles_per_cycle);

        // Reserve one output tile slot
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Acquire exclusive access to DST registers (locks out packer)
        tile_regs_acquire();

        // Copy three input tiles from CBs into DST registers via the unpacker
        // DST[0] = input_a, DST[1] = input_b, DST[2] = input_c
        copy_tile_init(cb_in0);
        copy_tile(cb_in0, 0 /*in_tile_index*/, 0 /*dst_tile_index*/);  // CB[0] -> DST[0]

        copy_tile_init(cb_in1);
        copy_tile(cb_in1, 0 /*in_tile_index*/, 1 /*dst_tile_index*/);  // CB[1] -> DST[1]

        copy_tile_init(cb_in2);
        copy_tile(cb_in2, 0 /*in_tile_index*/, 2 /*dst_tile_index*/);  // CB[2] -> DST[2]

        // TERNARY_SFPU_OP_INIT expands to addcmul_tile_init()
        // TERNARY_SFPU_OP_FUNC expands to addcmul_tile<DataFormat::Float32>(0, 1, 2, 0, scalar_arg)
        //   which computes DST[0] = DST[0] + scalar_arg * DST[1] * DST[2]
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);

        // Release DST registers to packer
        tile_regs_commit();
        tile_regs_wait();

        // Pack DST[0] into the output CB
        pack_tile(0, cb_out);

        // Release DST register lock
        tile_regs_release();

        // Signal writer that one output tile is ready
        cb_push_back(cb_out, num_tiles_per_cycle);
        // Free input CB slots for the reader to reuse
        cb_pop_front(cb_in0, num_tiles_per_cycle);
        cb_pop_front(cb_in1, num_tiles_per_cycle);
        cb_pop_front(cb_in2, num_tiles_per_cycle);
    }
}
```

#### Annotated Compute Kernel Source (Broadcast Variant)

**File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu_bcast.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"

// Helper function that handles broadcast-aware CB synchronization.
// Broadcast tensors are waited/popped outside the inner loop (once per cycle),
// while non-broadcast tensors are waited/popped inside the inner loop (once per tile).
ALWI void process_tile(
    tt::CBIndex cb_in0,
    tt::CBIndex cb_in1,
    tt::CBIndex cb_in2,
    tt::CBIndex cb_out,
    uint32_t freq,            // number of tiles in the broadcast cycle
    uint32_t tile_start,      // starting offset within the cycle
    uint32_t num_tiles_per_cycle,
    uint32_t scalar_arg) {
    using namespace ckernel;

    // Broadcast CBs: wait once before the loop (the single tile is reused)
#if BCAST_A
    cb_wait_front(cb_in0, num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_wait_front(cb_in1, num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_wait_front(cb_in2, num_tiles_per_cycle);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Non-broadcast CBs: wait for each new tile
#if !BCAST_A
        cb_wait_front(cb_in0, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_wait_front(cb_in1, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_wait_front(cb_in2, num_tiles_per_cycle);
#endif

        cb_reserve_back(cb_out, num_tiles_per_cycle);
        tile_regs_acquire();

        // Load all three inputs into DST registers
        copy_tile_init(cb_in0);
        copy_tile(cb_in0, 0, 0);    // DST[0] = input_a

        copy_tile_init(cb_in1);
        copy_tile(cb_in1, 0, 1);    // DST[1] = input_b

        copy_tile_init(cb_in2);
        copy_tile(cb_in2, 0, 2);    // DST[2] = input_c

        // Compute: DST[0] = DST[0] + scalar_arg * DST[1] * DST[2]
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_out);

        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);

        // Pop non-broadcast CBs inside loop
#if !BCAST_A
        cb_pop_front(cb_in0, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_pop_front(cb_in1, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_pop_front(cb_in2, num_tiles_per_cycle);
#endif
    }

    // Pop broadcast CBs after the full cycle completes
#if BCAST_A
    cb_pop_front(cb_in0, num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_pop_front(cb_in1, num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_pop_front(cb_in2, num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);     // broadcast cycle length (e.g., Wt for COL_BCAST)
    uint32_t tile_start = get_arg_val<uint32_t>(2);     // starting position in cycle
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_in0, cb_out);

    // Divide total tiles into complete broadcast cycles plus a remainder
    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(cb_in0, cb_in1, cb_in2, cb_out, tile_freq, tile_start, num_tiles_per_cycle, scalar_arg);
    }

    if (remaining_iterations > 0) {
        process_tile(cb_in0, cb_in1, cb_in2, cb_out, remaining_iterations, tile_start, num_tiles_per_cycle, scalar_arg);
    }
}
```

#### INT32 Compute Kernel (No Broadcast)

**File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addcmul_int_sfpu.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"       // fill_tile_int for scalar tile creation
#include "api/compute/mul_int_sfpu.h"              // integer multiply on SFPU
#include "api/compute/add_int_sfpu.h"              // integer add on SFPU

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_in0, num_tiles_per_cycle);
        cb_wait_front(cb_in1, num_tiles_per_cycle);
        cb_wait_front(cb_in2, num_tiles_per_cycle);

        cb_reserve_back(cb_out, num_tiles_per_cycle);

        tile_regs_acquire();

        // Copy inputs: DST[0]=a, DST[1]=b, DST[2]=c
        copy_tile_init(cb_in0);
        copy_tile(cb_in0, 0, 0);

        copy_tile_init(cb_in1);
        copy_tile(cb_in1, 0, 1);

        copy_tile_init(cb_in2);
        copy_tile(cb_in2, 0, 2);

        // Fill DST[3] with scalar value (broadcast to all elements)
        fill_tile_init();
        fill_tile_int<DataFormat::Int32>(3, scalar_arg);

        // Step 1: DST[3] = scalar * input_b  (value * b)
        mul_int_tile_init<DataFormat::Int32>();
        mul_int_tile<DataFormat::Int32>(3, 1, 3);

        // Step 2: DST[2] = (scalar * input_b) * input_c  (value * b * c)
        mul_int_tile<DataFormat::Int32>(3, 2, 2);

        // Step 3: DST[0] = input_a + (value * b * c)
        add_int_tile_init();
        add_int_tile<DataFormat::Int32>(0, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_out);

        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_in0, num_tiles_per_cycle);
        cb_pop_front(cb_in1, num_tiles_per_cycle);
        cb_pop_front(cb_in2, num_tiles_per_cycle);
    }
}
```

The INT32 path cannot use the SFPU addcmul instruction (which operates on floating point), so it decomposes the operation into discrete integer SFPU operations: fill a tile with the scalar, multiply twice, then add.

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h`
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h`
- **LLK wrapper**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_addcmul.h`
- **Compute API**: `tt_metal/hw/inc/api/compute/eltwise_unary/addcmul.h`

#### Annotated SFPU Kernel Source (Blackhole)

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_addcmul(
    const uint dst_index_in0,  // input_a (the addend, DST tile index)
    const uint dst_index_in1,  // input_b (first multiplicand, DST tile index)
    const uint dst_index_in2,  // input_c (second multiplicand, DST tile index)
    const uint dst_index_out,  // output (result DST tile index, typically same as dst_index_in0)
    const uint value) {        // scalar value, bit-pattern of a float packed into uint32

    // Compile-time check: only floating-point formats supported by SFPU
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Bfp8_b,
        "Unsupported data format for calculate_addcmul(). Only Float32, Float16_b (BFloat16), and Bfp8_b (BFloat8B) "
        "are allowed.");

    // Select load/store modifier based on data format:
    // FP32 uses 32-bit load/store, DEFAULT (FP16_b/Bfp8_b) uses 16-bit
    constexpr InstrModLoadStore mod0 =
        (data_format == DataFormat::Float32) ? InstrModLoadStore::FP32 : InstrModLoadStore::DEFAULT;

    // Each tile in the DEST register file occupies 64 rows (4 faces x 16 rows/face)
    constexpr uint dst_tile_size = 64;

    // Load the scalar "value" into LREG3 (SFPU local register 3).
    // SFPLOADI loads a 16-bit immediate, so we need two instructions for a 32-bit float:
    //   first the lower 16 bits, then the upper 16 bits.
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);  // LREG3[15:0] = lower half
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);     // LREG3[31:16] = upper half

    // Process all 8 sub-rows (ITERATIONS=8 by default).
    // Each iteration processes one 32-element row across all 4 faces of a tile.
    // The compiler unrolls this loop completely.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Step 1: Load input_b row from DEST into LREG1
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, dst_index_in1 * dst_tile_size);

        // Step 2: LREG4 = LREG1 * LREG3 = input_b * value
        // SFPMUL(src_a, src_b, lconst, dst, mod) -- lconst=0 means no additive constant
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);

        // Step 3: Load input_a row from DEST into LREG0
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, dst_index_in0 * dst_tile_size);

        // Step 4: Load input_c row from DEST into LREG2
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_7, dst_index_in2 * dst_tile_size);

        // Step 5: LREG5 = LREG2 * LREG4 + LREG0 = input_c * (value * input_b) + input_a
        // SFPMAD is multiply-add: dst = src_a * src_b + src_c
        // This computes the final addcmul result in a single fused operation
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, 0);

        // Pipeline NOP: required after SFPMAD before the result in LREG5 can be consumed
        TTI_SFPNOP;

        // For non-FP32 accumulation: round the FP32 SFPU result back to FP16A (BFloat16)
        // This is necessary because the SFPU always computes in FP32 internally
        if constexpr (!is_fp32_dest_acc_en) {
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,            // deterministic round-to-nearest-even
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16A,  // FP32 -> BFloat16 conversion
                0,
                p_sfpu::LREG5,                          // source
                p_sfpu::LREG5,                          // destination (in-place)
                InstrModLoadStore::FP16A);
        }

        // Store the result from LREG5 back to the output tile in DEST
        TT_SFPSTORE(p_sfpu::LREG5, mod0, ADDR_MOD_7, dst_index_out * dst_tile_size);

        // Advance the implicit row pointer to the next row within each face
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
```

#### Annotated SFPU Kernel Source (Wormhole B0)

The Wormhole B0 implementation is identical to Blackhole except for the address modifier used in SFPLOAD/SFPSTORE instructions:

```cpp
// Wormhole uses ADDR_MOD_3 instead of Blackhole's ADDR_MOD_7
TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
// ... (same sequence of SFPMUL, SFPLOAD, SFPLOAD, SFPMAD, SFPNOP, SFPSTORE)
TT_SFPSTORE(p_sfpu::LREG5, mod0, ADDR_MOD_3, dst_index_out * dst_tile_size);
```

The address modifier difference reflects different DEST register addressing configurations between the two architectures, but the mathematical computation is identical.

#### SFPU Instructions Used

| Instruction | Description | Usage in ADDCMUL |
|---|---|---|
| `SFPLOADI` | Load 16-bit immediate into SFPU local register | Load scalar `value` into LREG3 (two instructions for 32-bit) |
| `SFPLOAD` | Load a row from DEST register into SFPU local register | Load input_a, input_b, input_c rows from DEST tiles |
| `SFPMUL` | Multiply two SFPU registers | Compute `value * input_b` |
| `SFPMAD` | Fused multiply-add: `dst = src_a * src_b + src_c` | Compute `input_c * (value * input_b) + input_a` |
| `SFPNOP` | Pipeline no-op | Required after SFPMAD for pipeline latency |
| `SFP_STOCH_RND` | Stochastic/deterministic rounding | Round FP32 result to FP16A when not in FP32 accumulation mode |
| `SFPSTORE` | Store SFPU local register row back to DEST | Write final result to output tile in DEST |

#### SFPU Register Usage

| Register | Role | Contents |
|---|---|---|
| `LREG0` | Input A | Row of `input_a` loaded from DEST |
| `LREG1` | Input B | Row of `input_b` loaded from DEST |
| `LREG2` | Input C | Row of `input_c` loaded from DEST |
| `LREG3` | Scalar | The `value` scalar constant (loaded once, reused across all iterations) |
| `LREG4` | Intermediate | `value * input_b` (product of scalar and input_b) |
| `LREG5` | Result | Final `input_c * (value * input_b) + input_a` result |
| `LCONST_0` | Constant | Zero constant (used as additive identity in SFPMUL) |

#### SFPU Execution Flow

1. **Tile Acquisition**: The compute kernel calls `cb_wait_front` on CB0, CB1, CB2 to wait for the reader to produce one tile per input. Then calls `cb_reserve_back` on CB3 (output).

2. **Unpack to DEST**: `copy_tile(cb_in0, 0, 0)` triggers the unpacker to read one tile from each input CB into DEST registers at indices 0, 1, and 2. Each DEST tile occupies 64 rows (4 faces of 16 rows each).

3. **Scalar Loading**: Two `SFPLOADI` instructions load the 32-bit scalar `value` into LREG3. This happens once before the iteration loop and LREG3 is reused for all 8 iterations.

4. **Per-Row SFPU Computation** (8 iterations, one per row-group across all 4 faces):
   - `SFPLOAD LREG1` <- row of input_b from DEST[1]
   - `SFPMUL LREG4 = LREG1 * LREG3` (value * input_b)
   - `SFPLOAD LREG0` <- row of input_a from DEST[0]
   - `SFPLOAD LREG2` <- row of input_c from DEST[2]
   - `SFPMAD LREG5 = LREG2 * LREG4 + LREG0` (input_c * (value * input_b) + input_a)
   - `SFPNOP` (pipeline stall)
   - Optional: `SFP_STOCH_RND` to convert FP32 to FP16A if not in FP32 accumulation mode
   - `SFPSTORE LREG5` -> row of output to DEST[0]
   - `dst_reg++` advances the implicit row counter

5. **Pack from DEST**: `pack_tile(0, cb_out)` packs the output DEST tile into the output CB.

6. **CB Synchronization**: `cb_push_back(cb_out, 1)` signals the writer; `cb_pop_front` on input CBs frees slots for the reader.

#### SFPU Configuration

- **Compile-time defines**:
  - `TERNARY_SFPU_OP_INIT` = `addcmul_tile_init`
  - `TERNARY_SFPU_OP_FUNC` = `addcmul_tile<DataFormat::Float32>` or `addcmul_tile<DataFormat::Float16_b>`
  - `APPROX` template parameter (from compute config) -- not used by addcmul but required by template
  - `DST_ACCUM_MODE` template parameter -- controls `is_fp32_dest_acc_en`

- **Math fidelity**: ADDCMUL uses raw SFPU instructions (SFPMUL, SFPMAD) rather than LLK math fidelity settings. The SFPU always operates at full FP32 precision internally; fidelity only matters for FPU matrix operations.

- **Approximation mode**: The `APPROXIMATION_MODE` template parameter is accepted but not used by the addcmul kernel since it performs exact multiply-add operations (no transcendental approximations).

- **Initialization**: `addcmul_tile_init()` calls `llk_math_eltwise_ternary_sfpu_addcmul_init<APPROX>()` which calls `_llk_math_eltwise_ternary_sfpu_init_<SfpuType::addcmul>()`. This configures the SFPU address modifiers and resets math counters.

- **Face iteration**: The LLK wrapper `llk_math_eltwise_ternary_sfpu_addcmul` calls `_llk_math_eltwise_ternary_sfpu_params_` which handles the per-face iteration in `VectorMode::RC` mode (all 4 faces). It calls `calculate_addcmul` once per face, with each call processing 8 rows (ITERATIONS=8). The `_llk_math_eltwise_ternary_sfpu_start_` sets the DEST write address and stalls until SFPU is ready; `_llk_math_eltwise_ternary_sfpu_done_` clears the address and waits for completion.

#### Hardware Compatibility Notes

- **Blackhole vs Wormhole**: The only difference is the DEST address modifier (`ADDR_MOD_7` on Blackhole, `ADDR_MOD_3` on Wormhole B0). The address modifier controls how the SFPU addresses rows within the DEST register file. Both architectures support the same set of SFPU instructions used by addcmul.

- **Wormhole B0 specifics**: The `_llk_math_eltwise_ternary_sfpu_start_` on Wormhole also calls `math::set_addr_mod_base()` and `_llk_math_eltwise_ternary_sfpu_done_` calls `math::clear_addr_mod_base()`. These additional register management steps are not needed on Blackhole.

- **FP16A rounding**: The `SFP_STOCH_RND` instruction with `SFPSTOCHRND_RND_EVEN` mode performs deterministic round-to-nearest-even when converting the FP32 SFPU result back to BFloat16. This is active whenever `fp32_dest_acc_en` is false (i.e., BFloat16 or Bfp8_b output).

---

## LLK API Call Chain

```
addcmul_tile<DataFormat>()                              [compute API layer]
  -> llk_math_eltwise_ternary_sfpu_addcmul<...>()       [LLK wrapper]
    -> _llk_math_eltwise_ternary_sfpu_params_<...>()     [LLK infrastructure]
      -> _llk_math_eltwise_ternary_sfpu_start_()          [set DEST addr, stall SFPU]
      -> For each face (4 faces in VectorMode::RC):
           sfpu::calculate_addcmul<...>()                   [SFPU kernel - the actual instructions]
           TTI_SETRWC to advance to next face
      -> _llk_math_eltwise_ternary_sfpu_done_()           [clear addr, wait for SFPU]
```

---

## Work Distribution

### Interleaved Mode
Work is split across cores using `split_work_to_cores()`:
- Total output tiles are divided among available cores
- Two core groups may exist (group 1 gets `ceil(tiles/cores)`, group 2 gets `floor(tiles/cores)`)
- Each core processes a contiguous range of tiles starting at `start_tile_id`

### Sharded Mode
When inputs and output are sharded in L1:
- Each core processes exactly the tiles in its shard
- Input CBs are backed directly by the shard buffer (no NoC reads needed)
- Shard shapes can vary per core (edge cores may have smaller shards)
- Output CB is backed by the output shard buffer (no NoC writes needed)
- The `has_sharding` flag controls reader/writer behavior via compile-time defines

### Grid Selection
- If the worker grid is a single rectangle starting at (0,0) and sharding also starts at (0,0), a fast-path (`zero_start_grid`) is used
- Otherwise, a generic core enumeration handles arbitrary core ranges

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: ADDCMUL operation structure, kernel config map, program factory architecture, composite fallback behavior
- `tenstorrent/tt-llk`: `_llk_math_eltwise_ternary_sfpu_params_` face iteration pattern, `SfpuType::addcmul` initialization, VectorMode::RC handling, Wormhole vs Blackhole `_sfpu_start_`/`_sfpu_done_` differences

### Confluence References
Not consulted for this analysis. The SFPU instructions used (SFPLOADI, SFPLOAD, SFPMUL, SFPMAD, SFPNOP, SFP_STOCH_RND, SFPSTORE) are standard and well-documented via DeepWiki and source code.

### Glean References
Not consulted for this analysis. The SFPU behavior was fully determinable from source code and DeepWiki.

---

## Summary

ADDCMUL is a ternary SFPU operation that computes `output = input_a + (value * input_b * input_c)` in a single fused kernel. The key architectural decisions are:

1. **Shared kernel infrastructure**: ADDCMUL shares the `ternary_addc_ops_sfpu.cpp` compute kernel with ADDCDIV, differentiated only by compile-time defines (`TERNARY_SFPU_OP_INIT` / `TERNARY_SFPU_OP_FUNC`).

2. **Two-instruction SFPU math**: The core computation uses `SFPMUL` followed by `SFPMAD` to compute the fused multiply-add in just two arithmetic instructions per row, taking advantage of SFPMAD's built-in accumulation.

3. **Scalar preloading**: The scalar `value` is loaded into LREG3 once before the iteration loop and reused across all 8 row iterations, avoiding redundant immediate loads.

4. **INT32 fallback**: Integer operations cannot use the floating-point SFPMUL/SFPMAD path, so a separate kernel decomposes the operation into discrete `fill_tile_int`, `mul_int_tile`, and `add_int_tile` calls.

5. **Broadcast-aware CB management**: The broadcast variant kernel uses compile-time `#if BCAST_X` guards to control whether each input's CB is waited/popped once per broadcast cycle (outside the loop) or once per tile (inside the loop).

6. **FPU path optimization**: When all three inputs are BFloat16, an FPU path (`is_fpu=true`) may be selected for row broadcast, using the dedicated `ternary_addc_ops_fpu_rowbcast.cpp` kernel which leverages the FPU's native BFloat16 hardware.
