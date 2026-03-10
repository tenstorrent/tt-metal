# ADDCMUL Implementation Analysis

## Overview
The ADDCMUL operation computes an element-wise add-with-constant-multiply across three input tensors and a scalar:

```
output = input_a + (value * input_b * input_c)
```

where `input_a`, `input_b`, and `input_c` are tensors and `value` is a scalar constant. This is the PyTorch `torch.addcmul` equivalent.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

The operation uses the shared ternary program factory (`TernaryProgramFactory`) which also serves WHERE, LERP, and ADDCDIV operations. The factory selects ADDCMUL-specific compute kernels and defines (`addcmul_tile_init` / `addcmul_tile`) based on the `TernaryOpType::ADDCMUL` enum.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `output.physical_volume() / tile_hw` |
| **Loop structure** | Linear iteration over tiles assigned to each core; broadcast variants use nested loops with frequency/counter for CB reuse |

## Tensor Format and Layout

### Input Tensors

ADDCMUL is a TTT-variant-only operation (all three inputs are tensors). The scalar `value` is passed as a runtime argument to the compute kernel.

| Property | Input A (Predicate/CB0) | Input B (True/CB1) | Input C (False/CB2) |
|----------|------------------------|---------------------|---------------------|
| **Logical shape** | Arbitrary (up to rank 6+) | Arbitrary (broadcastable to output) | Arbitrary (broadcastable to output) |
| **Dimension convention** | [..., D, N, C, H, W] | [..., D, N, C, H, W] | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (L1) | INTERLEAVED or SHARDED (L1) | INTERLEAVED or SHARDED (L1) |
| **Buffer type** | DRAM or L1 | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32 | BFLOAT16, FLOAT32, INT32 | BFLOAT16, FLOAT32, INT32 |

### Output Tensor

| Property | Output (CB3) |
|----------|-------------|
| **Logical shape** | Broadcast-expanded shape of all three inputs |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (L1) |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input dtype |

### Layout Transformations
- No tilize/untilize conversions; all tensors must be pre-tiled.
- Broadcasting is handled by the reader kernel: input tensors with dimension-1 extents are replicated to match the output shape.
- For INT32 data type, a specialized compute kernel (`ternary_addcmul_int_sfpu.cpp` / `ternary_addcmul_int_sfpu_bcast.cpp`) is selected instead of the generic addc_ops kernel.

## Data Flow Pattern

1. **Reader kernel** reads tiles from three input tensors (DRAM or L1 shards) into CB0, CB1, CB2, one tile at a time per input.
2. **Compute kernel** waits for one tile in each of CB0, CB1, CB2; unpacks all three into DEST registers (indices 0, 1, 2); executes `addcmul_tile` SFPU operation which computes `DEST[0] + (value * DEST[1] * DEST[2])` and writes the result to DEST[0]; packs DEST[0] into CB3.
3. **Writer kernel** waits for one tile in CB3; writes it to DRAM or L1 output buffer.

For broadcast variants, the broadcast CB is loaded once outside the inner loop and popped after the loop completes (column broadcast, scalar broadcast). Non-broadcast CBs are loaded/popped per tile.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_in0 | Input A (predicate) staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Block |
| c_1 | cb_in1 | Input B (true value) staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Block |
| c_2 | cb_in2 | Input C (false value) staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Block |
| c_3 | cb_out | Output staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Block |
| c_4 | cb_bcast_a | Row-broadcast scratch for input A | 2 tiles | 1 tile | Double | Reader | Compute | Block (ROW_BCAST only) |
| c_5 | cb_bcast_b | Row-broadcast scratch for input B | 2 tiles | 1 tile | Double | Reader | Compute | Block (ROW_BCAST only) |
| c_6 | cb_bcast_c | Row-broadcast scratch for input C | 2 tiles | 1 tile | Double | Reader | Compute | Block (ROW_BCAST only) |

Note: CBs c_4, c_5, c_6 are only created when `broadcast_type == ROW_BCAST` and variant is TTT.

## Pipeline Pattern Summary

- **Interleaved mode**: All CBs have capacity = 2 tiles with block size = 1 tile, enabling double-buffering. The reader can write the next tile while compute processes the current one.
- **Sharded mode**: CB capacity equals the shard volume. Sharded input CBs are pre-loaded (reserve_back + push_back) before the compute loop. This is single-buffered since all data is resident in L1.

## Index Calculations

The reader kernel uses a multi-dimensional index decomposition to map a linear tile ID to (nD, D, N, C, Ht, Wt) coordinates. Each input tensor has independent strides computed by the host:

- `nD_stride = Ht * Wt * C * N * D * (ND > 1)`
- `d_stride = Ht * Wt * C * N * (D > 1)`
- `n_stride = Ht * Wt * C * (N > 1)`
- `c_stride = Ht * Wt * (C > 1)`

A stride of 0 indicates the tensor has extent 1 in that dimension, implementing broadcasting. The reader computes per-tensor tile offsets using these strides and reads from the correct DRAM bank via `TensorAccessor`.

For sharded mode, tile offsets are computed on the host from shard shapes and core positions, and width-sharding is supported via `dst_shard_width` to handle partial rows.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads using `noc_async_read_page()` through the TensorAccessor API. The 6-level nested loop (nD, D, N, C, Ht, Wt) traverses tiles in row-major order within each dimension. All three input tensors are read in lockstep per tile iteration.
- **Sharded**: Input data is already in L1. The reader simply issues `cb_reserve_back` / `cb_push_back` to make the shard available to compute.

### Write Pattern
- **Interleaved**: Sequential tile writes using `noc_async_write_page()` through the TensorAccessor API. Same multi-dimensional traversal as the reader.
- **Sharded**: Output data is written directly to the sharded L1 buffer. No explicit writes needed (DST_SHARDED define skips the write loop).

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D or 2D (depends on worker_grid from operation attributes) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (zero-start) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles (interleaved); shard volume tiles (sharded) |
| **Load balancing** | `split_work_to_cores()` divides total output tiles across cores; remainder tiles go to core_group_2 which gets one fewer tile; sharded mode uses shard geometry directly |

The factory uses `grid_to_cores` for ordered core enumeration and supports both zero-start grids (rectangular from (0,0)) and arbitrary core range sets. Cores outside both core groups receive zeroed-out runtime arguments and produce no output.

## Arguments

### Compile-Time Arguments

#### Reader Kernel (TTT variant)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_src0 | uint32_t | Circular buffer ID for input A (predicate) |
| 1 | cb_id_src1 | uint32_t | Circular buffer ID for input B (true) |
| 2 | cb_id_src2 | uint32_t | Circular buffer ID for input C (false) |
| 3+ | TensorAccessorArgs (src0) | varying | Compile-time args for predicate tensor accessor |
| N+ | TensorAccessorArgs (src1) | varying | Compile-time args for true tensor accessor |
| M+ | TensorAccessorArgs (src2) | varying | Compile-time args for false tensor accessor |
| last | has_sharding | uint32_t | 1 if any tensor is sharded, 0 otherwise |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles processed per read-compute-write cycle |
| 1 | scalar_is_true_value | uint32_t | 0 for TTT variant (not used by addcmul) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Circular buffer ID for output |
| 1+ | TensorAccessorArgs (dst) | varying | Compile-time args for output tensor accessor |
| last | has_sharding | uint32_t | 1 if output is sharded, 0 otherwise |

### Runtime Arguments

#### Reader Kernel (TTT variant, 27 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | DRAM address of input A buffer |
| 1 | src1_addr | uint32_t | DRAM address of input B buffer |
| 2 | src2_addr | uint32_t | DRAM address of input C buffer |
| 3 | num_tiles | uint32_t | Total tiles to process on this core |
| 4 | start_id | uint32_t | Starting tile ID for this core |
| 5 | nD_stride | uint32_t | Predicate ND-dimension stride |
| 6 | d_stride | uint32_t | Predicate D-dimension stride |
| 7 | n_stride | uint32_t | Predicate N-dimension stride |
| 8 | c_stride | uint32_t | Predicate C-dimension stride |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output collapsed ND dimensions |
| 15 | true_nD_stride | uint32_t | Input B ND-dimension stride |
| 16 | true_d_stride | uint32_t | Input B D-dimension stride |
| 17 | true_n_stride | uint32_t | Input B N-dimension stride |
| 18 | true_c_stride | uint32_t | Input B C-dimension stride |
| 19 | true_num_tiles | uint32_t | Input B total tiles (for sharding) |
| 20 | false_nD_stride | uint32_t | Input C ND-dimension stride |
| 21 | false_d_stride | uint32_t | Input C D-dimension stride |
| 22 | false_n_stride | uint32_t | Input C N-dimension stride |
| 23 | false_c_stride | uint32_t | Input C C-dimension stride |
| 24 | false_num_tiles | uint32_t | Input C total tiles (for sharding) |
| 25 | dst_shard_width | uint32_t | Output shard width in tiles (0 if not sharded) |
| 26 | src_num_tiles | uint32_t | Predicate total tiles (for sharding) |

#### Compute Kernel (4 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | tile_freq | uint32_t | Broadcast frequency (0 for NONE/OUTER/ROW) |
| 2 | tile_start | uint32_t | Starting position within broadcast cycle |
| 3 | scalar_arg | uint32_t | Packed scalar value (bit-cast float or int) |

#### Writer Kernel (11 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer DRAM address |
| 1 | num_tiles | uint32_t | Total tiles to write on this core |
| 2 | start_id | uint32_t | Starting output tile ID |
| 3 | dst_shard_width | uint32_t | Output shard width in tiles |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Output collapsed ND dimensions |
| 10 | padding | uint32_t | Unused (0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader (ternary_reader_nosubtilebcast_ttt.cpp) | RISCV_0 | NOC0 | DRAM/L1 input buffers | CB0, CB1, CB2 | Read 3 input tiles per iteration via TensorAccessor |
| compute (ternary_addc_ops_sfpu.cpp) | RISCV_2 (Math) | N/A | CB0, CB1, CB2 | CB3 | Unpack 3 tiles to DEST[0,1,2]; SFPU addcmul; pack to CB3 |
| writer (ternary_writer_nobcast.cpp) | RISCV_1 | NOC1 | CB3 | DRAM/L1 output buffer | Write 1 output tile per iteration via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_reader_nosubtilebcast_ttt.cpp`
- **Key Logic**: Uses a 6-level nested loop (nD, D, N, C, Ht, Wt) to traverse the multi-dimensional output tile space. Maintains three independent tile offsets (one per input) using per-tensor strides to support broadcasting. Sharded inputs bypass DRAM reads and instead issue `cb_reserve_back`/`cb_push_back` to expose the L1 shard to the compute kernel. Width-sharding support adjusts the innermost loop bounds via `dst_shard_width`.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_writer_nobcast.cpp`
- **Key Logic**: Same 6-level nested loop structure as the reader. Waits for one tile in the output CB, writes it to DRAM via `noc_async_write_page`, then pops. When `DST_SHARDED` is defined, the entire write loop is compiled out since the output is already in the correct L1 location.

### Compute Kernel

#### Compute Kernel File (No-broadcast variant)
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu.cpp`

#### Annotated Compute Kernel Source

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
#include "api/compute/eltwise_unary/addcmul.h"      // provides addcmul_tile and addcmul_tile_init
#include "api/compute/eltwise_unary/addcdiv.h"       // provides addcdiv_tile (shared kernel with ADDCDIV)

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);   // runtime arg 0: total tiles to process
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);   // runtime arg 3: packed scalar value (bit-cast float/int)
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // compile arg 0: always 1

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a (predicate): the addend tensor
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b: first multiplicand tensor
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c: second multiplicand tensor
    constexpr auto cb_out = tt::CBIndex::c_3;   // output buffer

    unary_op_init_common(cb_in0, cb_out);  // initialize unpacker and packer for cb_in0 -> cb_out path

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_in0, num_tiles_per_cycle);  // block until reader has produced 1 tile in cb_in0
        cb_wait_front(cb_in1, num_tiles_per_cycle);  // block until reader has produced 1 tile in cb_in1
        cb_wait_front(cb_in2, num_tiles_per_cycle);  // block until reader has produced 1 tile in cb_in2

        cb_reserve_back(cb_out, num_tiles_per_cycle); // reserve space for 1 output tile

        tile_regs_acquire();  // acquire exclusive access to DEST registers for unpack+math

        copy_tile_init(cb_in0);                       // configure unpacker for cb_in0 data format
        copy_tile(cb_in0, 0 /*in_tile_index*/, 0 /*dst_tile_index*/);  // unpack cb_in0[0] -> DEST[0]

        copy_tile_init(cb_in1);                       // reconfigure unpacker for cb_in1 data format
        copy_tile(cb_in1, 0 /*in_tile_index*/, 1 /*dst_tile_index*/);  // unpack cb_in1[0] -> DEST[1]

        copy_tile_init(cb_in2);                       // reconfigure unpacker for cb_in2 data format
        copy_tile(cb_in2, 0 /*in_tile_index*/, 2 /*dst_tile_index*/);  // unpack cb_in2[0] -> DEST[2]

        TERNARY_SFPU_OP_INIT();  // expands to addcmul_tile_init(); configures SFPU for addcmul
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);
        // expands to addcmul_tile<DataFormat::Float32>(0, 1, 2, 0, scalar_arg)
        // or addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, scalar_arg)
        // computes: DEST[0] = DEST[0] + (scalar_arg * DEST[1] * DEST[2])
        // result overwrites DEST[0]

        tile_regs_commit();   // transfer DEST ownership from math to packer
        tile_regs_wait();     // wait for packer to be ready

        pack_tile(0, cb_out); // pack DEST[0] into cb_out output buffer

        tile_regs_release();  // release DEST registers for next iteration

        cb_push_back(cb_out, num_tiles_per_cycle);    // signal writer that 1 output tile is ready
        cb_pop_front(cb_in0, num_tiles_per_cycle);    // free consumed input_a tile
        cb_pop_front(cb_in1, num_tiles_per_cycle);    // free consumed input_b tile
        cb_pop_front(cb_in2, num_tiles_per_cycle);    // free consumed input_c tile
    }
}
```

#### Compute Kernel File (Broadcast variant)
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu_bcast.cpp`

This variant handles column-broadcast and scalar-broadcast cases. It uses conditional compilation (`BCAST_A`, `BCAST_B`, `BCAST_C`) to determine which CBs are loaded once (outside the loop) versus per-tile. The `tile_freq` and `tile_start` runtime args control the broadcast cycling pattern.

### SFPU Kernel Implementation

#### SFPU Kernel File
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h`

(Identical implementation exists for Blackhole at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h`, with only `ADDR_MOD_7` replacing `ADDR_MOD_3`.)

#### Annotated SFPU Kernel Source

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
    const uint dst_index_in0,  // input_a: the addend (DEST tile index 0)
    const uint dst_index_in1,  // input_b: first multiplicand (DEST tile index 1)
    const uint dst_index_in2,  // input_c: second multiplicand (DEST tile index 2)
    const uint dst_index_out,  // output destination (DEST tile index 0, overwrites input_a)
    const uint value) {        // scalar multiplier, packed as uint32_t (bit-cast float)

    // Static assert: only Float32, Float16_b, and Bfp8_b data formats are supported
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Bfp8_b,
        "Unsupported data format for calculate_addcmul(). Only Float32, Float16_b (BFloat16), and Bfp8_b (BFloat8B) "
        "are allowed.");

    // Select load/store modifier: FP32 mode for Float32 data, DEFAULT (FP16) otherwise
    constexpr InstrModLoadStore mod0 =
        (data_format == DataFormat::Float32) ? InstrModLoadStore::FP32 : InstrModLoadStore::DEFAULT;

    // Each tile in DEST occupies 64 rows (4 faces * 16 rows per face)
    constexpr uint dst_tile_size = 64;

    // Load the scalar value into LREG3 as a 32-bit float:
    // First load the lower 16 bits, then the upper 16 bits
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);  // LREG3[15:0] = value[15:0]
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);     // LREG3[31:16] = value[31:16]

    // Process 8 iterations, one per row-group within a face
    // The outer _llk_math_eltwise_ternary_sfpu_params_ framework calls this function
    // 4 times (once per face in VectorMode::RC), advancing dst_reg between calls
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Step 1: Load input_b row from DEST into LREG1
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        // LREG1 = DEST[dst_index_in1][current_row], 32 lanes loaded

        // Step 2: Multiply input_b by scalar value: LREG4 = LREG1 * LREG3
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
        // LREG4 = value * input_b (element-wise across 32 lanes)

        // Step 3: Load input_a row from DEST into LREG0
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        // LREG0 = DEST[dst_index_in0][current_row]

        // Step 4: Load input_c row from DEST into LREG2
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_3, dst_index_in2 * dst_tile_size);
        // LREG2 = DEST[dst_index_in2][current_row]

        // Step 5: Multiply-add: LREG5 = LREG2 * LREG4 + LREG0
        //   = input_c * (value * input_b) + input_a
        //   = input_a + value * input_b * input_c
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, 0);

        // Pipeline NOP: required on Wormhole to avoid read-after-write hazard after SFPMAD
        TTI_SFPNOP;

        // Step 6: Round result for non-FP32 accumulation modes
        if constexpr (!is_fp32_dest_acc_en) {
            // Round FP32 result in LREG5 to FP16A (bfloat16) precision using even rounding
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,           // round-to-nearest-even mode
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16A, // convert FP32 -> FP16A (bfloat16)
                0,
                p_sfpu::LREG5,                         // source register
                p_sfpu::LREG5,                         // destination register (in-place)
                InstrModLoadStore::FP16A);             // store format
        }

        // Step 7: Store result from LREG5 back to DEST at the output tile index
        TT_SFPSTORE(p_sfpu::LREG5, mod0, ADDR_MOD_3, dst_index_out * dst_tile_size);
        // DEST[dst_index_out][current_row] = LREG5

        // Advance to next row within the current face
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOADI` | Loads a 16-bit immediate value into the lower or upper half of all 32 lanes of an LREG. Used here to construct the 32-bit scalar value in LREG3. |
| `SFPLOAD` | Loads 32 elements from a row of the DEST register file into an LREG. Supports FP32 or FP16 data format conversion. |
| `SFPMUL` | Performs lane-wise FP32 multiplication: `VD = VA * VB`. Used to multiply input_b by the scalar value. |
| `SFPMAD` | Performs lane-wise fused multiply-add: `VD = VA * VB + VC`. Used to compute `input_c * (value * input_b) + input_a`. |
| `SFPNOP` | No-operation. Inserted after SFPMAD on Wormhole to satisfy the read-after-write pipeline hazard (1-cycle latency). |
| `SFP_STOCH_RND` | Reduces mantissa precision from FP32 to FP16A (bfloat16) using round-to-nearest-even. Only executed when DEST accumulation is not in FP32 mode. |
| `SFPSTORE` | Writes 32 elements from an LREG back to a row of the DEST register file. |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0` | Holds input_a row loaded from DEST[0] |
| `LREG1` | Holds input_b row loaded from DEST[1] |
| `LREG2` | Holds input_c row loaded from DEST[2] |
| `LREG3` | Holds the scalar `value` (loaded once before the loop, persistent across iterations) |
| `LREG4` | Intermediate: `value * input_b` |
| `LREG5` | Final result: `input_a + value * input_b * input_c` |
| `LCONST_0` | Zero constant used as the addend in SFPMUL (which requires 3 operands; the add term is zeroed out) |
| `DEST[0]` | Tile slot for input_a / output |
| `DEST[1]` | Tile slot for input_b |
| `DEST[2]` | Tile slot for input_c |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` to gain exclusive DEST access, then uses `copy_tile` to unpack three input tiles from CB0, CB1, CB2 into DEST[0], DEST[1], DEST[2].
2. **SFPU initialization**: `addcmul_tile_init()` calls `_llk_math_eltwise_ternary_sfpu_init_<SfpuType::addcmul>()` which configures the SFPU pipeline.
3. **LLK dispatch**: `addcmul_tile<DataFormat>` calls `llk_math_eltwise_ternary_sfpu_addcmul` which delegates to `_llk_math_eltwise_ternary_sfpu_params_`. This framework function:
   - Asserts all DEST indices are in range
   - Calls `_llk_math_eltwise_ternary_sfpu_start_` to set DEST write address and stall until SFPU ready
   - In `VectorMode::RC` (default), calls `calculate_addcmul` 4 times (once per tile face), issuing `TTI_SETRWC` between calls to advance the face pointer
   - Calls `_llk_math_eltwise_ternary_sfpu_done_` to finalize
4. **Per-face processing** (`calculate_addcmul`): For each of 8 iterations within a face:
   - Loads scalar value into LREG3 (once before loop)
   - Loads input_b row from DEST[1] into LREG1
   - Multiplies LREG1 * LREG3 -> LREG4 (value * input_b)
   - Loads input_a from DEST[0] into LREG0
   - Loads input_c from DEST[2] into LREG2
   - Fused multiply-add: LREG2 * LREG4 + LREG0 -> LREG5
   - Optional rounding from FP32 to BF16
   - Stores LREG5 back to DEST[0] (output)
   - Advances `dst_reg++` to next row
5. **Pack and output**: After SFPU completes, `tile_regs_commit()` transfers DEST ownership to the packer. `pack_tile(0, cb_out)` packs DEST[0] into the output CB.

#### SFPU Configuration

- **APPROX template parameter**: Controls approximation mode (propagated from LLK layer; typically false for addcmul which uses exact multiply-add).
- **DST_ACCUM_MODE**: Determines whether DEST accumulation is in FP32 or default (FP16) mode. When FP32 (`fp32_dest_acc_en = true`), the `SFP_STOCH_RND` rounding step is skipped.
- **UnpackToDestMode**: Set per-CB based on input dtype. When dtype is FLOAT32, uses `UnpackToDestFp32` to unpack to full-precision DEST registers.
- **Compile defines**: `TERNARY_SFPU_OP_INIT` -> `addcmul_tile_init`, `TERNARY_SFPU_OP_FUNC` -> `addcmul_tile<DataFormat::Float32>` or `addcmul_tile<DataFormat::Float16_b>`.

#### Hardware Compatibility Notes

- **Wormhole B0 vs Blackhole**: The SFPU kernel is identical between architectures except for the ADDR_MOD field used in SFPLOAD/SFPSTORE: Wormhole uses `ADDR_MOD_3` (3-bit encoding), Blackhole uses `ADDR_MOD_7` (wider encoding). This difference is transparent to the algorithm.
- **SFPNOP requirement**: The `TTI_SFPNOP` after `SFPMAD` is required on Wormhole to avoid read-after-write pipeline hazards. On Blackhole, automatic instruction scheduling may eliminate this need, but the NOP is kept for compatibility.
- **FPU path**: When all three inputs share the same non-FP32/non-INT32 dtype (e.g., all BFLOAT16), the program factory sets `is_fpu = true` and uses FPU-based compute kernels (`ternary_addc_ops_fpu.cpp`, `ternary_addc_ops_fpu_rowbcast.cpp`) instead of SFPU kernels. The FPU path uses the matrix unit for higher throughput. The SFPU path is used when dtypes are FP32 or INT32.
- **INT32 override**: When output dtype is INT32, the compute kernel is overridden to `ternary_addcmul_int_sfpu.cpp` / `ternary_addcmul_int_sfpu_bcast.cpp` which handles integer arithmetic.

## Implementation Notes

1. **Shared kernel architecture**: The ADDCMUL operation reuses the same compute kernel files as ADDCDIV. The `TERNARY_SFPU_OP_INIT` and `TERNARY_SFPU_OP_FUNC` preprocessor defines select the specific operation at compile time.

2. **Scalar packing**: The scalar `value` is packed into a `uint32_t` by bit-casting the float (or casting int to float then bit-casting). This packed value is passed as runtime arg index 3 to the compute kernel and then forwarded directly to the SFPU as an immediate operand for SFPLOADI.

3. **Broadcast support**: Five broadcast types are supported for TTT variant: NONE, OUTER_BCAST, ROW_BCAST, COL_BCAST, SCALAR_BCAST. The broadcast variant uses `ternary_addc_ops_sfpu_bcast.cpp` which has conditional compilation for BCAST_A/B/C defines and a frequency-based iteration pattern. For ROW_BCAST with all-BF16 inputs, an LLK-based row broadcast path is available.

4. **Operation count per tile**: The SFPU kernel performs 2 multiplications and 1 addition per element (value * input_b, then * input_c, then + input_a), using the SFPMUL + SFPMAD instruction pair for efficient fusion of the second multiply with the addition.

5. **Register pressure**: The kernel uses 6 LREGs (LREG0-LREG5) out of the available 8 (LREG0-LREG7), leaving 2 LREGs unused. LREG3 is loaded once with the scalar and reused across all iterations and faces.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the ternary eltwise operation program factory work? What kernels does it use?"
   **Reason**: Needed to understand the overall architecture of the ternary program factory and how ADDCMUL fits within it.
   **Key Findings**: The factory uses a `TernaryKernelConfig` lookup table keyed by (op_type, variant, broadcast_type) to select reader, compute, and writer kernels. ADDCMUL uses `ComputeNoBcastAddcOp` and `ComputeBcastAddcOp` compute kernels.

2. **Query**: "How do circular buffers work in eltwise operations? What is the typical CB setup?"
   **Reason**: Needed to understand the standard CB configuration for ternary SFPU operations.
   **Key Findings**: Ternary operations use 3 input CBs and 1 output CB. Double-buffering (2 tiles) is standard for interleaved mode. The DEST register flow is: acquire -> copy_tile (unpack) -> SFPU op -> commit -> wait -> pack -> release.

3. **Query**: "What does _llk_math_eltwise_ternary_sfpu_params_ do?"
   **Reason**: Needed to understand the dispatch framework that calls `calculate_addcmul` per tile face.
   **Key Findings**: The function validates DEST indices, initializes SFPU, then calls the SFPU function 4 times (VectorMode::RC) with TTI_SETRWC face-advancing instructions between calls. Each call processes one 16x32 face of the 32x32 tile.

4. **Query**: "SFPU instructions: SFPLOADI, SFPLOAD, SFPMUL, SFPMAD, SFPSTORE, SFPNOP, SFP_STOCH_RND"
   **Reason**: Needed precise semantics of each SFPU instruction used in calculate_addcmul.
   **Key Findings**: SFPLOADI loads 16-bit immediates into LREGs. SFPLOAD/SFPSTORE move data between DEST and LREGs. SFPMUL does lane-wise FP32 multiply. SFPMAD does fused multiply-add. SFPNOP avoids pipeline hazards on Wormhole. SFP_STOCH_RND reduces precision from FP32 to FP16A.

5. **Query**: "What is dst_reg++ in SFPI?"
   **Reason**: Needed to understand row advancement within tile face processing.
   **Key Findings**: `dst_reg++` calls `__builtin_rvtt_ttincrwc` to advance the DEST row pointer by `SFP_DESTREG_STRIDE` (2). This advances to the next pair of rows within a face during the 8-iteration inner loop.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.cpp`
   **Reason**: Needed to understand kernel file path mapping and compute defines selection.
   **Key Information**: `get_kernel_file_path()` maps `KernelName::ComputeNoBcastAddcOp` to `ternary_addc_ops_sfpu.cpp`. `get_compute_defines()` maps `TernaryOpType::ADDCMUL` to `addcmul_tile_init` / `addcmul_tile<DataFormat>`.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/addcmul.h`
   **Reason**: Needed the public API layer that bridges the compute kernel to the LLK SFPU implementation.
   **Key Information**: `addcmul_tile<DataFormat>` calls `llk_math_eltwise_ternary_sfpu_addcmul` with APPROX and DST_ACCUM_MODE template parameters. `addcmul_tile_init()` calls `llk_math_eltwise_ternary_sfpu_addcmul_init`.

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_ternary_sfpu_params.h`
   **Reason**: Needed to understand the face-dispatching framework for ternary SFPU operations.
   **Key Information**: VectorMode::RC processes all 4 faces, calling the SFPU function 4 times with TTI_SETRWC instructions advancing between faces.
