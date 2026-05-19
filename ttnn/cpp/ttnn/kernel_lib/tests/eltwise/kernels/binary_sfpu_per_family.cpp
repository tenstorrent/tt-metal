// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Per-family coverage test for SFPU binary op-structs added to
// eltwise_binary_sfpu.hpp. The struct is selected at compile time via
// SFPU_FAMILY define (matching pytest parameterization):
//
//   FAMILY_ADD_FLOAT     → AddBinary (existing) — bf16 path
//   FAMILY_BITWISE_AND   → BitwiseAndBinary<DataFormat::UInt16> — uint16 path
//   FAMILY_ADD_INT32     → AddIntBinary<DataFormat::Int32> — int32 path
//   FAMILY_MAX_UINT32    → BinaryMaxUint32 — uint32 path
//   FAMILY_LT_FLOAT      → LtBinary — float compare path (0/1 mask output)

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 2
#endif

#ifndef SFPU_FAMILY
#define SFPU_FAMILY FAMILY_ADD_FLOAT
#endif

#define FAMILY_ADD_FLOAT 0
#define FAMILY_BITWISE_AND 1
#define FAMILY_ADD_INT32 2
#define FAMILY_MAX_UINT32 3
#define FAMILY_LT_FLOAT 4

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using ATile =
        CopyTile<cb_a, Dst::D0, CopyTilePolicy::WaitAndPopPerBlock, CbIndexMode::BlockIter, CopyTileReconfig::Input>;
    using BTile =
        CopyTile<cb_b, Dst::D1, CopyTilePolicy::WaitAndPopPerBlock, CbIndexMode::BlockIter, CopyTileReconfig::Input>;
    using PackElt = PackTile<
        cb_out,
        Dst::D0,
        PackTilePolicy::PerBlockReserveAndPush,
        PackTileIndexMode::BlockIter,
        PackTileReconfig::None>;

#if SFPU_FAMILY == FAMILY_ADD_FLOAT
    using SfpuBin = AddBinary<Dst::D0, Dst::D1, Dst::D0>;
#elif SFPU_FAMILY == FAMILY_BITWISE_AND
    using SfpuBin = BitwiseAndBinary<DataFormat::UInt16, Dst::D0, Dst::D1, Dst::D0>;
#elif SFPU_FAMILY == FAMILY_ADD_INT32
    using SfpuBin = AddIntBinary<DataFormat::Int32, Dst::D0, Dst::D1, Dst::D0>;
#elif SFPU_FAMILY == FAMILY_MAX_UINT32
    using SfpuBin = BinaryMaxUint32<Dst::D0, Dst::D1, Dst::D0>;
#elif SFPU_FAMILY == FAMILY_LT_FLOAT
    using SfpuBin = LtBinary<Dst::D0, Dst::D1, Dst::D0>;
#else
#error "Unknown SFPU_FAMILY"
#endif

    eltwise_chain<BLOCK_SIZE>(num_tiles, ATile{}, BTile{}, SfpuBin{}, PackElt{});
}
