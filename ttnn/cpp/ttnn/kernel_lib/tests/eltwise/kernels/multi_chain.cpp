// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Multi-element chain validation kernels — selected via define CHAIN_VARIANT.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_trig.hpp"

#ifndef CHAIN_VARIANT
#define CHAIN_VARIANT 0
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

#if CHAIN_VARIANT == 0
    // 14.2: Copy + Sigmoid + Tanh + Pack (two SFPU ops in series)
    using Chain = EltwiseChain<
        CopyTile<cb_a, Dst::D0, CopyTilePolicy::WaitAndPop>,
        Sigmoid<Dst::D0>,
        Tanh<Dst::D0>,
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Chain>();
    eltwise_chain(
        num_tiles,
        CopyTile<cb_a, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        Sigmoid<Dst::D0>{},
        Tanh<Dst::D0>{},
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#elif CHAIN_VARIANT == 1
    // 14.3: 2-input BinaryFpu Add + Pack (no extra CopyTile — BinaryFpu owns lifecycle)
    using BinElt = BinaryFpu<
        cb_a,
        cb_b,
        BinaryFpuOp::Add,
        BroadcastDim::None,
        BinaryFpuOutputPolicy::PerTile,
        BinaryDataFormatReconfig::None,
        CopyTilePolicy::WaitAndPop,
        CopyTilePolicy::WaitAndPop,
        CbIndexMode::FirstTile,
        CbIndexMode::FirstTile,
        Dst::D0>;
    using Chain = EltwiseChain<BinElt, PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Chain>();
    eltwise_chain(num_tiles, BinElt{}, PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#elif CHAIN_VARIANT == 2
    // 14.4: 2-input BinaryFpu Add + post-SFPU Sqrt + Pack — sqrt(a+b)
    using BinElt = BinaryFpu<
        cb_a,
        cb_b,
        BinaryFpuOp::Add,
        BroadcastDim::None,
        BinaryFpuOutputPolicy::PerTile,
        BinaryDataFormatReconfig::None,
        CopyTilePolicy::WaitAndPop,
        CopyTilePolicy::WaitAndPop,
        CbIndexMode::FirstTile,
        CbIndexMode::FirstTile,
        Dst::D0>;
    using Chain = EltwiseChain<
        BinElt,
        Sqrt<Approx::Exact, Dst::D0>,
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Chain>();
    eltwise_chain(
        num_tiles,
        BinElt{},
        Sqrt<Approx::Exact, Dst::D0>{},
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#elif CHAIN_VARIANT == 3
    // 14.5: Copy + Exp + Sqrt + Pack (chain-length 4 SFPU back-to-back)
    using Chain = EltwiseChain<
        CopyTile<cb_a, Dst::D0, CopyTilePolicy::WaitAndPop>,
        Exp<>,
        Sqrt<Approx::Exact, Dst::D0>,
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>>;
    eltwise_pipeline_init<Chain>();
    eltwise_chain(
        num_tiles,
        CopyTile<cb_a, Dst::D0, CopyTilePolicy::WaitAndPop>{},
        Exp<>{},
        Sqrt<Approx::Exact, Dst::D0>{},
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
#endif
}
