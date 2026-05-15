// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr auto input_cb = tt::CBIndex::c_0;
    constexpr auto partial_prod_cb = tt::CBIndex::c_2;
    constexpr auto final_output_cb = tt::CBIndex::c_3;

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    (void)per_core_block_dim;

    // Caller-init: boot engine for the union of CB triples used by the chain
    // stages below. Stage 1 reads input_cb; stages 2+ read partial_prod_cb as
    // srcb; final stage packs to final_output_cb. One boot at top of MAIN()
    // is sufficient — each chain's reconfig fold emits per-element reconfigs.
    compute_kernel_hw_startup(input_cb, partial_prod_cb, final_output_cb);

    if constexpr (num_tiles == 1) {
        // Single tile: copy input[0] straight to final output.
        compute_kernel_lib::eltwise_chain(
            1,
            compute_kernel_lib::
                CopyTile<input_cb, compute_kernel_lib::Dst::D0, compute_kernel_lib::CopyTilePolicy::WaitAndPop>{},
            compute_kernel_lib::PackTile<
                final_output_cb,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush>{});
    } else {
        // Seed: copy first input tile into the partial-product CB.
        compute_kernel_lib::eltwise_chain(
            1,
            compute_kernel_lib::
                CopyTile<input_cb, compute_kernel_lib::Dst::D0, compute_kernel_lib::CopyTilePolicy::WaitAndPop>{},
            compute_kernel_lib::PackTile<
                partial_prod_cb,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush>{});

        // Middle iters: partial_prod = input[t] * partial_prod (in-place).
        for (uint32_t t = 1; t < num_tiles - 1; ++t) {
            compute_kernel_lib::eltwise_chain(
                1,
                compute_kernel_lib::BinaryFpu<
                    input_cb,
                    partial_prod_cb,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CbIndexMode::FirstTile,
                    compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    partial_prod_cb,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush,
                    compute_kernel_lib::PackTileIndexMode::FirstTile,
                    compute_kernel_lib::PackTileReconfig::Output>{});
        }

        // Final iter: final_output = input[N-1] * partial_prod.
        compute_kernel_lib::eltwise_chain(
            1,
            compute_kernel_lib::BinaryFpu<
                input_cb,
                partial_prod_cb,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                compute_kernel_lib::CbIndexMode::FirstTile,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                final_output_cb,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush,
                compute_kernel_lib::PackTileIndexMode::FirstTile,
                compute_kernel_lib::PackTileReconfig::Output>{});
    }
}
