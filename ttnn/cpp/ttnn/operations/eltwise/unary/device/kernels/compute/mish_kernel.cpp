// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef INP_FLOAT32
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"
#endif

#ifdef INP_FLOAT
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/compute_kernel_api.h"
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t approx_arg = get_arg_val<uint32_t>(1);
    const bool use_approx = (approx_arg != 0u);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);
    init_sfpu(cb_input, cb_output);

#ifdef INP_FLOAT32
    // mish(x) = x * tanh(log1p(exp(x)))
    // FP32 path: use SFPU helpers with binary mul (needs two DEST slots)
    using namespace compute_kernel_lib;
    if (use_approx) {
        auto chain = sfpu_chain(
            Load<cb_input, Dst::D0>{},
            Load<cb_input, Dst::D1>{},
            Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Fast, Dst::D0>{},
            Tanh<Approx::Exact, Dst::D0>{},
            SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
        sfpu_pipeline<
            SfpuBatching::Auto,
            SfpuInputPolicy::WaitAndPopPerTile,
            SfpuOutputPolicy::Bulk,
            SfpuDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
    } else {
        auto chain = sfpu_chain(
            Load<cb_input, Dst::D0>{},
            Load<cb_input, Dst::D1>{},
            Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Exact, Dst::D0>{},
            Tanh<Approx::Exact, Dst::D0>{},
            SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
        sfpu_pipeline<
            SfpuBatching::Auto,
            SfpuInputPolicy::WaitAndPopPerTile,
            SfpuOutputPolicy::Bulk,
            SfpuDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
    }
#endif

#ifdef INP_FLOAT
    // BFloat16 path: use FPU binary_dest_reuse for mul (preserves original precision path)
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_output, 1);
        cb_wait_front(cb_input, 1);
        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, 0);

        if (use_approx) {
            exp_tile_init<true>();
            exp_tile<true>(0);
            log1p_tile_init<true>();
            log1p_tile<true>(0);
        } else {
            exp_tile_init<false>();
            exp_tile<false>(0);
            log1p_tile_init<false>();
            log1p_tile<false>(0);
        }
        tanh_tile_init<false>();
        tanh_tile<false>(0);

        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_input, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_output);
        tile_regs_release();
        cb_pop_front(cb_input, 1);
        cb_push_back(cb_output, 1);
    }
#endif
}
