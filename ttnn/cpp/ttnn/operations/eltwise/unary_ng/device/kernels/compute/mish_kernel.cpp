// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef INP_FLOAT32
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"
#endif

#ifdef INP_FLOAT
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"
#endif

void kernel_main() {
    const uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const bool use_approx = (get_arg_val<uint32_t>(1) != 0u);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);
    init_sfpu(cb_input, cb_output);

#ifdef INP_FLOAT32
    using namespace compute_kernel_lib;
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        if (use_approx) {
            auto chain = sfpu_chain(
                Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
                Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
                Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
                Log1p<Approx::Fast, Dst::D0>{},
                Tanh<Approx::Exact, Dst::D0>{},
                SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
            sfpu_pipeline<
                SfpuBatching::Auto,
                SfpuInputPolicy::WaitAndPopPerTile,
                SfpuOutputPolicy::Bulk,
                SfpuDataFormatReconfig::NONE>(chain, cb_output, per_core_block_dim);
        } else {
            auto chain = sfpu_chain(
                Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
                Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
                Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
                Log1p<Approx::Exact, Dst::D0>{},
                Tanh<Approx::Exact, Dst::D0>{},
                SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
            sfpu_pipeline<
                SfpuBatching::Auto,
                SfpuInputPolicy::WaitAndPopPerTile,
                SfpuOutputPolicy::Bulk,
                SfpuDataFormatReconfig::NONE>(chain, cb_output, per_core_block_dim);
        }
    }
#endif

#ifdef INP_FLOAT
    using namespace compute_kernel_lib;
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        if (use_approx) {
            auto chain = sfpu_chain(
                Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
                Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
                Log1p<Approx::Fast, Dst::D0>{},
                Tanh<Approx::Exact, Dst::D0>{},
                DestReuseOp<
                    cb_input,
                    EltwiseBinaryType::ELWMUL,
                    EltwiseBinaryReuseDestType::DEST_TO_SRCA,
                    Dst::D0,
                    DestReuseInputPolicy::NoWaitPop>{});
            eltwise_op<cb_output, Dst::D0, EltwiseOutputPolicy::Bulk>(
                chain, EltwiseTileShape::flat(per_core_block_dim));
        } else {
            auto chain = sfpu_chain(
                Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
                Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
                Log1p<Approx::Exact, Dst::D0>{},
                Tanh<Approx::Exact, Dst::D0>{},
                DestReuseOp<
                    cb_input,
                    EltwiseBinaryType::ELWMUL,
                    EltwiseBinaryReuseDestType::DEST_TO_SRCA,
                    Dst::D0,
                    DestReuseInputPolicy::NoWaitPop>{});
            eltwise_op<cb_output, Dst::D0, EltwiseOutputPolicy::Bulk>(
                chain, EltwiseTileShape::flat(per_core_block_dim));
        }
    }
#endif
}
