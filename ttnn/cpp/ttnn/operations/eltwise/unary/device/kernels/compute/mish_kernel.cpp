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
    const bool use_approx = (get_arg_val<uint32_t>(0) != 0u);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);
    init_sfpu(cb_input, cb_output);

#ifdef INP_FLOAT32
    // FP32 path: SFPU mul_binary_tile for final x * tanh(softplus(x)).
    // Uses sfpu_chain with fan-out Load × 2 and SfpuMul (already SFPU in original).
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
    // BF16 path: FPU binary_dest_reuse for final x * tanh(softplus(x)).
    // Load<WaitNoPop>: wait once, copy x → D0, do NOT pop yet.
    // Exp + Log1p + Tanh: D0 = tanh(softplus(x)).
    // DestReuseOp<ELWMUL, DEST_TO_SRCA, NoWaitPop>:
    //   SRCA ← D0, SRCB ← cb_input[0] (x) → D0 = tanh(softplus(x)) * x; pops tile.
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
