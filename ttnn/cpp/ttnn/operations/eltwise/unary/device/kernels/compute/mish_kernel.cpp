// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t approx_arg = get_arg_val<uint32_t>(0);
    const bool use_approx = (approx_arg != 0u);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);
    init_sfpu(cb_input, cb_output);

#ifdef INP_FLOAT32
    // mish(x) = x * tanh(log1p(exp(x)))
    // FP32 path: use SFPU helpers with binary mul (needs two DEST slots)
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        if (use_approx) {
            auto chain = sfpu_chain(
                Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
                Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
                Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
                Log1p<Approx::Fast, Dst::D0>{},
                Tanh<Approx::Exact, Dst::D0>{},
                SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
            sfpu_pipeline<SfpuOutputPolicy::Bulk, SfpuDataFormatReconfig::NONE>(chain, cb_output, per_core_block_dim);
        } else {
            auto chain = sfpu_chain(
                Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
                Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
                Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
                Log1p<Approx::Exact, Dst::D0>{},
                Tanh<Approx::Exact, Dst::D0>{},
                SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
            sfpu_pipeline<SfpuOutputPolicy::Bulk, SfpuDataFormatReconfig::NONE>(chain, cb_output, per_core_block_dim);
        }
    }
#endif

#ifdef INP_FLOAT
    // BFloat16 path: use FPU binary_dest_reuse for the final x * tanh(log1p(exp(x))) multiply.
    // This preserves the original BFloat16 precision path (FPU mul vs SFPU mul in INP_FLOAT32).
    // DestReuseOp clobbers copy_tile init; sfpu_pipeline detects this via
    // chain_has_non_load_fpu_clash_v and reinits copy_tile_to_dst_init_short per tile.
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
                LoadPolicy::WaitAndPop>{});
        for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
            sfpu_pipeline<SfpuOutputPolicy::Bulk, SfpuDataFormatReconfig::NONE>(chain, cb_output, per_core_block_dim);
        }
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
                LoadPolicy::WaitAndPop>{});
        for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
            sfpu_pipeline<SfpuOutputPolicy::Bulk, SfpuDataFormatReconfig::NONE>(chain, cb_output, per_core_block_dim);
        }
    }
#endif
}
