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

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);
    init_sfpu(cb_input, cb_output);

    // tanhshrink(x) = x - tanh(x)
    // DEST_TO_SRCB: loads x from CB into SRCB, then computes SRCA(x) - SRCB(tanh(x)).
    // DestReuseOp clobbers copy_tile init state; sfpu_pipeline reinits per tile.
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
        Tanh<Approx::Exact, Dst::D0>{},
        DestReuseOp<
            cb_input,
            EltwiseBinaryType::ELWSUB,
            EltwiseBinaryReuseDestType::DEST_TO_SRCB,
            Dst::D0,
            LoadPolicy::WaitAndPop>{});

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        sfpu_pipeline<SfpuOutputPolicy::Bulk, SfpuDataFormatReconfig::NONE, SfpuBatching::Auto>(
            chain, cb_output, per_core_block_dim);
    }
}
