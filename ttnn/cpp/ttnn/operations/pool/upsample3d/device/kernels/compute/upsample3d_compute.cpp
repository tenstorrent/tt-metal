// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t i = 0; i < per_core_block_cnt; ++i) {
        copy_tile_init();
        copy_tile(cb_in, 0, cb_out, 0);
        copy_tile_done();
    }
}
}  // namespace NAMESPACE
