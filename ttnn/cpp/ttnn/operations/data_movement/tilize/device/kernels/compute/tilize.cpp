// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(3);
    compute_kernel_hw_startup(cb_id_in0, cb_id_out0);
    compute_kernel_lib::tilize(cb_id_in0, per_core_block_tile_cnt, cb_id_out0, per_core_block_cnt);
}
}  // namespace NAMESPACE
