// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(0) == 1;
    constexpr bool use_welford = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
    const uint32_t scalar_w_bits = get_arg_val<uint32_t>(1);
    float scalar_w_f = __builtin_bit_cast(float, scalar_w_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_in_2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scalar_w_f);

    if constexpr (is_all_to_all_worker && !use_welford) {
        constexpr uint32_t cb_in_4 = tt::CBIndex::c_4;
        const uint32_t scalar_c_bits = get_arg_val<uint32_t>(0);
        float scalar_c_f = __builtin_bit_cast(float, scalar_c_bits);
        dataflow_kernel_lib::prepare_reduce_scaler<
            cb_in_4,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            tt::constants::TILE_WIDTH,
            true>(scalar_c_f);
    }
}
