// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr bool is_all_to_all_worker = get_arg(args::is_all_to_all_worker) == 1;
    constexpr bool use_welford = get_arg(args::use_welford) == 1;
    constexpr uint32_t cb_in_2 = dfb::cb_scaler;  // host binds c_2 under DFB name "cb_scaler"
    const uint32_t scalar_w_bits = get_arg(args::packed_winv);
    float scalar_w_f = __builtin_bit_cast(float, scalar_w_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_in_2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scalar_w_f);

    if constexpr (is_all_to_all_worker && !use_welford) {
        constexpr uint32_t cb_in_4 = dfb::cb_scaler_global;
        const uint32_t scalar_c_bits = get_arg(args::packed_cinv);
        float scalar_c_f = __builtin_bit_cast(float, scalar_c_bits);
        dataflow_kernel_lib::prepare_reduce_scaler<
            cb_in_4,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            /*compute_uses_reduce_tile=*/true>(scalar_c_f);
    }
}
