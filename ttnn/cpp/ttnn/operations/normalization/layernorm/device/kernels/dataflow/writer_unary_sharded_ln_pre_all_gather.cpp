// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"


void kernel_main() {
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_in_2 = tt::CB::c_in2;
    const uint32_t scalar_w = get_arg_val<uint32_t>(1);
    generate_reduce_scaler(cb_in_2, scalar_w);

    if constexpr(is_all_to_all_worker) {
        constexpr uint32_t cb_in_4 = tt::CB::c_in4;
        const uint32_t scalar_c = get_arg_val<uint32_t>(0);
        generate_reduce_scaler(cb_in_4, scalar_c);
    }
}
