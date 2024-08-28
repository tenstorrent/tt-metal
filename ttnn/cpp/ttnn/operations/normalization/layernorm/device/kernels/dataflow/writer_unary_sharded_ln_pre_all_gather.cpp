// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"


void kernel_main() {
    constexpr uint32_t cb_in_4 = tt::CB::c_in4;
    const uint32_t scalar_c = get_arg_val<uint32_t>(0);
    generate_reduce_scaler(cb_in_4, scalar_c);
}
