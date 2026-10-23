// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    experimental::CB cb(cb_in);
    cb.push_back(total_tiles);
}
