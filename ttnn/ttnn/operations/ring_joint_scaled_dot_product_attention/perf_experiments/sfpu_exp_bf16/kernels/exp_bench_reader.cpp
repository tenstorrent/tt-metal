// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// sfpu_exp_bf16 bench reader: every input CB is backed by a resident L1 shard; just publish the credit.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t n_cbs = get_compile_time_arg_val(0);
    for (uint32_t i = 0; i < n_cbs; ++i) {
        const uint32_t cb = get_arg_val<uint32_t>(2 * i);
        const uint32_t n = get_arg_val<uint32_t>(2 * i + 1);
        cb_reserve_back(cb, n);
        cb_push_back(cb, n);
    }
}
