// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pv_rescale_fusion bench reader: Q / K / V CBs are backed by resident L1 shards (Q: one block, K / V: two key-block
// slots), so the reader only publishes credit — zero NoC traffic, the compute kernel is the bottleneck.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t Sq = get_compile_time_arg_val(3);
    constexpr uint32_t Sk = get_compile_time_arg_val(4);
    constexpr uint32_t DHt = get_compile_time_arg_val(5);
    constexpr uint32_t DVt = get_compile_time_arg_val(6);
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    const uint32_t n_blocks = get_arg_val<uint32_t>(1);
    for (uint32_t u = 0; u < n_units; ++u) {
        cb_reserve_back(cb_q, Sq * DHt);
        cb_push_back(cb_q, Sq * DHt);
        for (uint32_t b = 0; b < n_blocks; ++b) {
            cb_reserve_back(cb_k, Sk * DHt);
            cb_push_back(cb_k, Sk * DHt);
            cb_reserve_back(cb_v, Sk * DVt);
            cb_push_back(cb_v, Sk * DVt);
        }
    }
}
