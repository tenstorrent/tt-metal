// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
// #include "debug/dprint.h"

#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    uint32_t src_cb_id = get_compile_time_arg_val(2);
    uint32_t out_cb_id = get_compile_time_arg_val(3);
    untilize_init(src_cb_id, out_cb_id);

// If LLK perf is measured on OP level put profiler zone around complete operation
#ifdef LLK_PERF_OP
    {
        DeviceZoneScopedN("UNTILIZE-OP")
#endif
            for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
// If LLK perf is measured disable sync with DM cores/kernels
#ifndef LLK_PERF_NO_DM
            cb_wait_front(src_cb_id, per_core_block_tile_cnt);
            cb_reserve_back(out_cb_id, per_core_block_tile_cnt);
#endif
// If LLK perf is measured on block level put profiler zone around *_block operation
#ifdef LLK_PERF_BLOCK
            {
                DeviceZoneScopedN("UNTILIZE-BLOCK");
#endif
                untilize_block(src_cb_id, per_core_block_tile_cnt, out_cb_id);
// If LLK perf is measured on block level put profiler zone around *_block operation
#ifdef LLK_PERF_BLOCK
            }
#endif
// If LLK perf is measured disable sync with DM cores/kernels
#ifndef LLK_PERF_NO_DM
            cb_push_back(out_cb_id, per_core_block_tile_cnt);
            cb_pop_front(src_cb_id, per_core_block_tile_cnt);
#endif
        }
// If LLK perf is measured on OP level put profiler zone around complete operation
#ifdef LLK_PERF_OP
    }
#endif
}
}  // namespace NAMESPACE
