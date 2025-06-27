// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "reader_pool2d_sharded_common.hpp"
#include "compute_kernel_api/tile_move_copy.h"
#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(1);
    constexpr uint32_t nblocks = get_compile_time_arg_val(2);
    constexpr uint32_t ntiles = get_compile_time_arg_val(3);
    for (uint32_t i = 0; i < nblocks; ++i) {
        for (uint32_t j = 0; j < ntiles; ++j) {
            cb_wait_front(cb_src, 1);
            cb_reserve_back(cb_dst, 1);

            /***
             * TODO: check this! may be just one raw.
             * maybe noc async write
             * maybe reader untouched.
             * ***/
            copy_tile(cb_src, 0, 0);
            pack_tile(0, cb_dst);

            cb_push_back(cb_dst, 1);
            cb_pop_front(cb_src, 1);
        }
    }
}  // kernel_main()
