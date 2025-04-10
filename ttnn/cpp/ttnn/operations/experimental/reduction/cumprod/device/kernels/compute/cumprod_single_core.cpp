// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/fill.h"

#include "../cumprod_common.hpp"

#include "debug/dprint.h"

namespace {
FORCE_INLINE void process_tile(
    const uint32_t& batch,
    const uint32_t& channel,
    const uint32_t& ht,
    const uint32_t& wt,
    const CumprodCompileTimeArgs& args) {
    cb_wait_front(args.cb_input, ONE_TILE);
    tile_regs_acquire();

    mul_tiles_init(args.cb_input, args.cb_acc);
    mul_tiles(args.cb_input, args.cb_acc, FIRST_TILE, FIRST_TILE, WORKING_REG);

    cb_pop_front(args.cb_input, ONE_TILE);

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(WORKING_REG, args.cb_acc);

    cb_reserve_back(args.cb_output, ONE_TILE);
    pack_tile(WORKING_REG, args.cb_output);
    cb_push_back(args.cb_output, ONE_TILE);

    tile_regs_release();
}

FORCE_INLINE void process_tiles(const CumprodCompileTimeArgs& compile_time_args) {
    for (uint32_t b{0}; b < compile_time_args.batches + 1; ++b) {
        for (uint32_t ht{0}; ht < compile_time_args.height_tiles; ++ht) {
            for (uint32_t wt{0}; wt < compile_time_args.width_tiles; ++wt) {
                // prepare cb_acc
                tile_regs_acquire();

                cb_reserve_back(compile_time_args.cb_acc, ONE_TILE);

                fill_tile_init();
                fill_tile(WORKING_REG, ACC_START_VALUE);

                tile_regs_commit();
                tile_regs_wait();

                pack_tile(WORKING_REG, compile_time_args.cb_acc);

                tile_regs_release();

                // process tiles along the channel axis using cb_acc
                for (uint32_t c{0}; c < compile_time_args.channels; ++c) {
                    process_tile(b, c, ht, wt, compile_time_args);
                }

                // release cb_acc
                cb_push_back(compile_time_args.cb_acc, ONE_TILE);
                cb_wait_front(compile_time_args.cb_acc, ONE_TILE);
                cb_pop_front(compile_time_args.cb_acc, ONE_TILE);
            }
        }
    }
}
}  // namespace

namespace NAMESPACE {
void MAIN {
    constexpr auto compile_time_args{get_compile_time_args()};

    process_tiles(compile_time_args);
}
}  // namespace NAMESPACE
