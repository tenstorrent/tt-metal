// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #include "compute_kernel_api.h"
// #include "compute_kernel_api/common.h"
// #include "compute_kernel_api/eltwise_binary.h"
// #include "compute_kernel_api/eltwise_unary/fill.h"

// #include "../cumprod_common.hpp"

// #include "debug/dprint.h"

// namespace {
// FORCE_INLINE void process_tile(
//     const uint32_t& batch,
//     const uint32_t& channel,
//     const uint32_t& ht,
//     const uint32_t& wt,
//     const CumprodCompileTimeArgs& args) {
//     cb_wait_front(args.cb_input, ONE_TILE);
//     tile_regs_acquire();

//     mul_tiles_init(args.cb_input, args.cb_acc);
//     mul_tiles(args.cb_input, args.cb_acc, FIRST_TILE, FIRST_TILE, WORKING_REG);

//     cb_pop_front(args.cb_input, ONE_TILE);

//     tile_regs_commit();
//     tile_regs_wait();

//     pack_tile(WORKING_REG, args.cb_acc);

//     cb_reserve_back(args.cb_output, ONE_TILE);
//     pack_tile(WORKING_REG, args.cb_output);
//     cb_push_back(args.cb_output, ONE_TILE);

//     tile_regs_release();
// }

// FORCE_INLINE void process_tiles(const CumprodCompileTimeArgs& compile_time_args) {
//     for (uint32_t b{0}; b < compile_time_args.batches + 1; ++b) {
//         for (uint32_t ht{0}; ht < compile_time_args.height_tiles; ++ht) {
//             for (uint32_t wt{0}; wt < compile_time_args.width_tiles; ++wt) {
//                 // prepare cb_acc
//                 tile_regs_acquire();

//                 cb_reserve_back(compile_time_args.cb_acc, ONE_TILE);

//                 fill_tile_init();
//                 fill_tile(WORKING_REG, ACC_START_VALUE);

//                 tile_regs_commit();
//                 tile_regs_wait();

//                 pack_tile(WORKING_REG, compile_time_args.cb_acc);

//                 tile_regs_release();

//                 // process tiles along the channel axis using cb_acc
//                 for (uint32_t c{0}; c < compile_time_args.channels; ++c) {
//                     process_tile(b, c, ht, wt, compile_time_args);
//                 }

//                 // release cb_acc
//                 cb_push_back(compile_time_args.cb_acc, ONE_TILE);
//                 cb_wait_front(compile_time_args.cb_acc, ONE_TILE);
//                 cb_pop_front(compile_time_args.cb_acc, ONE_TILE);
//             }
//         }
//     }
// }
// }  // namespace

// namespace NAMESPACE {
// void MAIN {
//     constexpr auto compile_time_args{get_compile_time_args()};

//     unary_op_init_common();

//     process_tiles(compile_time_args);
// }
// }  // namespace NAMESPACE

// #include <stdint.h>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

#define APPROX false
#include "compute_kernel_api/add_int32_sfpu.h"
#include "compute_kernel_api/common.h"
#include "debug/dprint_tensix.h"
#include "debug/dprint.h"

#include "tt_metal/hw/inc/debug/dprint_tensix.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_rows = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_1;
    constexpr uint32_t cb_one = tt::CBIndex::c_16;
    constexpr uint32_t cb_intermed = tt::CBIndex::c_24;

    constexpr uint32_t TILE_DEST = 0;
    constexpr uint32_t TILE_ACC = 1;

    constexpr uint32_t first_tile = 0;

    unary_op_init_common(cb_in, cb_out);

    cb_wait_front(cb_one, 1);

    for (unsigned i = 0; i < num_rows; i++) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_one);
        copy_tile(cb_one, first_tile, TILE_DEST);
        // dprint_tensix_dest_reg(TILE_DEST);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_intermed, 1);
        pack_tile(TILE_DEST, cb_intermed);
        cb_push_back(cb_intermed, 1);
        tile_regs_release();

        // DPRINT << "[COMPUTE] ONE and ACC complete" << ENDL();

        for (unsigned j = 0; j < tiles_per_row; j++) {
            cb_wait_front(cb_in, 1);
            // copy_tile_to_dst_init_short(cb_in);
            // copy_tile(cb_in, first_tile, TILE_DEST);

            cb_wait_front(cb_intermed, 1);
            // copy_tile_to_dst_init_short(cb_intermed);
            // copy_tile(cb_intermed, first_tile, TILE_ACC);

            // DPRINT << "000000000" << ENDL();
            // dprint_tensix_dest_reg(TILE_ACC);

            tile_regs_acquire();  // acquire 8 tile registers

            // DPRINT << "111111111" << ENDL();
            // dprint_tensix_dest_reg(TILE_DEST);

            mul_tiles_init(cb_in, cb_intermed);
            mul_tiles(cb_in, cb_intermed, 0, 0, TILE_DEST);

            // DPRINT << "2222222222" << ENDL();
            // dprint_tensix_dest_reg(TILE_DEST);

            tile_regs_commit();

            cb_pop_front(cb_in, 1);
            cb_pop_front(cb_intermed, 1);

            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(TILE_DEST, cb_out);
            cb_push_back(cb_out, 1);

            cb_reserve_back(cb_intermed, 1);
            pack_tile(TILE_DEST, cb_intermed);
            cb_push_back(cb_intermed, 1);

            tile_regs_release();  // release 8 tile registers
            // DPRINT << "[COMPUTE] 3 PACK complete" << ENDL();
        }

        // If we keep reserve_back() and push_back() into the CircularBuffer
        // then it will eventually get filled if multiple iterations are performed.
        // To avoid this, we pop the circular buffer.
        cb_wait_front(cb_intermed, 1);
        cb_pop_front(cb_intermed, 1);
        // DPRINT << "[COMPUTE] 4 complete" << ENDL();
    }

    cb_pop_front(cb_one, 1);
}

}  // namespace NAMESPACE
