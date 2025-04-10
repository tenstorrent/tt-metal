// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "../cumprod_common.hpp"

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

void MAIN {
    const auto compile_time_args{get_compile_time_args()};

    mul_tiles_init(compile_time_args.cb_input, compile_time_args.cb_acc);
    fill_tile_init();

    auto acc_handler{[](const CumprodCompileTimeArgs& args) -> void {
        const auto& cb_acc{args.cb_acc};
        tile_regs_acquire();
        cb_reserve_back(cb_acc, ONE_TILE);
        fill_tile(WORKING_REG, Scaler{get_dataformat(cb_acc)}.u);
        tile_regs_commit();
        tile_regs_wait();
        // TODO(jbbieniekTT): is cb_acc reusable without pushing/popping?
        pack_tile(WORKING_REG, cb_acc);
        cb_push_back(cb_acc, ONE_TILE);
        tile_regs_release();
    }};

    auto acc_end_handler{[](const CumprodCompileTimeArgs& args) -> void {
        const auto& cb_acc{args.cb_acc};
        cb_wait_front(cb_acc, ONE_TILE);
        cb_pop_front(cb_acc, ONE_TILE);
    }};

    auto tile_handler{
        [](const uint32_t& batch,
           const uint32_t& channel,
           const uint32_t& ht,
           const uint32_t& wt,
           const CumprodCompileTimeArgs& args) -> void {
            tile_regs_acquire();

            cb_wait_front(args.cb_input, ONE_TILE);
            copy_tile(args.cb_input, FIRST_TILE, WORKING_REG);
            mul_tiles(args.cb_input, args.cb_acc, FIRST_TILE, FIRST_TILE, WORKING_REG);

            tile_regs_commit();
            tile_regs_wait();

            cb_pop_front(args.cb_input, ONE_TILE);

            pack_tile(WORKING_REG, args.cb_acc);

            cb_reserve_back(args.cb_output, ONE_TILE);
            pack_tile(WORKING_REG, args.cb_output);
            cb_push_back(args.cb_output, ONE_TILE);

            tile_regs_release();
        }};

    for_each_tile_grouped_by_channels(compile_time_args, tile_handler, acc_handler, acc_end_handler);
}
}  // namespace NAMESPACE
