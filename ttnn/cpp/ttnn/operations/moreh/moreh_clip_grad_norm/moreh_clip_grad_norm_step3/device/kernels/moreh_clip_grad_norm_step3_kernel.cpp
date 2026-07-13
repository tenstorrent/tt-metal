// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    std::uint8_t input_id{0};
    const auto cb_x = input_id++;
    DataflowBuffer dfb_x_obj(cb_x);  // input
    const auto cb_clip_coef_clamped = input_id++;
    DataflowBuffer dfb_clip_coef_clamped_obj(cb_clip_coef_clamped);  // clip_coef_clamped

    std::uint8_t output_id{16};
    const auto cb_y = output_id++;
    DataflowBuffer dfb_y_obj(cb_y);  // output

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(cb_x, cb_clip_coef_clamped, cb_y);

    dfb_clip_coef_clamped_obj.wait_front(onetile);  // comes from the reader

    // Compute cb_y
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        tile_regs_acquire();
        dfb_x_obj.wait_front(onetile);  // comes from the reader
        dfb_y_obj.reserve_back(onetile);

        mul_tiles_bcast_scalar_init_short(cb_x, cb_clip_coef_clamped);
        mul_tiles_bcast_scalar(cb_x, cb_clip_coef_clamped, 0, 0, dst0);
        dfb_x_obj.pop_front(onetile);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(dst0, cb_y);
        dfb_y_obj.push_back(onetile);
        tile_regs_release();
    }

    dfb_clip_coef_clamped_obj.pop_front(onetile);
}
