// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_input_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t input_granularity = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
#ifdef FUSE_EPILOGUE
    constexpr auto cb_epilogue_a = tt::CBIndex::c_2;
    constexpr auto cb_epilogue_b = tt::CBIndex::c_3;
#endif
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t dst2 = 2;
    constexpr uint32_t first_tile = 0;

    experimental::CircularBuffer cb_in0_obj(cb_in0);
    experimental::CircularBuffer cb_in1_obj(cb_in1);
    experimental::CircularBuffer cb_out0_obj(cb_out0);
#ifdef FUSE_EPILOGUE
    experimental::CircularBuffer cb_epilogue_a_obj(cb_epilogue_a);
    experimental::CircularBuffer cb_epilogue_b_obj(cb_epilogue_b);
#endif

    constexpr uint32_t num_input_tiles_iter = num_input_tiles / input_granularity;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    cb_in1_obj.wait_front(onetile);

    // For each assigned output tile, process the input tiles in a doubly nested
    // loop. The inner loop processes the number of tiles specified by
    // input_granularity. The outer loop executes num_input_tiles / input_granularity
    // times.
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        add_tiles_init(cb_in0, cb_in1, true);
        reconfig_data_format(cb_in0, cb_in1);
        tile_regs_acquire();
        for (uint32_t j = 0; j < num_input_tiles_iter; ++j) {
            cb_in0_obj.wait_front(input_granularity);
            for (uint32_t k = 0; k < input_granularity; k++) {
                add_tiles(cb_in0, cb_in1, k, first_tile, dst0);
            }
            cb_in0_obj.pop_front(input_granularity);
        }
#ifdef FUSE_EPILOGUE
        cb_epilogue_a_obj.wait_front(onetile);
        reconfig_data_format_srca(cb_epilogue_a);
        copy_tile_init(cb_epilogue_a);
        copy_tile(cb_epilogue_a, first_tile, dst1);
        cb_epilogue_a_obj.pop_front(onetile);

        cb_epilogue_b_obj.wait_front(onetile);
        reconfig_data_format_srca(cb_epilogue_b);
        copy_tile_init(cb_epilogue_b);
        copy_tile(cb_epilogue_b, first_tile, dst2);
        cb_epilogue_b_obj.pop_front(onetile);

        add_binary_tile_init();
        add_binary_tile(dst0, dst1, dst0);
        add_binary_tile(dst0, dst2, dst0);
#endif
        tile_regs_commit();
        cb_out0_obj.reserve_back(onetile);
        pack_reconfig_data_format(cb_out0);
        tile_regs_wait();
        pack_tile(dst0, cb_out0);
        tile_regs_release();
        cb_out0_obj.push_back(onetile);
    }
}
