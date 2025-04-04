// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "debug/dprint.h"

namespace NAMESPACE {

void MAIN {
    DPRINT << "Starting COMPUTE kernel" << ENDL();

    // Compile time args
    // TODO: More arguments
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr bool descending = get_compile_time_arg_val(5);
    constexpr bool stable = get_compile_time_arg_val(6);

    // Tensors config
    constexpr uint32_t one_tile = 1;

    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            DPRINT << "     > 2.1 Starting COMPUTE loop" << ENDL();
            // cb_reserve_back(index_tensor_output_cb_index, one_tile);
            // cb_wait_front(index_tensor_cb_index, one_tile);

            // binary_op_init_common(index_tensor_cb_index, index_tensor_output_cb_index, index_tensor_output_cb_index);
            // add_tiles_init(index_tensor_cb_index, index_tensor_output_cb_index);
            // tile_regs_acquire();
            // tile_regs_commit();  // signal the packer
            // tile_regs_wait();  // packer waits here
            // pack_tile(0, index_tensor_output_cb_index);
            // tile_regs_release();  // packer releases

            // // pack_tile(0, index_tensor_output_cb_index, 0, );

            // cb_pop_front(index_tensor_cb_index, one_tile);
            // cb_push_back(index_tensor_output_cb_index, one_tile);

        }  // Wt loop
    }  // Ht loop
}

}  // namespace NAMESPACE
