// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "elemwise_args_kernel.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"

#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary.h"

#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace eltwise_dram_optimized;
    auto args = make_runtime_struct_from_args<EltwiseComputeArgs>();

    constexpr auto c_args = make_compile_time_struct_from_args<EltwiseComputeCTArgs>();
    constexpr auto cb_in0 = c_args.a_tensor_cb;
    constexpr auto cb_in1 = c_args.b_tensor_cb;
    constexpr auto cb_out0 = c_args.output_cb;

    constexpr uint32_t max_num_tiles_per_batch = 4;
    uint32_t num_tiles_per_batch = args.num_tiles > max_num_tiles_per_batch ? max_num_tiles_per_batch : args.num_tiles;

    // Metalium API Calls                              Involved Cores
    binary_op_init_common(cb_in0, cb_in1, cb_out0);  // Unpack, Math, Pack
    add_tiles_init(cb_in0, cb_in1);                  // Unpack, Math

    // DPRINT << "COMPUTE KERNEL: num_tiles: " << args.num_tiles << ", num_tiles_per_batch " << num_tiles_per_batch
    //        << ENDL();

    auto num_tail_tiles = args.num_tiles % num_tiles_per_batch;
    auto num_tiles = args.num_tiles - num_tail_tiles;

    for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id += num_tiles_per_batch) {
        {
            DeviceZoneScopedN("WAIT_CB_DATA");
            // cb_wait_front(cb_in0, num_tiles_per_batch);  // Unpack
            // cb_wait_front(cb_in1, num_tiles_per_batch);  // Unpack
        }
        // cb_reserve_back(cb_out0, num_tiles_per_batch);

        // {
        //     DeviceZoneScopedN("ADD_TILES");
        //     // acquire 8 tile registers to perform the addition
        //     tile_regs_acquire();
        //     for (uint32_t i = 0; i < num_tiles_per_batch; i++) {
        //         // DPRINT << "Compute kernel is processing tile " << i << ENDL();
        //         add_tiles(cb_in0, cb_in1, i, i, i);
        //     }
        //     tile_regs_commit();
        // }

        {
            // DeviceZoneScopedN("PACK_TILES");  // TRISC 2
            //   packer waits here
            // tile_regs_wait();  // Pack

            // for (uint32_t i = 0; i < num_tiles_per_batch; i++) {
            //     // DPRINT << "Compute kernel is packing tile " << i << ENDL();
            //     pack_tile(i, cb_out0);  // Pack
            // }

            // DPRINT << "Compute kernel is releasing tile " << i << ENDL();
            // cb_push_back(cb_out0, num_tiles_per_batch);  // Pack
            // tile_regs_release();                         // Pack
        }

        // DPRINT << "CB free processed input tiles " << ENDL();
        // cb_pop_front(cb_in0, num_tiles_per_batch);  // Unpack
        // cb_pop_front(cb_in1, num_tiles_per_batch);  // Unpack

        // DPRINT << "Send out tile " << ENDL();
    }

    if (num_tail_tiles != 0) {
        num_tiles_per_batch = num_tail_tiles;
        {
            // DeviceZoneScopedN("COMPUTE_KERNEL_WAIT_CB_DATA");
            // cb_wait_front(cb_in0, num_tiles_per_batch);  // Unpack
            // cb_wait_front(cb_in1, num_tiles_per_batch);  // Unpack
        }
        // cb_reserve_back(cb_out0, num_tiles_per_batch);

        // // acquire 8 tile registers to perform the addition
        // tile_regs_acquire();
        // for (uint32_t i = 0; i < num_tiles_per_batch; i++) {
        //     // DPRINT << "Compute kernel is processing tile " << i << ENDL();
        //     add_tiles(cb_in0, cb_in1, i, i, i);
        // }
        // tile_regs_commit();

        // // packer waits here
        // tile_regs_wait();  // Pack

        // for (uint32_t i = 0; i < num_tiles_per_batch; i++) {
        //     // DPRINT << "Compute kernel is packing tile " << i << ENDL();
        //     pack_tile(i, cb_out0);  // Pack
        // }

        // DPRINT << "Compute kernel is releasing tile " << i << ENDL();
        // cb_push_back(cb_out0, num_tiles_per_batch);  // Pack
        // tile_regs_release();                         // Pack

        // DPRINT << "CB free processed input tiles " << ENDL();
        // cb_pop_front(cb_in0, num_tiles_per_batch);  // Unpack
        // cb_pop_front(cb_in1, num_tiles_per_batch);  // Unpack
    }
    // DPRINT << "Compute kernel is done" << ENDL();
}
