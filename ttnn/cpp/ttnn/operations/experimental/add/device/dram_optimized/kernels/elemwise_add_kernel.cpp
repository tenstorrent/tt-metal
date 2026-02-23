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

    // Metalium API Calls                              Involved Cores
    binary_op_init_common(cb_in0, cb_in1, cb_out0);  // Unpack, Math, Pack
    add_tiles_init(cb_in0, cb_in1);                  // Unpack, Math

    constexpr uint32_t num_tiles_per_cycle = c_args.num_tiles_per_cycle;

    DPRINT << "COMPUTE KERNEL: num_tiles: " << args.num_tiles << ", num_tiles_per_cycle " << num_tiles_per_cycle
           << ENDL();
    for (uint32_t num_processed_tiles = 0; num_processed_tiles < args.num_tiles;
         num_processed_tiles += num_tiles_per_cycle) {
        // wait for a tile to be ready in the input CBs
        {
            DeviceZoneScopedN("COMPUTE_KERNEL_WAIT_CB_DATA");
            cb_wait_front(cb_in0, num_tiles_per_cycle);  // Unpack
            cb_wait_front(cb_in1, num_tiles_per_cycle);  // Unpack
        }

        // DPRINT << "CB from reader are ready" << ENDL();

        // Take data from cb_in0 offset 0th page and
        // cb_in1 offset 0th page. Add them together
        // and store the result in cb_out0 (as
        // configured) offset 0th page.
        for (uint32_t i = 0; i < num_tiles_per_cycle; i++) {
            // acquire 8 tile registers to perform the addition
            tile_regs_acquire();                 // Math
            add_tiles(cb_in0, cb_in1, i, i, i);  // Unpack, Math
                                                 // signal the packer
            tile_regs_commit();                  // Math

            // packer waits here
            tile_regs_wait();  // Pack
            // Copy the result from tile registers to the
            // output circular buffer (also called packing)
            pack_tile(i, cb_out0);  // Pack
            // packer releases
            tile_regs_release();  // Pack
        }

        // DPRINT << "CB free processed tiles " << ENDL();
        cb_pop_front(cb_in0, num_tiles_per_cycle);  // Unpack
        cb_pop_front(cb_in1, num_tiles_per_cycle);  // Unpack

        // DPRINT << "Send out tile " << ENDL();
        cb_push_back(cb_out0, num_tiles_per_cycle);  // Pack
    }
    DPRINT << "Compute kernel is done" << ENDL();
}
