// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "elemwise_add_kernel_args.hpp"
#include "ttnn/kernel/kernel_utils.hpp"
#include "../tiles_config.hpp"

#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"

#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel::eltwise::add_args;
    auto args = make_runtime_struct_from_args<ElemwiseComputeKernelArgs>();

    constexpr auto c_args = make_compile_time_struct_from_args<CompileTimeComputeKernelArgs>();
    constexpr auto cb_in0 = c_args.a_tensor_cb;
    constexpr auto cb_in1 = c_args.b_tensor_cb;
    constexpr auto cb_out0 = c_args.output_cb;
    DPRINT << "elemwise_add_kernel " << ENDL();

    // Metalium API Calls                              Involved Cores
    binary_op_init_common(cb_in0, cb_in1, cb_out0);  // Unpack, Math, Pack
    add_tiles_init(cb_in0, cb_in1);                  // Unpack, Math

    constexpr uint32_t num_tiles_per_cycle = c_args.num_tiles_per_cycle;
    for (uint32_t tile_id = args.tile_ofs; tile_id < args.tile_ofs + args.num_tiles; tile_id += num_tiles_per_cycle) {
        // wait for a tile to be ready in the input CBs
        cb_wait_front(cb_in0, num_tiles_per_cycle);  // Unpack
        cb_wait_front(cb_in1, num_tiles_per_cycle);  // Unpack

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

        cb_pop_front(cb_in0, num_tiles_per_cycle);  // Unpack
        cb_pop_front(cb_in1, num_tiles_per_cycle);  // Unpack

        cb_push_back(cb_out0, num_tiles_per_cycle);  // Pack
    }
}
