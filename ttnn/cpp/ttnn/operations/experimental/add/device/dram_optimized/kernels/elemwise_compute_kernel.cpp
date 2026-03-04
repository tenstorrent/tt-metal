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

    constexpr auto ct_args = make_compile_time_struct_from_args<EltwiseComputeCTArgs>();
    constexpr auto cb_in0 = ct_args.a_tensor_cb;
    constexpr auto cb_in1 = ct_args.b_tensor_cb;
    constexpr auto cb_out0 = ct_args.output_cb;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    // add_tiles_init(cb_in0, cb_in1);

    mul_tiles_init(cb_in0, cb_in1);

    auto num_batches = ct_args.num_batches;
    auto num_tiles_per_batch = ct_args.num_tiles_per_batch;
    const uint32_t large_chunk = num_batches * num_tiles_per_batch;

    uint32_t remaining = args.num_tiles;
    while (remaining > 0) {
        uint32_t n_tiles_proc;
        if (remaining >= large_chunk) {
            n_tiles_proc = large_chunk;
        } else if (remaining >= num_tiles_per_batch) {
            n_tiles_proc = num_tiles_per_batch;
        } else {
            n_tiles_proc = remaining;
        }

        cb_wait_front(cb_in0, n_tiles_proc);
        cb_wait_front(cb_in1, n_tiles_proc);

        cb_reserve_back(cb_out0, n_tiles_proc);

        tile_regs_acquire();
        for (uint32_t i = 0; i < n_tiles_proc; i++) {
            // add_tiles(cb_in0, cb_in1, i, i, i);
            mul_tiles(cb_in0, cb_in1, i, i, i);
        }
        tile_regs_commit();

        tile_regs_wait();

        for (uint32_t i = 0; i < n_tiles_proc; i++) {
            pack_tile(i, cb_out0);
        }

        cb_push_back(cb_out0, n_tiles_proc);
        tile_regs_release();

        cb_pop_front(cb_in0, n_tiles_proc);
        cb_pop_front(cb_in1, n_tiles_proc);
        remaining -= n_tiles_proc;
    }
}
