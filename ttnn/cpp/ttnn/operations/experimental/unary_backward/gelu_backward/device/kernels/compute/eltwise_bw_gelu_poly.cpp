// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for GELU backward using polynomial-based GELU derivative
// Uses Sollya-derived minimax polynomials for high accuracy (Max ULP = 1)

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    // grad_out / input are consumed from the reader; grad_in is produced for the writer.
    DataflowBuffer dfb_grad_out(dfb::grad_out);
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_grad_in(dfb::grad_in);

    compute_kernel_hw_startup(dfb::grad_out, dfb::grad_in);
    copy_init(dfb::grad_out);
    gelu_derivative_tile_init<false>();
    mul_binary_tile_init();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_grad_in.reserve_back(1);
        dfb_grad_out.wait_front(1);
        dfb_input.wait_front(1);

        tile_regs_acquire();

        copy_tile(dfb::grad_out, 0, 0);  // dest[0] = grad_out
        copy_tile(dfb::input, 0, 1);     // dest[1] = input
        gelu_derivative_tile<false>(1);  // dest[1] = GELU'(input)
        mul_binary_tile(0, 1, 0);        // dest[0] = grad_out * GELU'(input)

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dfb::grad_in);

        tile_regs_release();

        dfb_grad_out.pop_front(1);
        dfb_input.pop_front(1);
        dfb_grad_in.push_back(1);
    }
}
