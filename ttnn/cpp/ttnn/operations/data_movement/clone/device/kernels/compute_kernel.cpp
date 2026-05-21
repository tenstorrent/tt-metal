// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Clone dtype-conversion compute kernel, ported to Metal 2.0.
//
// Host bindings expected (per CloneOperation::ProgramFactory's compute KernelSpec):
//   compile_time_arg_bindings: { {"num_units", ...} }   (per-group; multiplicity preserved)
//   dfb_bindings: { INPUT_DFB (CONSUMER, name="src_dfb"),
//                   OUTPUT_DFB (PRODUCER, name="dst_dfb") }

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_units = get_arg(args::num_units);
    DataflowBuffer src_dfb(dfb::src_dfb);
    DataflowBuffer dst_dfb(dfb::dst_dfb);
    unary_op_init_common(dfb::src_dfb, dfb::dst_dfb);
    for (uint32_t i = 0; i < num_units; ++i) {
        src_dfb.wait_front(1);
        tile_regs_acquire();
        copy_tile(dfb::src_dfb, 0, 0);
        tile_regs_commit();
        src_dfb.pop_front(1);

        dst_dfb.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::dst_dfb, 0);
        tile_regs_release();
        dst_dfb.push_back(1);
    }
}
