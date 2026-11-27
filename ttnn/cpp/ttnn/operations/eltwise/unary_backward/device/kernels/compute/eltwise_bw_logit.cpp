// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/logit_bw.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

// Reads D0 and D1 and writes D0; the kernel needs no scratch tile, so a lane spans 2 slots.
struct LogitBw : ckl::BinaryOp<LogitBw, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0> {
    static constexpr uint32_t lane_width = 2;
    static ALWI void init() { logit_bw_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { logit_bw_tile(slot_offset); }
};

void kernel_main() {
    compute_kernel_hw_startup(dfb::grad_out, dfb::grad_in);

    const uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);
    DataflowBuffer dfb_grad_out(dfb::grad_out);
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_grad_in(dfb::grad_in);

    const auto shape = ckl::IterationShape::tiles(per_core_tile_cnt);

    ckl::eltwise_chain(
        shape,
        // dest[0] = input
        ckl::CopyTile<
            ckl::input(
                dfb::input,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{dfb_input},
        // dest[1] = grad_out
        ckl::CopyTile<
            ckl::input(
                dfb::grad_out,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{dfb_grad_out},
        LogitBw{},  // dest[0] = grad_out * d/dx logit(input)
        ckl::PackTile<ckl::output(
            dfb::grad_in,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{dfb_grad_in});
}
