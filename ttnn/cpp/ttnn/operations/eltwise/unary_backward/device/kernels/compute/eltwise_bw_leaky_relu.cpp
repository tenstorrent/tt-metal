// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// grad * d/dx leaky_relu(x), evaluated over BF16 DEST by leaky_relu_bw_tile, which reads the input
// from dest[0] and the incoming gradient from dest[1].

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/leaky_relu_bw.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace ckl = compute_kernel_lib;

// Reads D0 and D1 and writes D0; the kernel needs no scratch tile, so a lane spans 2 slots.
struct LeakyReluBw : ckl::BinaryOp<LeakyReluBw, ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0> {
    static constexpr uint32_t lane_width = 2;
    static ALWI void init() { leaky_relu_bw_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { leaky_relu_bw_tile(slot_offset); }
};

void kernel_main() {
    const uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto dfb_grad_out_id = tt::CBIndex::c_0;
    constexpr auto dfb_input_id = tt::CBIndex::c_1;
    constexpr auto dfb_grad_in_id = tt::CBIndex::c_2;

    compute_kernel_hw_startup(dfb_grad_out_id, dfb_grad_in_id);

    const auto shape = ckl::IterationShape::tiles(per_core_tile_cnt);

    ckl::eltwise_chain(
        shape,
        // dest[0] = input
        ckl::CopyTile<
            ckl::input(
                dfb_input_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        // dest[1] = grad_out
        ckl::CopyTile<
            ckl::input(
                dfb_grad_out_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        LeakyReluBw{},  // dest[0] = grad_out * d/dx leaky_relu(input)
        ckl::PackTile<ckl::output(
            dfb_grad_in_id,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
}
