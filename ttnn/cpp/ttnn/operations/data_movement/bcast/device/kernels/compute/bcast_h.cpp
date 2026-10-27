// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

void kernel_main() {
    auto B = get_arg(args::B);
    auto Ht = get_arg(args::Ht);
    auto Wt = get_arg(args::Wt);
    DataflowBuffer dfb_in0_obj(dfb::in0);
    DataflowBuffer dfb_in1_obj(dfb::in1);
    DataflowBuffer dfb_out_obj(dfb::out);

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);

    // The reader repeats the RHS row every Wt tiles, so compute can consume both streams
    // linearly while broadcasting RHS down H.
    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::IterationShape::tiles(B * Ht * Wt),
        compute_kernel_lib::BinaryFpu<
            CHAIN_BCAST_OP,
            compute_kernel_lib::input(
                dfb::in0,
                compute_kernel_lib::WaitPolicy::PerTile,
                compute_kernel_lib::PopPolicy::PerTile,
                compute_kernel_lib::DataFormatReconfig::Disabled),
            compute_kernel_lib::input(
                dfb::in1,
                CHAIN_BCAST_DIM,
                compute_kernel_lib::WaitPolicy::PerTile,
                compute_kernel_lib::PopPolicy::PerTile,
                compute_kernel_lib::DataFormatReconfig::Disabled)>{dfb_in0_obj, dfb_in1_obj},
        compute_kernel_lib::PackTile<compute_kernel_lib::output(
            dfb::out,
            compute_kernel_lib::ReservePolicy::PerTile,
            compute_kernel_lib::PushPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled)>{dfb_out_obj});
}
