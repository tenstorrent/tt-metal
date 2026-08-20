// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"  // MulUnary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

inline void run_addcmul(uint32_t num_tiles, uint32_t scalar_arg) {
    constexpr auto dfb_in0_id = tt::CBIndex::c_0;
    constexpr auto dfb_in1_id = tt::CBIndex::c_1;
    constexpr auto dfb_in2_id = tt::CBIndex::c_2;
    constexpr auto dfb_out_id = tt::CBIndex::c_3;

    // output = input_a + value * input_b * input_c
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles),
        // (input_b * input_c)
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Mul,
            ckl::input(
                dfb_in1_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::input(
                dfb_in2_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{},
        // Step 2: (input_b * input_c) * value -> DST[0]
        ckl::runtime_if(scalar_arg != 1u, ckl::MulUnary<ckl::Dst::D0>{scalar_arg}),  // DST[0] * scalar -> DST[0]
        // Now wait for input_a (only when we need it)
        // Step 3: Load A and add with result DST[0] + dfb_in0_id -> DST[0]
        ckl::DestReuseBinary<ckl::BinaryFpuOp::Add, ckl::input(dfb_in0_id), ckl::DestReuseType::DEST_TO_SRCA>{},
        ckl::PackTile<ckl::output(
            dfb_out_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr auto dfb_in1_id = tt::CBIndex::c_1;
    constexpr auto dfb_in2_id = tt::CBIndex::c_2;
    constexpr auto dfb_out_id = tt::CBIndex::c_3;

    compute_kernel_hw_startup(dfb_in1_id, dfb_in2_id, dfb_out_id);

    run_addcmul(num_tiles, scalar_arg);
}
