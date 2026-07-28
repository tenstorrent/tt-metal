// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Data-format reconfiguration scenarios:
//   0: both operands rotate with previous formats;
//   1: first binary operation configures both operands;
//   2: srcA rotates while srcB is first-use;
//   3: srcA-only rotation;
//   4: heterogeneous pack outputs;
//   5: repeated streaming reads from one CB.
//
// CT args: [num_tiles, mode].

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_d = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);
    static_assert(mode < 6);

    using namespace compute_kernel_lib;
    if constexpr (mode == 0) {
        compute_kernel_hw_startup(cb_a, cb_b, cb_out);
        eltwise_chain(
            EltwiseShape::tiles(num_tiles),
            BinaryFpu<input(cb_a), input(cb_b)>{},
            BinaryFpu<input(cb_c), input(cb_d)>{},
            PackTile<output(cb_out)>{});
    } else if constexpr (mode == 1) {
        constexpr InputSpec default_input = input(cb_a);
        constexpr OutputSpec default_output = output(cb_out);
        static_assert(default_input.cb_id == cb_a);
        static_assert(default_input.wait == WaitPolicy::PerTile);
        static_assert(default_input.pop == PopPolicy::PerTile);
        static_assert(default_input.index == OperandKind::Scalar);
        static_assert(default_input.offset == TileOffset::Unset);
        static_assert(default_input.reconfig == DataFormatReconfig::Enabled);
        static_assert(default_output.reserve == ReservePolicy::PerTile);
        static_assert(default_output.push == PushPolicy::PerTile);
        static_assert(default_output.cb_id == cb_out);
        static_assert(default_output.reconfig == DataFormatReconfig::Enabled);
        static_assert(default_output.relu == PackRelu::Disabled);
        static_assert(default_output.l1_accumulation == L1Accumulation::Disabled);
        static_assert(default_output.dest_accumulation == DestAccumulation::Disabled);
        static_assert(default_output.offset == TileOffset::Unset);

        using SrcAOnly = BinaryFpu<
            input(cb_a),
            input(cb_b, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled),
            BinaryFpuOp::Add,
            BroadcastDim::None>;
        using SrcBOnly = BinaryFpu<
            input(cb_a, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled),
            input(cb_b),
            BinaryFpuOp::Add,
            BroadcastDim::None>;
        static_assert(SrcAOnly::reconfig_srca_dfb == cb_a && SrcAOnly::reconfig_srcb_dfb == NO_PREV_DFB);
        static_assert(SrcBOnly::reconfig_srca_dfb == NO_PREV_DFB && SrcBOnly::reconfig_srcb_dfb == cb_b);

        compute_kernel_hw_startup(cb_a, cb_b, cb_out);
        eltwise_chain(
            EltwiseShape::tiles(num_tiles), BinaryFpu<input(cb_a), input(cb_b)>{}, PackTile<output(cb_out)>{});
    } else if constexpr (mode == 2) {
        compute_kernel_hw_startup(cb_a, cb_b, cb_out);
        eltwise_chain(
            EltwiseShape::tiles(num_tiles),
            CopyTile<input(cb_a)>{},
            BinaryFpu<input(cb_b), input(cb_c), BinaryFpuOp::Add, BroadcastDim::None, Dst::D1>{},
            AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
            PackTile<output(cb_out)>{});
    } else if constexpr (mode == 3) {
        compute_kernel_hw_startup(cb_a, cb_b, cb_out);
        eltwise_chain(
            EltwiseShape::tiles(num_tiles),
            CopyTile<input(cb_a)>{},
            CopyTile<input(cb_b)>{},
            PackTile<output(cb_out)>{});
    } else if constexpr (mode == 4) {
        constexpr uint32_t cb_out_2 = tt::CBIndex::c_17;
        compute_kernel_hw_startup(cb_a, cb_a, cb_out);
        eltwise_chain(
            EltwiseShape::tiles(num_tiles),
            CopyTile<input(cb_a)>{},
            PackTile<output(cb_out)>{},
            PackTile<output(cb_out_2)>{});
    } else {
        compute_kernel_hw_startup(cb_a, cb_a, cb_out);
        eltwise_chain(
            EltwiseShape::tiles(num_tiles),
            CopyTile<input(cb_a)>{},
            CopyTile<input(cb_a)>{},
            CopyTile<input(cb_a)>{},
            PackTile<output(cb_out)>{});
    }
}
