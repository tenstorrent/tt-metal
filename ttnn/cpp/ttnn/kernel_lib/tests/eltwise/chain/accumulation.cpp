// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Accumulation scenarios. CT args:
// [mode, tiles_per_output, block_size, caller_managed, num_outputs, whole_shape]
// mode 0 accumulates in DEST; mode 1 accumulates through an L1 output CB.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr bool caller_managed = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t num_outputs = get_compile_time_arg_val(4);
    constexpr bool whole_shape = get_compile_time_arg_val(5) != 0;
    static_assert(mode < 2);
    static_assert(n > 0);

    using namespace compute_kernel_lib;
    if constexpr (mode == 0) {
        constexpr uint32_t cb_a = tt::CBIndex::c_0;
        constexpr uint32_t cb_b = tt::CBIndex::c_1;
        constexpr uint32_t cb_out = tt::CBIndex::c_16;
        static_assert(block_size > 0);
        static_assert(num_outputs > 0);
        compute_kernel_hw_startup(cb_a, cb_b, cb_out);

        DataflowBuffer cb_a_obj(cb_a);
        DataflowBuffer cb_b_obj(cb_b);

        using PerRowAccumulate = BinaryFpu<
            BinaryFpuOp::Add,
            input(cb_a, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            input(cb_b, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            Dst::D0,
            DestAccumulation::PerRow>;
        using PerRowManagedPack = PackTile<output(
            cb_out,
            ReservePolicy::PerOuter,
            PushPolicy::PerOuter,
            DataFormatReconfig::Enabled,
            TileAddressing::Direct,
            DestAccumulation::PerRow)>;
        using PerRowCallerPack = PackTile<output(
            cb_out,
            ReservePolicy::None,
            PushPolicy::None,
            DataFormatReconfig::Enabled,
            TileAddressing::Direct,
            DestAccumulation::PerRow)>;
        using WholeShapeAccumulate = BinaryFpu<
            BinaryFpuOp::Add,
            input(cb_a, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            input(cb_b, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            Dst::D0,
            DestAccumulation::WholeShape>;
        using WholeShapeManagedPack = PackTile<output(
            cb_out,
            ReservePolicy::OneUpfront,
            PushPolicy::OneAtEnd,
            DataFormatReconfig::Enabled,
            TileAddressing::Direct,
            DestAccumulation::WholeShape)>;
        using WholeShapeCallerPack = PackTile<output(
            cb_out,
            ReservePolicy::None,
            PushPolicy::None,
            DataFormatReconfig::Enabled,
            TileAddressing::Direct,
            DestAccumulation::WholeShape)>;

        DataflowBuffer output_buffer(cb_out);
        if constexpr (whole_shape) {
            if constexpr (caller_managed) {
                output_buffer.reserve_back(1);
                eltwise_chain(
                    IterationShape::grid(num_outputs, n).block_size(block_size),
                    WholeShapeAccumulate{cb_a_obj, cb_b_obj},
                    WholeShapeCallerPack{output_buffer});
                output_buffer.push_back(1);
            } else {
                eltwise_chain(
                    IterationShape::grid(num_outputs, n).block_size(block_size),
                    WholeShapeAccumulate{cb_a_obj, cb_b_obj},
                    WholeShapeManagedPack{output_buffer});
            }
        } else {
            if constexpr (caller_managed) {
                output_buffer.reserve_back(num_outputs);
                eltwise_chain(
                    IterationShape::grid(num_outputs, n).block_size(block_size),
                    PerRowAccumulate{cb_a_obj, cb_b_obj},
                    PerRowCallerPack{output_buffer});
                output_buffer.push_back(num_outputs);
            } else {
                eltwise_chain(
                    IterationShape::grid(num_outputs, n).block_size(block_size),
                    PerRowAccumulate{cb_a_obj, cb_b_obj},
                    PerRowManagedPack{output_buffer});
            }
        }
    } else {
        constexpr uint32_t cb_in = tt::CBIndex::c_0;
        constexpr uint32_t cb_acc = tt::CBIndex::c_15;
        constexpr uint32_t cb_out = tt::CBIndex::c_16;
        static_assert(n > 1);
        compute_kernel_hw_startup(cb_in, cb_acc);

        DataflowBuffer cb_in_obj(cb_in);
        DataflowBuffer accumulator(cb_acc);
        DataflowBuffer cb_out_obj(cb_out);
        using ManagedPack = PackTile<output(
            cb_acc,
            ReservePolicy::OneUpfront,
            PushPolicy::OneAtEnd,
            DataFormatReconfig::Disabled,
            TileAddressing::Direct,
            DestAccumulation::Disabled,
            L1Accumulation::Enabled)>;
        using CallerPack = PackTile<output(
            cb_acc,
            ReservePolicy::None,
            PushPolicy::None,
            DataFormatReconfig::Disabled,
            TileAddressing::Direct,
            DestAccumulation::Disabled,
            L1Accumulation::Enabled)>;

        if constexpr (caller_managed) {
            accumulator.reserve_back(1);
            eltwise_chain(
                IterationShape::tiles(n),
                CopyTile<input(cb_in, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled)>{
                    cb_in_obj},
                CallerPack{accumulator});
            accumulator.push_back(1);
        } else {
            eltwise_chain(
                IterationShape::tiles(n),
                CopyTile<input(cb_in, WaitPolicy::PerTile, PopPolicy::PerTile, DataFormatReconfig::Disabled)>{
                    cb_in_obj},
                ManagedPack{accumulator});
        }
        eltwise_chain(
            IterationShape::one_tile(), CopyTile<input(cb_acc)>{accumulator}, PackTile<output(cb_out)>{cb_out_obj});
    }
}
