// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 2-arg combined reconfig_data_format(curr_a, curr_b) (no _with_dt; unconditional reprogram).
//
// BinaryFpu(CbA,CbB) -> PackTile(CbOut). Element 0 is first, so both srca/srcb are first-emit
// (NO_PREV_DFB) and the chain emits the 2-arg combined overload. CbA=bfp8, CbB=fp32 is a max
// format delta, so any srca/srcb arg-routing regression fails.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    constexpr compute_kernel_lib::InputSpec default_input = compute_kernel_lib::input();
    constexpr compute_kernel_lib::OutputSpec default_output = compute_kernel_lib::output();
    static_assert(default_input.lifecycle == compute_kernel_lib::InputLifecycle::Streaming);
    static_assert(default_input.index == compute_kernel_lib::OperandKind::Scalar);
    static_assert(default_input.offset == compute_kernel_lib::TileOffset::Unset);
    static_assert(default_input.reconfig == compute_kernel_lib::DataFormatReconfig::Enabled);
    static_assert(default_output.lifecycle == compute_kernel_lib::OutputLifecycle::Streaming);
    static_assert(default_output.reconfig == compute_kernel_lib::DataFormatReconfig::Enabled);
    static_assert(default_output.relu == compute_kernel_lib::PackRelu::Disabled);
    static_assert(default_output.l1_accumulation == compute_kernel_lib::L1Accumulation::Disabled);
    static_assert(default_output.dest_accumulation == compute_kernel_lib::DestAccumulation::Disabled);
    static_assert(default_output.offset == compute_kernel_lib::TileOffset::Unset);

    using SrcAOnly = compute_kernel_lib::BinaryFpu<
        cb_a,
        cb_b,
        compute_kernel_lib::BinaryFpuOp::Add,
        compute_kernel_lib::BroadcastDim::None,
        compute_kernel_lib::input(
            compute_kernel_lib::InputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Enabled),
        compute_kernel_lib::input(
            compute_kernel_lib::InputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Disabled)>;
    using SrcBOnly = compute_kernel_lib::BinaryFpu<
        cb_a,
        cb_b,
        compute_kernel_lib::BinaryFpuOp::Add,
        compute_kernel_lib::BroadcastDim::None,
        compute_kernel_lib::input(
            compute_kernel_lib::InputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::input(
            compute_kernel_lib::InputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Enabled)>;
    static_assert(
        SrcAOnly::reconfig_srca_dfb == cb_a && SrcAOnly::reconfig_srcb_dfb == compute_kernel_lib::NO_PREV_DFB);
    static_assert(
        SrcBOnly::reconfig_srca_dfb == compute_kernel_lib::NO_PREV_DFB && SrcBOnly::reconfig_srcb_dfb == cb_b);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(total_tiles),
        compute_kernel_lib::BinaryFpu<cb_a, cb_b>{},
        compute_kernel_lib::PackTile<cb_out>{});
}
