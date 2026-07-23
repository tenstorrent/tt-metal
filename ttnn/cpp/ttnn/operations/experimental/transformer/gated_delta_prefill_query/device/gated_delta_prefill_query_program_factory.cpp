// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the experimental gated-delta prefill-then-query op.
//
// NOTE: This is SCAFFOLDING. The placeholder kernels do NOT yet compute the gated
// delta-rule recurrence. They wire the full data path end-to-end so the op builds,
// dispatches, and returns correctly-shaped/typed outputs:
//   * state' = exact copy of the input state (a verifiable passthrough), and
//   * O      = placeholder values (copied from state tiles).
// Inputs q/k/v/gate/decay are validated but unused by the placeholder compute;
// they are reserved for the real recurrence kernel.

#include "gated_delta_prefill_query_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor GatedDeltaPrefillQueryProgramFactory::create_descriptor(
    const GatedDeltaPrefillQueryParams& attrs, const GatedDeltaPrefillQueryInputs& in, std::vector<Tensor>& outputs) {
    const auto& state = in.state;
    Tensor& o_out = outputs[0];      // [1, 1, Nv, d]  bf16
    Tensor& state_out = outputs[1];  // [1, Nv, d, d]  fp32

    // Tile counts driven directly off the physical tensor volumes so the scaffold
    // is agnostic to head-count / dim padding.
    const uint32_t num_state_tiles = static_cast<uint32_t>(state_out.physical_volume() / TILE_HW);
    const uint32_t num_o_tiles = static_cast<uint32_t>(o_out.physical_volume() / TILE_HW);

    const tt::DataFormat state_df = datatype_to_dataformat_converter(state.dtype());  // fp32
    const tt::DataFormat o_df = datatype_to_dataformat_converter(o_out.dtype());      // bf16
    const uint32_t state_tile_bytes = tt::tile_size(state_df);
    const uint32_t o_tile_bytes = tt::tile_size(o_df);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(state.device()->arch(), attrs.compute_kernel_config);

    // Single-core scaffold.
    CoreRangeSet all_cores(CoreRange({0, 0}, {0, 0}));

    constexpr uint8_t cb_in = static_cast<uint8_t>(tt::CBIndex::c_0);          // state input tiles
    constexpr uint8_t cb_state_out = static_cast<uint8_t>(tt::CBIndex::c_16);  // new state (fp32)
    constexpr uint8_t cb_o_out = static_cast<uint8_t>(tt::CBIndex::c_17);      // output token (bf16)
    constexpr uint32_t double_buffer = 2;

    const std::string kdir =
        "ttnn/cpp/ttnn/operations/experimental/transformer/gated_delta_prefill_query/device/kernels/";

    ProgramDescriptor program;

    // ---- Reader: streams `state` tiles into cb_in (num_state_tiles for the state
    //      passthrough, then num_o_tiles more for the placeholder O). ----
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(state.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_kernel;
    reader_kernel.kernel_source = kdir + "dataflow/reader_gated_delta_prefill_query.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = all_cores;
    reader_kernel.compile_time_args = std::move(reader_ct_args);
    reader_kernel.config = ReaderConfigDescriptor{};
    reader_kernel.emplace_runtime_args(CoreCoord{0, 0}, {state.buffer(), num_state_tiles, num_o_tiles});

    // ---- Writer: drains cb_state_out -> state_out, then cb_o_out -> o_out. ----
    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(state_out.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(o_out.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = kdir + "dataflow/writer_gated_delta_prefill_query.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = all_cores;
    writer_kernel.compile_time_args = std::move(writer_ct_args);
    writer_kernel.config = WriterConfigDescriptor{};
    writer_kernel.emplace_runtime_args(
        CoreCoord{0, 0}, {state_out.buffer(), o_out.buffer(), num_state_tiles, num_o_tiles});

    // ---- Compute: copies cb_in -> cb_state_out (passthrough) and cb_in -> cb_o_out. ----
    KernelDescriptor compute_kernel;
    compute_kernel.kernel_source = kdir + "compute/gated_delta_prefill_query.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = all_cores;
    compute_kernel.compile_time_args = {};
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};
    compute_kernel.emplace_runtime_args(CoreCoord{0, 0}, {num_state_tiles, num_o_tiles});

    program.kernels.push_back(std::move(reader_kernel));
    program.kernels.push_back(std::move(writer_kernel));
    program.kernels.push_back(std::move(compute_kernel));

    // ---- Circular buffers ----
    program.cbs.push_back(CBDescriptor{
        .total_size = double_buffer * state_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_in, .data_format = state_df, .page_size = state_tile_bytes}}}});

    program.cbs.push_back(CBDescriptor{
        .total_size = double_buffer * state_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_state_out, .data_format = state_df, .page_size = state_tile_bytes}}}});

    program.cbs.push_back(CBDescriptor{
        .total_size = double_buffer * o_tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {
            {CBFormatDescriptor{.buffer_index = cb_o_out, .data_format = o_df, .page_size = o_tile_bytes}}}});

    return program;
}

}  // namespace ttnn::experimental::prim
