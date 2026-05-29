// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

using namespace tt::constants;

tt::tt_metal::ProgramDescriptor PrefixScanProgramFactory::create_descriptor(
    const PrefixScanParams& operation_attributes, const PrefixScanInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.a;
    const auto& bx = tensor_args.bx;
    const auto& h_prev = tensor_args.h_prev;
    auto& output = tensor_return_value;

    Buffer* a_buffer = a.buffer();
    Buffer* bx_buffer = bx.buffer();
    Buffer* h_buffer = h_prev.buffer();
    Buffer* output_buffer = output.buffer();
    TT_FATAL(a_buffer != nullptr, "Input a buffer should be allocated on device");
    TT_FATAL(bx_buffer != nullptr, "Input bx buffer should be allocated on device");
    TT_FATAL(h_buffer != nullptr, "Input h_prev buffer should be allocated on device");
    TT_FATAL(output_buffer != nullptr, "Output buffer should be allocated on device");

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_row_size = tt::datum_size(intermediary_format) * TILE_WIDTH;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

    const auto all_cores = a.shard_spec()->grid;

    const uint32_t sharded_sequence_length = a.shard_spec()->shape[0];
    const uint32_t sharded_hidden_state_length = a.shard_spec()->shape[1];

    const uint32_t total_tiles_per_row = sharded_hidden_state_length / TILE_HEIGHT;
    const uint32_t total_tiles_per_col = sharded_sequence_length / TILE_HEIGHT;
    const uint32_t total_tiles = total_tiles_per_row * total_tiles_per_col;

    // One chunk is a row of 32 tiles where an untilize call will move each row into a separate tile
    constexpr uint32_t num_tiles_in_chunk = 32;
    const uint32_t num_chunks_per_row = tt::div_up(total_tiles_per_row, num_tiles_in_chunk);

    // Create all non-shard-backed CBs BEFORE shard-backed CBs to keep them at low L1 addresses.
    // The CB allocator tracks the sequential allocation pointer; shard-backed CBs at high buffer
    // addresses would push the pointer up, placing subsequent non-shard-backed CBs near the L1 limit.

    // Non-shard-backed staging CB shared between a and bx inputs.
    // The untilize operation always reads 32 tiles. When total_tiles_per_row is not a multiple of 32,
    // the last chunk has fewer tiles in the shard, causing OOB reads on a shard-backed CB.
    // Using a staging CB with exactly 32 tiles ensures the untilize reads stay within allocated memory.
    // The reader kernel copies data from the shard to this staging CB chunk by chunk via NOC read,
    // alternating between a and bx data. The compute kernel consumes each in sequence.
    const uint32_t cb_a_in_id = tt::CBIndex::c_0;
    const uint32_t cb_bx_in_id = cb_a_in_id;  // shared staging CB

    const uint32_t num_tiles_in_row_to_tile_cb = 32;  // Tilizing 32 tiles will pack tensor rows into separate tiles
    const uint32_t cb_a_tilize_in_id = tt::CBIndex::c_24;
    const uint32_t cb_bx_tilize_in_id = tt::CBIndex::c_25;
    const uint32_t cb_tilize_out_id = tt::CBIndex::c_26;
    const uint32_t cb_h_prev_id = tt::CBIndex::c_27;
    const uint32_t cb_ah_id = tt::CBIndex::c_28;
    const uint32_t cb_h_id = tt::CBIndex::c_29;
    const uint32_t cb_h_acc_id = tt::CBIndex::c_31;

    // Non-shard-backed staging CB for h_in. The h_prev shard uses intermediary_row_size pages
    // (64 bytes) but Float16_b format. The unpacker reads full-format tiles (~2KB) which overshoot
    // the 64-byte page boundary. If the shard is near the L1 limit this causes OOB.
    // Using a non-shard-backed CB at a low L1 address avoids this.
    const uint32_t cb_h_in_id = tt::CBIndex::c_2;

    // Shard-backed output CB last - buffer address may be high in L1
    const uint32_t cb_out_id = tt::CBIndex::c_16;

    std::vector<uint32_t> reader_compile_time_args = {
        cb_a_in_id, cb_bx_in_id, cb_h_in_id, input_tile_size, intermediary_row_size};
    std::vector<uint32_t> writer_compile_time_args = {cb_out_id, cb_h_acc_id};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    std::vector<uint32_t> compute_compile_time_args = {
        cb_a_in_id,
        cb_bx_in_id,
        cb_h_in_id,
        cb_a_tilize_in_id,
        cb_bx_tilize_in_id,
        cb_h_prev_id,
        cb_ah_id,
        cb_h_id,
        cb_tilize_out_id,
        cb_out_id,
        cb_h_acc_id};

    auto device_compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    std::vector<CoreCoord> cores = grid_to_cores(
        all_cores.num_cores(), device_compute_with_storage_grid_size.x, device_compute_with_storage_grid_size.y, true);

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    // Non-shard-backed staging CB shared between a and bx inputs (see above).
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_in_chunk * input_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_a_in_id),
            .data_format = input_format,
            .page_size = input_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_in_row_to_tile_cb * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_a_tilize_in_id),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_in_row_to_tile_cb * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_bx_tilize_in_id),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_in_row_to_tile_cb * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_tilize_out_id),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_h_prev_id),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_ah_id),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_h_id),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_chunks_per_row * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_h_acc_id),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    // Non-shard-backed staging CB for h_in (see above).
    desc.cbs.push_back(CBDescriptor{
        .total_size = total_tiles_per_row * intermediary_row_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_h_in_id),
            .data_format = intermediary_format,
            .page_size = intermediary_row_size}}}});

    // Shard-backed output CB last - buffer address may be high in L1.
    desc.cbs.push_back(CBDescriptor{
        .total_size = total_tiles * input_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_out_id),
            .data_format = input_format,
            .page_size = input_tile_size}}},
        .buffer = output_buffer});

    // Build kernels with per-core runtime args.
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/kernels/reader_ssm_prefix_scan.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};
    reader_kernel_desc.runtime_args.reserve(cores.size());

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/kernels/writer_ssm_prefix_scan.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};
    writer_kernel_desc.runtime_args.reserve(cores.size());

    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/kernels/ssm_prefix_scan.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = operation_attributes.math_fidelity, .fp32_dest_acc_en = false, .math_approx_mode = false};
    compute_kernel_desc.runtime_args.reserve(cores.size());

    for (const auto& core : cores) {
        // (total_tiles_per_row, total_tiles_per_col, a_shard_addr, bx_shard_addr, h_shard_addr)
        reader_kernel_desc.emplace_runtime_args(
            core, {total_tiles_per_row, total_tiles_per_col, a_buffer, bx_buffer, h_buffer});

        // (num_tiles_per_core, hidden_state_len, h_shard_addr)
        writer_kernel_desc.emplace_runtime_args(core, {total_tiles, sharded_hidden_state_length, h_buffer});

        // (total_tiles, total_tiles_per_row, total_tiles_per_col, num_chunks_per_row)
        compute_kernel_desc.emplace_runtime_args(
            core, {total_tiles, total_tiles_per_row, total_tiles_per_col, num_chunks_per_row});
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
