// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor HCSumReduceProgramFactory::create_descriptor(
    const HcSumReduceParams& operation_attributes, const HcSumReduceInputs& tensor_args, Tensor& output) {
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t LATENT_DIM = TILE_WIDTH;

    auto* input_buffer = tensor_args.input.buffer();
    tt::tt_metal::Buffer* out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto& ashape = tensor_args.input.padded_shape();
    auto num_output_blocks_total = tensor_args.input.padded_shape()[-1] / (TILE_WIDTH * TILE_WIDTH);

    const bool row_major = false;
    auto device_compute_with_storage_grid_size = tensor_args.input.device()->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(
                device_compute_with_storage_grid_size, num_output_blocks_total, row_major);

    TT_ASSERT(tensor_args.input.dtype() == output.dtype(), "Input and output tensors must be of same type");

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);

    const tt::DataFormat intermediary_format =
        (input_format == tt::DataFormat::Float32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

    const uint32_t cb_size = 2;

    const uint32_t input_cb_id = tt::CBIndex::c_0;
    const uint32_t scalar_cb_id = tt::CBIndex::c_2;
    const uint32_t intermed_cb_id0 = tt::CBIndex::c_24;
    const uint32_t intermed_cb_id1 = tt::CBIndex::c_25;
    const uint32_t intermed_cb_id2 = tt::CBIndex::c_26;
    const uint32_t output_cb_id = tt::CBIndex::c_16;

    std::vector<uint32_t> reader_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {
        intermed_cb_id1,
        intermed_cb_id2,
        output_cb_id,
    };
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(writer_compile_time_args);
    std::vector<uint32_t> compute_compile_time_args = {
        input_cb_id,
        scalar_cb_id,
        intermed_cb_id0,
        intermed_cb_id1,
        intermed_cb_id2,
        output_cb_id,
    };

    uint32_t g1_numcores = core_group_1.num_cores();
    std::vector<CoreCoord> cores = grid_to_cores(
        num_cores, device_compute_with_storage_grid_size.x, device_compute_with_storage_grid_size.y, row_major);

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    // Reader writes input tiles to this
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size * input_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_id),
            .data_format = input_format,
            .page_size = input_tile_size}}}});

    // Reader writes scaling tile to this CB. We need it because the reduce LLK requires a scaling factor tile.
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(scalar_cb_id),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    // Compute writes transposed tile (loopback)
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(intermed_cb_id0),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    // Compute writes reduced tile for writer
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(intermed_cb_id1),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    // Writer concats and writes back to compute
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size * intermediary_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(intermed_cb_id2),
            .data_format = intermediary_format,
            .page_size = intermediary_tile_size}}}});

    // Compute transposes and writes back to writer
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size * input_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_id),
            .data_format = input_format,
            .page_size = input_tile_size}}}});

    // Reuse the reader from reduce since we want the same behavior
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/kernels/reader_ssm_1d_sum_reduce.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/kernels/writer_ssm_1d_sum_reduce.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/kernels/ssm_1d_sum_reduce.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = operation_attributes.math_fidelity, .fp32_dest_acc_en = false, .math_approx_mode = false};

    // Build runtime args per core
    reader_kernel_desc.runtime_args.reserve(num_cores);
    writer_kernel_desc.runtime_args.reserve(num_cores);
    compute_kernel_desc.runtime_args.reserve(num_cores);

    uint32_t num_blocks_per_core = 0;
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else {
            num_blocks_per_core = num_blocks_per_core_group_2;
        }

        // (src_addr, num_tiles, start_id, ashape[2]/TILE_HEIGHT, ashape[-1]/TILE_WIDTH)
        reader_kernel_desc.emplace_runtime_args(
            cores[i],
            {input_buffer,
             num_blocks_per_core * LATENT_DIM,
             num_blocks_written * LATENT_DIM,
             ashape[2] / TILE_HEIGHT,
             ashape[-1] / TILE_WIDTH});

        // (dst_addr, num_tiles, start_id, ashape[2]/TILE_HEIGHT, ashape[-1]/(LATENT_DIM*TILE_WIDTH))
        writer_kernel_desc.emplace_runtime_args(
            cores[i],
            {out_buffer,
             num_blocks_per_core,
             num_blocks_written,
             ashape[2] / TILE_HEIGHT,
             ashape[-1] / (LATENT_DIM * TILE_WIDTH)});

        compute_kernel_desc.emplace_runtime_args(cores[i], {num_blocks_per_core, ashape[2] / TILE_HEIGHT});

        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
