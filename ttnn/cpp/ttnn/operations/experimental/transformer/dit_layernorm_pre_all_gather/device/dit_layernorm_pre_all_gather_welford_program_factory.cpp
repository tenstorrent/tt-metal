// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_layernorm_pre_all_gather_device_operation.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor PreAllGatherWelfordProgramFactory::create_descriptor(
    const DitLayernormPreAllGatherParams& operation_attributes,
    const DitLayernormPreAllGatherInputs& tensor_args,
    Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;

    IDevice* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t num_tile_rows = NC * Ht;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    uint32_t block_size = get_dest_reg_count(operation_attributes.compute_kernel_config);
    uint32_t output_tiles_per_row = 2;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    constexpr uint32_t double_buffer = 2;
    const uint32_t in0_tiles = block_size * double_buffer;
    const uint32_t out0_tiles = output_tiles_per_row * double_buffer;

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    std::vector<uint32_t> reader_compile_time_args = {
        block_size,
    };
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {output_tiles_per_row};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_args = {Wt, W, block_size};

    const auto* compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/device/kernels/compute/"
        "layernorm_pre_allgather_welford.cpp";

    const auto& recip_tensor = tensor_args.recip_tensor;
    const uint32_t reciprocal_CB_size_bytes = recip_tensor.buffer()->aligned_size_per_bank();
    constexpr tt::DataFormat reciprocal_cb_data_format = tt::DataFormat::Float32;

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor program_descriptor;

    // Reader kernel
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/device/kernels/dataflow/"
        "reader_layernorm_preallgather_dit.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/device/kernels/dataflow/"
        "writer_layernorm_preallgather_dit.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};

    // Compute kernel
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source = compute_kernel_file;
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = std::move(compute_args);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};

    // Build runtime args per core
    reader_kernel_desc.runtime_args.reserve(num_cores);
    writer_kernel_desc.runtime_args.reserve(num_cores);
    compute_kernel_desc.runtime_args.reserve(num_cores);

    uint32_t curr_row = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t in_tile_offset = curr_row * Wt;
        uint32_t out_tile_offset = curr_row * output_tiles_per_row;

        reader_kernel_desc.emplace_runtime_args(core, {a.buffer(), num_tile_rows_per_core, Wt, in_tile_offset});
        compute_kernel_desc.emplace_runtime_args(core, {num_tile_rows_per_core});
        writer_kernel_desc.emplace_runtime_args(core, {output.buffer(), num_tile_rows_per_core, out_tile_offset});

        curr_row += num_tile_rows_per_core;
    }

    program_descriptor.kernels.push_back(std::move(reader_kernel_desc));
    program_descriptor.kernels.push_back(std::move(writer_kernel_desc));
    program_descriptor.kernels.push_back(std::move(compute_kernel_desc));

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    // c_in0 -> a
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = in0_tiles * in_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in_data_format,
            .page_size = in_single_tile_size}}}});

    // c_intermed0 -> x^2 (CB 1)
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = in0_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = cb_data_format,
            .page_size = single_tile_size}}}});

    // Output (CB 14)
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = out0_tiles * out_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_14),
            .data_format = out_data_format,
            .page_size = out_single_tile_size}}}});

    // Reciprocal LUT (CB 2) - sharded, with backing buffer
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = reciprocal_CB_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
            .data_format = reciprocal_cb_data_format,
            .page_size = reciprocal_CB_size_bytes}}},
        .buffer = recip_tensor.buffer()});

    return program_descriptor;
}

}  // namespace ttnn::experimental::prim
