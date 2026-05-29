// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::data_movement {

namespace detail {
uint32_t num_pages(const ttnn::Tensor& input_tensor) {
    const auto& shape = input_tensor.logical_shape();
    return shape.volume() / shape[-1];
}

uint32_t page_size(const ttnn::Tensor& input_tensor) {
    auto BUFFER_ALIGNMENT = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                ? tt::tt_metal::hal::get_dram_alignment()
                                : tt::tt_metal::hal::get_l1_alignment();
    const auto& shape = input_tensor.logical_shape();  // in anticipation of RM padding
    return tt::round_up(shape[-1] * input_tensor.element_size(), BUFFER_ALIGNMENT);
}

std::vector<uint32_t> get_row_strides(const ttnn::Shape& shape) {
    std::vector<uint32_t> strides(shape.rank());
    strides[shape.rank() - 1] = 1;
    strides[shape.rank() - 2] = 1;
    for (int i = shape.rank() - 3; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

}  // namespace detail

tt::tt_metal::ProgramDescriptor PermuteDeviceOperation::MultiCoreRowInvariant::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    ProgramDescriptor desc;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_rm_page_size = detail::page_size(input_tensor);

    uint32_t output_rm_page_size = detail::page_size(tensor_return_value);

    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_pages_to_read = 2;

    uint32_t num_rows = detail::num_pages(input_tensor);

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_pages_to_read * input_rm_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = input_rm_page_size,
        }}},
    });

    uint32_t N = operation_attributes.dims.size();

    std::vector<uint32_t> reader_compile_time_args = {};
    KernelDescriptor::NamedCompileTimeArgs reader_named_compile_time_args = {
        {"N", N}, {"page_size", input_rm_page_size}, {"num_rows", num_rows}};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "reader_permute_interleaved_rm_row_invariant.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.named_compile_time_args = std::move(reader_named_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {};
    KernelDescriptor::NamedCompileTimeArgs writer_named_compile_time_args = {
        {"N", N}, {"page_size", output_rm_page_size}, {"num_rows", num_rows}};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "writer_permute_interleaved_rm_row_invariant.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.named_compile_time_args = std::move(writer_named_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    auto input_shape_view = input_tensor.logical_shape().view();
    auto output_strides = detail::get_row_strides(output_tensor.logical_shape());  // in anticipation of RM padding

    // Trailing writer args (everything after dst_addr/start/end) are the same for every core; build once.
    std::vector<uint32_t> writer_trailing_args;
    writer_trailing_args.reserve(input_shape_view.size() + operation_attributes.dims.size() + output_strides.size());
    writer_trailing_args.insert(writer_trailing_args.end(), input_shape_view.begin(), input_shape_view.end());
    writer_trailing_args.insert(
        writer_trailing_args.end(), operation_attributes.dims.begin(), operation_attributes.dims.end());
    writer_trailing_args.insert(writer_trailing_args.end(), output_strides.begin(), output_strides.end());

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_row = 0;
    uint32_t num_rows_per_core = 0;
    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_rows_per_core = 0;
        }
        uint32_t end_row = start_row + num_rows_per_core;
        reader_desc.emplace_runtime_args(core, {src_buffer, start_row, end_row});

        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(dst_buffer);
        writer_args.push_back(start_row);
        writer_args.push_back(end_row);
        writer_args.append(writer_trailing_args);
        writer_desc.emplace_runtime_args(core, writer_args);
        start_row = end_row;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

tt::tt_metal::ProgramDescriptor PermuteDeviceOperation::MultiCoreBlockedGeneric::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    ProgramDescriptor desc;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t w_block_size = constants::TILE_WIDTH;
    uint32_t input_cb_page_size = w_block_size * input_tensor.element_size();

    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t x_block_size = constants::TILE_HEIGHT;
    uint32_t output_cb_page_size = x_block_size * input_tensor.element_size();

    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint8_t src1_cb_index = tt::CBIndex::c_2;
    constexpr uint8_t src2_cb_index = tt::CBIndex::c_1;
    uint32_t num_input_pages_to_read = 2;

    // we are focused on reading one row at a time, in a pattern that allows us to write an entire output row at a time
    // if W is being swapped with another dim X (e.g. H), then we need to read X rows at a time (X is the new row
    // dimension) CB is thus X pages in size (X*W*element_size) we read in X input rows of size W, and write out W
    // output rows of size X find the new row dimension (X)

    uint32_t x_dim = operation_attributes.dims.back();
    uint32_t X = input_tensor.logical_shape()[x_dim];
    // stride from one row to the next for each dim in the input tensor
    auto input_strides = detail::get_row_strides(input_tensor.logical_shape());
    uint32_t X_stride = input_strides[x_dim];

    auto output_strides = detail::get_row_strides(output_tensor.logical_shape());
    // after we transpose X and W, we need to stride from one row to the next for each dim in the output tensor
    uint32_t W = input_tensor.logical_shape()[-1];
    uint32_t W_stride = output_strides[x_dim];

    uint32_t N = operation_attributes.dims.size();
    uint32_t num_rows = detail::num_pages(input_tensor);

    // treat the input tensor as 3D with rows * x_blocks * w_blocks
    uint32_t x_blocks = tt::div_up(X, x_block_size);
    uint32_t w_blocks = tt::div_up(W, w_block_size);
    uint32_t num_blocks_total = (num_rows / X) * x_blocks * w_blocks;

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_total);

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_pages_to_read * input_cb_page_size * x_block_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = input_cb_page_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_pages_to_read * output_cb_page_size * w_block_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = cb_data_format,
            .page_size = output_cb_page_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_pages_to_read * x_block_size * w_block_size * input_tensor.element_size(),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src2_cb_index,
            .data_format = cb_data_format,
            .page_size = x_block_size * w_block_size * input_tensor.element_size(),
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {};

    KernelDescriptor::NamedCompileTimeArgs reader_named_compile_time_args = {
        {"N", N},
        {"page_size", input_cb_page_size},
        {"num_rows", num_rows},
        {"x_dim", x_dim},
        {"num_blocks_total", num_blocks_total},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"x_block_size", x_block_size},
        {"w_block_size", w_block_size},
        {"element_size", input_tensor.element_size()},
        {"input_tensor_page_size", static_cast<uint32_t>(src_buffer->aligned_page_size())}};

    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "reader_permute_interleaved_rm_blocked_generic.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.named_compile_time_args = std::move(reader_named_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {};

    KernelDescriptor::NamedCompileTimeArgs writer_named_compile_time_args = {
        {"N", N},
        {"output_page_size", output_cb_page_size},
        {"num_rows", num_rows},
        {"X", X},
        {"X_stride", X_stride},
        {"x_dim", x_dim},
        {"W_stride", W_stride},
        {"input_page_size", input_cb_page_size},
        {"element_size", input_tensor.element_size()},
        {"num_blocks_total", num_blocks_total},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"x_block_size", x_block_size},
        {"w_block_size", w_block_size},
        {"W", W},
        {"output_tensor_page_size", static_cast<uint32_t>(dst_buffer->aligned_page_size())}};

    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "writer_permute_interleaved_rm_blocked_generic.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.named_compile_time_args = std::move(writer_named_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_kernel_args = {x_block_size, w_block_size};
    KernelDescriptor::NamedCompileTimeArgs compute_named_compile_time_args = {
        {"x_block_size", x_block_size}, {"w_block_size", w_block_size}};
    bool fp32_dest_acc_en = cb_data_format_output == tt::DataFormat::Float32 ||
                            cb_data_format_output == tt::DataFormat::Int32 ||
                            cb_data_format_output == tt::DataFormat::UInt32;

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_kernel_args);
    compute_desc.named_compile_time_args = std::move(compute_named_compile_time_args);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    auto input_shape_view = input_tensor.logical_shape().view();

    // Trailing args (everything after src_addr/start/end) are core-invariant; build once.
    std::vector<uint32_t> reader_trailing_args;
    reader_trailing_args.reserve(input_shape_view.size() + input_strides.size());
    reader_trailing_args.insert(reader_trailing_args.end(), input_shape_view.begin(), input_shape_view.end());
    reader_trailing_args.insert(reader_trailing_args.end(), input_strides.begin(), input_strides.end());

    std::vector<uint32_t> writer_trailing_args;
    writer_trailing_args.reserve(input_shape_view.size() + operation_attributes.dims.size() + output_strides.size());
    writer_trailing_args.insert(writer_trailing_args.end(), input_shape_view.begin(), input_shape_view.end());
    writer_trailing_args.insert(
        writer_trailing_args.end(), operation_attributes.dims.begin(), operation_attributes.dims.end());
    writer_trailing_args.insert(writer_trailing_args.end(), output_strides.begin(), output_strides.end());
    auto cores = corerange_to_cores(all_cores, std::nullopt);

    uint32_t start_block = 0;
    uint32_t num_blocks_per_core = 0;
    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());
    compute_desc.runtime_args.reserve(cores.size());
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_blocks_per_core = 0;
        }
        uint32_t end_block = start_block + num_blocks_per_core;

        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(src_buffer);
        reader_args.push_back(start_block);
        reader_args.push_back(end_block);
        reader_args.append(reader_trailing_args);
        reader_desc.emplace_runtime_args(core, reader_args);

        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(dst_buffer);
        writer_args.push_back(start_block);
        writer_args.push_back(end_block);
        writer_args.append(writer_trailing_args);
        writer_desc.emplace_runtime_args(core, writer_args);

        // Compute only consumes num_blocks_per_core in slot 0; the remaining two
        // slots are unused/padding (preserve historical layout).
        compute_desc.emplace_runtime_args(core, {num_blocks_per_core, 0u, 0u});
        start_block = end_block;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::data_movement
