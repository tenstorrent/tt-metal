// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "noc/noc_parameters.h"  // DRAM_ALIGNMENT

namespace ttnn::operations::data_movement {

namespace detail {
uint32_t num_pages(const ttnn::Tensor& input_tensor) {
    const auto& shape = input_tensor.get_logical_shape();
    return shape.volume() / shape[-1];
}

uint32_t page_size(const ttnn::Tensor& input_tensor) {
    auto BUFFER_ALIGNMENT =
        input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? DRAM_ALIGNMENT : L1_ALIGNMENT;
    const auto& shape = input_tensor.get_logical_shape();  // in anticipation of RM padding
    return tt::round_up(shape[-1] * input_tensor.element_size(), BUFFER_ALIGNMENT);
}

std::vector<uint32_t> get_row_strides(const ttnn::SimpleShape& shape) {
    std::vector<uint32_t> strides(shape.rank());
    strides[shape.rank() - 1] = 1;
    strides[shape.rank() - 2] = 1;
    for (int i = shape.rank() - 3; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

}  // namespace detail

PermuteDeviceOperation::MultiCoreRowInvariant::cached_program_t PermuteDeviceOperation::MultiCoreRowInvariant::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_rm_page_size = detail::page_size(input_tensor);

    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t output_rm_page_size = detail::page_size(tensor_return_value);

    uint32_t num_input_pages = detail::num_pages(input_tensor);

    tt::tt_metal::Device* device = input_tensor.device();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_pages_to_read = 2;

    uint32_t num_rows = input_tensor.volume() / input_tensor.get_logical_shape()[-1];

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_pages_to_read * input_rm_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, input_rm_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t N = operation_attributes.dims.size();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram, N, input_rm_page_size, num_rows};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "reader_permute_interleaved_rm_row_invariant.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)dst_is_dram, N, output_rm_page_size, num_rows};
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "writer_permute_interleaved_rm_row_invariant.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), 0, 0};

    auto input_shape_view = input_tensor.get_logical_shape().view();
    auto output_strides = detail::get_row_strides(output_tensor.get_logical_shape());  // in anticipation of RM padding

    std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), 0, 0};
    writer_runtime_args.insert(writer_runtime_args.end(), input_shape_view.begin(), input_shape_view.end());
    writer_runtime_args.insert(
        writer_runtime_args.end(), operation_attributes.dims.begin(), operation_attributes.dims.end());
    writer_runtime_args.insert(writer_runtime_args.end(), output_strides.begin(), output_strides.end());

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_row = 0;
    uint32_t num_rows_per_core = 0;
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
        reader_runtime_args[1] = start_row;
        reader_runtime_args[2] = end_row;
        writer_runtime_args[1] = start_row;
        writer_runtime_args[2] = end_row;
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        start_row = end_row;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id, .unary_writer_kernel_id = unary_writer_kernel_id}};
}

void PermuteDeviceOperation::MultiCoreRowInvariant::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();
    auto& all_cores = cached_program.shared_variables.core_range;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    for (const auto& core : cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
        auto& runtime_args_writer = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        runtime_args_writer[0] = dst_buffer->address();
    }
}

PermuteDeviceOperation::MultiCoreBlockedGeneric::cached_program_t
PermuteDeviceOperation::MultiCoreBlockedGeneric::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t w_block_size = constants::TILE_WIDTH;
    uint32_t input_cb_page_size = w_block_size * input_tensor.element_size();

    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t x_block_size = constants::TILE_HEIGHT;
    uint32_t output_cb_page_size = x_block_size * input_tensor.element_size();

    tt::tt_metal::Device* device = input_tensor.device();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_2;
    uint32_t src2_cb_index = tt::CBIndex::c_1;
    uint32_t num_input_pages_to_read = 2;

    // we are focused on reading one row at a time, in a pattern that allows us to write an entire output row at a time
    // if W is being swapped with another dim X (e.g. H), then we need to read X rows at a time (X is the new row
    // dimension) CB is thus X pages in size (X*W*element_size) we read in X input rows of size W, and write out W
    // output rows of size X find the new row dimension (X)

    uint32_t x_dim = operation_attributes.dims.back();
    uint32_t X = input_tensor.get_logical_shape()[x_dim];
    // stride from one row to the next for each dim in the input tensor
    auto input_strides = detail::get_row_strides(input_tensor.get_logical_shape());
    uint32_t X_stride = input_strides[x_dim];

    auto output_strides = detail::get_row_strides(output_tensor.get_logical_shape());
    // after we transpose X and W, we need to stride from one row to the next for each dim in the output tensor
    uint32_t W = input_tensor.get_logical_shape()[-1];
    uint32_t W_stride = output_strides[x_dim];

    uint32_t N = operation_attributes.dims.size();
    uint32_t num_rows = input_tensor.volume() / input_tensor.get_logical_shape()[-1];

    // treat the input tensor as 3D with rows * x_blocks * w_blocks
    uint32_t x_blocks = tt::div_up(X, x_block_size);
    uint32_t w_blocks = tt::div_up(W, w_block_size);
    uint32_t num_blocks_total = (num_rows / X) * x_blocks * w_blocks;

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_total);

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_pages_to_read * input_cb_page_size * x_block_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, input_cb_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_pages_to_read * output_cb_page_size * w_block_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, output_cb_page_size);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    tt::tt_metal::CircularBufferConfig cb_src2_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_pages_to_read * x_block_size * w_block_size * input_tensor.element_size(),
            {{src2_cb_index, cb_data_format}})
            .set_page_size(src2_cb_index, x_block_size * w_block_size * input_tensor.element_size());
    auto cb_src2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src_is_dram,
        N,
        input_cb_page_size,
        num_rows,
        x_dim,
        num_blocks_total,
        x_blocks,
        w_blocks,
        x_block_size,
        w_block_size,
        input_tensor.element_size(),
        input_tensor.get_logical_shape()[-1] * input_tensor.element_size()};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "reader_permute_interleaved_rm_blocked_generic.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)dst_is_dram,
        N,
        output_cb_page_size,
        num_rows,

        X,
        X_stride,
        x_dim,

        W_stride,
        input_cb_page_size,
        input_tensor.element_size(),

        num_blocks_total,
        x_blocks,
        w_blocks,
        x_block_size,
        w_block_size,

        W,
        output_tensor.get_logical_shape()[-1] * output_tensor.element_size()};
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "writer_permute_interleaved_rm_blocked_generic.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {x_block_size, w_block_size};
    bool fp32_dest_acc_en = cb_data_format_output == tt::DataFormat::Float32;
    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args,
        });

    auto input_shape_view = input_tensor.get_logical_shape().view();

    std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), 0, 0};
    reader_runtime_args.insert(reader_runtime_args.end(), input_shape_view.begin(), input_shape_view.end());
    reader_runtime_args.insert(reader_runtime_args.end(), input_strides.begin(), input_strides.end());

    std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), 0, 0};

    writer_runtime_args.insert(writer_runtime_args.end(), input_shape_view.begin(), input_shape_view.end());
    writer_runtime_args.insert(
        writer_runtime_args.end(), operation_attributes.dims.begin(), operation_attributes.dims.end());
    writer_runtime_args.insert(writer_runtime_args.end(), output_strides.begin(), output_strides.end());
    auto cores = corerange_to_cores(all_cores, std::nullopt);

    std::vector<uint32_t> compute_runtime_args = {dst_buffer->address(), 0, 0};

    uint32_t start_block = 0;
    uint32_t num_blocks_per_core = 0;
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_blocks_per_core = 0;
        }
        compute_runtime_args[0] = num_blocks_per_core;
        uint32_t end_block = start_block + num_blocks_per_core;
        reader_runtime_args[1] = start_block;
        reader_runtime_args[2] = end_block;
        writer_runtime_args[1] = start_block;
        writer_runtime_args[2] = end_block;
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
        start_block = end_block;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .core_range = all_cores}};
}

void PermuteDeviceOperation::MultiCoreBlockedGeneric::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();
    auto& all_cores = cached_program.shared_variables.core_range;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    for (const auto& core : cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
        auto& runtime_args_writer = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        runtime_args_writer[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::data_movement
