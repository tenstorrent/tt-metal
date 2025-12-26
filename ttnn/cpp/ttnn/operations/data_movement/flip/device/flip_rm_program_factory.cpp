// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/data_movement/flip/device/flip_device_operation.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

static uint32_t get_rm_page_size(const ttnn::Tensor& input_tensor) {
    auto BUFFER_ALIGNMENT = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                ? tt::tt_metal::hal::get_dram_alignment()
                                : tt::tt_metal::hal::get_l1_alignment();
    const auto& shape = input_tensor.logical_shape();  // in anticipation of RM padding
    return tt::round_up(shape[-1] * input_tensor.element_size(), BUFFER_ALIGNMENT);
}

static std::vector<uint32_t> get_row_strides(const ttnn::Shape& shape) {
    std::vector<uint32_t> strides(shape.rank());
    strides[shape.rank() - 1] = 1;
    strides[shape.rank() - 2] = 1;
    for (int i = shape.rank() - 3; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}
}  // namespace detail

FlipDeviceOperation::MultiCoreRowMajor::cached_program_t FlipDeviceOperation::MultiCoreRowMajor::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    Program program{};

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();
    bool src_is_dram = src_buffer->buffer_type() == BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;

    uint32_t rank = input_tensor.logical_shape().rank();
    uint32_t element_size = input_tensor.element_size();
    uint32_t num_rows = input_tensor.physical_volume() / input_tensor.logical_shape()[-1];
    const auto& input_shape = input_tensor.logical_shape();
    std::vector<uint32_t> input_row_strides = detail::get_row_strides(input_shape);

    auto dims = operation_attributes.dims;
    std::vector<uint32_t> dims_to_flip(rank, 0);
    for (const auto& d : dims) {
        dims_to_flip[d] = 1;
    }

    // ------------------------------------------------------------------------
    // 1) Split work to all available cores
    // ------------------------------------------------------------------------
    auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        split_work_to_cores(core_grid, num_rows);

    // log_debug(tt::LogOp, "num_rows: {}\n", num_rows);
    // log_debug(tt::LogOp, "core_grid: {}\n", core_grid);
    // log_debug(tt::LogOp, "num_cores: {}\n", num_cores);
    // log_debug(tt::LogOp, "all_cores: {}\n", all_cores);
    // log_debug(tt::LogOp, "core_group_1: {}\n", core_group_1);
    // log_debug(tt::LogOp, "core_group_2: {}\n", core_group_2);
    // log_debug(tt::LogOp, "num_rows_per_core_group_1: {}\n", num_rows_per_core_group_1);
    // log_debug(tt::LogOp, "num_rows_per_core_group_2: {}\n", num_rows_per_core_group_2);

    // ------------------------------------------------------------------------
    // 2) Create circular buffer
    // ------------------------------------------------------------------------
    DataFormat input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_page_size = detail::get_rm_page_size(input_tensor);
    // uint32_t input_row_width = input_page_size / input_tensor.element_size();
    uint32_t num_input_pages_to_read = 2;  // double buffering
    uint32_t cb_size = num_input_pages_to_read * input_page_size;

    // log_debug(tt::LogOp, "input_row_width: {}\n", input_row_width);

    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(cb_size, {{CBIndex::c_0, input_data_format}})
            .set_page_size(CBIndex::c_0, input_page_size));

    // ------------------------------------------------------------------------
    // 3) Set compile time arguments for kernels
    // ------------------------------------------------------------------------
    std::vector<uint32_t> reader_compile_time_args = {};
    std::unordered_map<std::string, uint32_t> reader_named_compile_time_args = {
        {"src_is_dram", (uint32_t)src_is_dram},
        {"page_size", input_page_size},
        {"rank", rank},
        {"element_size", element_size},
    };
    std::vector<uint32_t> writer_compile_time_args = {};
    std::unordered_map<std::string, uint32_t> writer_named_compile_time_args = {
        {"dst_is_dram", (uint32_t)dst_is_dram},
        {"page_size", input_page_size},
    };

    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // ------------------------------------------------------------------------
    // 4) Create kernels
    // ------------------------------------------------------------------------
    KernelHandle reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/flip/device/kernels/dataflow/"
        "reader_interleaved_rm.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_compile_time_args));

    KernelHandle writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/flip/device/kernels/dataflow/"
        "writer_interleaved_rm.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args, {}, writer_named_compile_time_args));

    // ------------------------------------------------------------------------
    // 5) Set runtime arguments for kernels
    // ------------------------------------------------------------------------
    std::vector<uint32_t> reader_runtime_args = {input_tensor.buffer()->address(), 0, 0};
    std::vector<uint32_t> writer_runtime_args = {output_tensor.buffer()->address(), 0, 0};

    reader_runtime_args.insert(reader_runtime_args.end(), input_shape.cbegin(), input_shape.cend());
    reader_runtime_args.insert(reader_runtime_args.end(), input_row_strides.begin(), input_row_strides.end());
    reader_runtime_args.insert(reader_runtime_args.end(), dims_to_flip.begin(), dims_to_flip.end());

    uint32_t start_row = 0;
    uint32_t end_row = 0;
    auto work_groups = {
        std::make_pair(core_group_1, num_rows_per_core_group_1),
        std::make_pair(core_group_2, num_rows_per_core_group_2)};

    for (const auto& [ranges, rows_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                end_row += rows_per_core;

                reader_runtime_args[1] = start_row;
                reader_runtime_args[2] = end_row;
                SetRuntimeArgs(program, reader_id, core, reader_runtime_args);

                writer_runtime_args[1] = start_row;
                writer_runtime_args[2] = end_row;
                SetRuntimeArgs(program, writer_id, core, writer_runtime_args);

                start_row += rows_per_core;
            }
        }
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_id, .unary_writer_kernel_id = writer_id, .core_range = all_cores},
    };
}

void FlipDeviceOperation::MultiCoreRowMajor::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::data_movement
