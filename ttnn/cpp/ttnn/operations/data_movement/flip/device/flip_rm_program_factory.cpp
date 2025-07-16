// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/flip/device/flip_device_operation.hpp"

namespace ttnn::operations::data_movement {

namespace detail {
static uint32_t num_pages(const ttnn::Tensor& input_tensor) {
    const auto& shape = input_tensor.logical_shape();
    return shape.volume() / shape[-1];
}

static uint32_t page_size(const ttnn::Tensor& input_tensor) {
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

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();
    bool src_is_dram = src_buffer->buffer_type() == BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;

    Program program{};
    IDevice* device = input_tensor.device();

    uint32_t N = operation_attributes.dims.size();
    uint32_t num_input_pages = detail::num_pages(input_tensor);
    uint32_t num_input_pages_to_read = 2;  // how do we know?
    uint32_t num_rows = input_tensor.physical_volume() / input_tensor.logical_shape()[-1];

    uint32_t input_rm_page_size = detail::page_size(input_tensor);
    uint32_t output_rm_page_size = detail::page_size(tensor_return_value);

    uint32_t src0_cb_index = CBIndex::c_0;

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_rows);

    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram, N, input_rm_page_size, num_rows};
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/flip/device/kernels/dataflow/"
        "reader_interleaved_rm.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram, N, output_rm_page_size, num_rows};
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/flip/device/kernels/dataflow/"
        "writer_interleaved_rm.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .core_range = all_cores},
    };
}

void FlipDeviceOperation::MultiCoreRowMajor::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::data_movement
