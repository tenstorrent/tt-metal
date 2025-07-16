// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/flip/device/flip_device_operation.hpp"

namespace ttnn::operations::data_movement {

FlipDeviceOperation::MultiCoreTiled::cached_program_t FlipDeviceOperation::MultiCoreTiled::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();
    uint32_t rank = operation_attributes.dims.size();

    Program program{};

    // uint32_t input_page_size = detail::tile_size(input_tensor);
    // uint32_t num_tiles = detail::num_tiles(tensor_return_value);
    uint32_t input_page_size = 8;
    uint32_t num_tiles = 1;
    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    bool src_is_dram = src_buffer->buffer_type() == BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram, rank, input_page_size, num_tiles};
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/flip/device/kernels/dataflow/"
        "reader_interleaved_tiled.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/flip/device/kernels/dataflow/"
        "writer_interleaved_tiled.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    log_debug(tt::LogOp, "input_tensor.get_strides(): {}", input_tensor.strides());

    auto lol = ttnn::Shape(tt::tt_metal::compute_strides(input_tensor.padded_shape()));
    log_debug(tt::LogOp, "lol: {}", lol);

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .core_range = all_cores},
    };
}

void FlipDeviceOperation::MultiCoreTiled::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::data_movement
