// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "iterative_topk_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk {

IterativeTopkDeviceOperation::TiledProgramFactory::cached_program_t
IterativeTopkDeviceOperation::TiledProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    auto& output_values = tensor_return_value[0];
    auto& output_indices = tensor_return_value[1];

    auto* device = input.device();
    TT_FATAL(device != nullptr, "Device must be non-null");
    tt::tt_metal::Program program{};

    uint32_t width = input.logical_shape()[-1];
    uint32_t num_rows = input.logical_shape().volume() / width;
    uint32_t k = operation_attributes.k;
    uint32_t width_tiles = width / 32;
    uint32_t height_tiles = num_rows / 32;

    uint32_t input_tile_page_size = input.buffer()->aligned_page_size();
    uint32_t output_values_page_size = output_values.buffer()->aligned_page_size();
    uint32_t output_indices_page_size = output_indices.buffer()->aligned_page_size();

    auto grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_group_1, tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, height_tiles);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_out_values = tt::CBIndex::c_1;
    constexpr auto cb_out_indices = tt::CBIndex::c_2;

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    auto values_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_values.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_indices.dtype());

    // Input CB holds width_tiles pages (one full row of tiles = 32 logical rows)
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(width_tiles * input_tile_page_size, {{cb_input, input_data_format}})
            .set_page_size(cb_input, input_tile_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_input_config);

    // Output values CB: double-buffered, one row-major page per row
    tt::tt_metal::CircularBufferConfig cb_out_values_config =
        tt::tt_metal::CircularBufferConfig(2 * output_values_page_size, {{cb_out_values, values_data_format}})
            .set_page_size(cb_out_values, output_values_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_values_config);

    // Output indices CB: double-buffered, one row-major page per row
    tt::tt_metal::CircularBufferConfig cb_out_indices_config =
        tt::tt_metal::CircularBufferConfig(2 * output_indices_page_size, {{cb_out_indices, indices_data_format}})
            .set_page_size(cb_out_indices, output_indices_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_indices_config);

    // Reader kernel compile-time args
    std::vector<uint32_t> reader_compile_time_args = {
        cb_input,
        input_tile_page_size,
        width_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/iterative_topk/device/kernels/dataflow/"
        "reader_iterative_topk_tiled.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer kernel compile-time args
    std::vector<uint32_t> writer_compile_time_args = {
        cb_input,
        cb_out_values,
        cb_out_indices,
        width,
        width_tiles,
        k,
        input_tile_page_size,
        output_values_page_size,
        output_indices_page_size,
    };
    tt::tt_metal::TensorAccessorArgs(output_values.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_indices.buffer()).append_to(writer_compile_time_args);

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/iterative_topk/device/kernels/dataflow/"
        "writer_iterative_topk_tiled.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Set runtime args per core
    std::vector<uint32_t> reader_runtime_args = {input.buffer()->address(), 0, 0};
    std::vector<uint32_t> writer_runtime_args = {
        output_values.buffer()->address(), output_indices.buffer()->address(), 0, 0, num_rows};

    uint32_t start_ht = 0;
    uint32_t end_ht = 0;
    auto cores = corerange_to_cores(all_cores, std::nullopt);
    for (const auto& core : cores) {
        uint32_t workload_per_core = 0;
        if (core_group_1.contains(core)) {
            workload_per_core = tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            workload_per_core = tiles_per_core_group_2;
        }
        start_ht = end_ht;
        end_ht = start_ht + workload_per_core;

        reader_runtime_args[1] = start_ht;
        reader_runtime_args[2] = end_ht;

        writer_runtime_args[2] = start_ht;
        writer_runtime_args[3] = end_ht;

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, cores}};
}

void IterativeTopkDeviceOperation::TiledProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores;
    for (const auto& core : cores) {
        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
        reader_runtime_args[0] = tensor_args.input.buffer()->address();
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        writer_runtime_args[0] = tensor_return_value[0].buffer()->address();
        writer_runtime_args[1] = tensor_return_value[1].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk
