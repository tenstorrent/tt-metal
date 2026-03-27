// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "redistribute_to_memory_config_row_major_default_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

RedistributeToMemoryConfigRowMajorDefaultProgramFactory::cached_program_t
RedistributeToMemoryConfigRowMajorDefaultProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    tt::tt_metal::Program program{};

    const auto bytes_per_element = input.element_size();
    const auto elements_per_tensor_row = input.logical_shape()[-1];
    uint32_t num_input_pages_in_row = 1;
    uint32_t num_output_pages_in_row = 1;
    uint32_t elements_per_output_page = output.logical_shape()[-1];
    uint32_t elements_per_input_page = input.logical_shape()[-1];

    if (input.is_sharded() && input.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t input_shard_width =
            (input.shard_spec().has_value() ? input.shard_spec().value().shape[1]
                                            : input.nd_shard_spec().value().shard_shape[-1]);
        num_input_pages_in_row = tt::div_up(elements_per_tensor_row, input_shard_width);
        elements_per_input_page = input_shard_width;
    }
    if (output.is_sharded() && output.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t output_shard_width =
            (output.shard_spec().has_value() ? output.shard_spec().value().shape[1]
                                             : output.nd_shard_spec().value().shard_shape[-1]);
        num_output_pages_in_row = tt::div_up(elements_per_tensor_row, output_shard_width);
        elements_per_output_page = output_shard_width;
    }

    const auto input_pages_cb_index = tt::CBIndex::c_0;
    const auto output_page_cb_index = tt::CBIndex::c_1;

    // Computation of core_grid
    auto* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    const uint32_t total_logical_rows = input.logical_volume() / input.logical_shape()[-1];
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_logical_rows);
    std::vector<CoreCoord> ordered_cores = corerange_to_cores(all_cores, num_cores, true);

    constexpr uint32_t MAX_SUBBLOCK_SIZE_BYTES = 65536;  // Chosen empirically to prevent large row OOM CB error
    const auto input_page_size = std::min(static_cast<uint32_t>(input.buffer()->page_size()), MAX_SUBBLOCK_SIZE_BYTES);
    const auto aligned_output_page_size = std::min(
        static_cast<uint32_t>(output.buffer()->aligned_page_size()),
        MAX_SUBBLOCK_SIZE_BYTES);  // Since we are double buffering, the output page_size must be aligned so
                                   // the noc_write reads from an aligned address in the CB

    // Configuring the CB that store input pages
    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::tt_metal::CircularBufferConfig input_pages_cb_config =
        tt::tt_metal::CircularBufferConfig(input_page_size, {{input_pages_cb_index, input_cb_data_format}})
            .set_page_size(input_pages_cb_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, input_pages_cb_config);

    // Configuring the CB that stores output pages
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::tt_metal::CircularBufferConfig output_pages_cb_config =
        tt::tt_metal::CircularBufferConfig(
            2 * aligned_output_page_size, {{output_page_cb_index, output_cb_data_format}})
            .set_page_size(output_page_cb_index, aligned_output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, output_pages_cb_config);

    uint32_t input_subblock_size_bytes = std::min(elements_per_input_page * bytes_per_element, MAX_SUBBLOCK_SIZE_BYTES);
    uint32_t output_subblock_size_bytes =
        std::min(elements_per_output_page * bytes_per_element, MAX_SUBBLOCK_SIZE_BYTES);
    // Reader kernel config with compile-time args
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(input_pages_cb_index),
        static_cast<uint32_t>(output_page_cb_index),
        static_cast<uint32_t>(num_output_pages_in_row),
        static_cast<uint32_t>(num_input_pages_in_row),
        static_cast<uint32_t>(elements_per_output_page),
        static_cast<uint32_t>(bytes_per_element),
        static_cast<uint32_t>(elements_per_input_page),
        static_cast<uint32_t>(elements_per_tensor_row),
        input_subblock_size_bytes,
        output_subblock_size_bytes,

    };
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);

    // Writer kernel config with compile-time args
    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(output_page_cb_index),
        static_cast<uint32_t>(num_output_pages_in_row),
        static_cast<uint32_t>(elements_per_output_page),
        static_cast<uint32_t>(bytes_per_element),
        static_cast<uint32_t>(elements_per_tensor_row),
        output_subblock_size_bytes,
    };

    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/redistribute_to_memory_config/device/kernels/dataflow/"
        "redistribute_pages_row_major_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/redistribute_to_memory_config/device/kernels/dataflow/"
        "redistribute_pages_row_major_writer.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Set runtime args
    uint32_t start_row_id = 0;
    for (const auto& core : ordered_cores) {
        // Reader run-time args
        uint32_t num_rows_to_process = num_pages_per_core_group_1;
        if (core_group_2.contains(core)) {
            num_rows_to_process = num_pages_per_core_group_2;
        }
        std::vector<uint32_t> reader_run_time_args = {input.buffer()->address(), start_row_id, num_rows_to_process};
        std::vector<uint32_t> writer_run_time_args = {output.buffer()->address(), start_row_id, num_rows_to_process};

        start_row_id++;
        // Set run-time arg
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_run_time_args);
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, ordered_cores}};
}
void RedistributeToMemoryConfigRowMajorDefaultProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& runtime_args_by_core_reader = GetRuntimeArgs(program, reader_kernel_id);
    auto& runtime_args_by_core_writer = GetRuntimeArgs(program, writer_kernel_id);
    const auto& cores = cached_program.shared_variables.cores;
    const auto& input_buffer = tensor_args.input_tensor.buffer();
    const auto& output_buffer = output_tensor.buffer();
    for (const auto& core : cores) {
        auto& runtime_args_reader = runtime_args_by_core_reader[core.x][core.y];
        runtime_args_reader[0] = input_buffer->address();
        auto& runtime_args_writer = runtime_args_by_core_writer[core.x][core.y];
        runtime_args_writer[0] = output_buffer->address();
    }
}

}  // namespace ttnn::prim
