// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_sharded_tilized_sharded_program_factory.hpp"

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

ToShardedTilizedProgramFactory::cached_program_t ToShardedTilizedProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    tt::tt_metal::Program program{};

    const auto& output_distribution_spec = output.buffer()->buffer_distribution_spec().value();
    const auto num_output_shards = output_distribution_spec.num_shards();
    // const auto bytes_per_element = input.element_size();
    // const auto elements_per_tensor_row = input.logical_shape()[-1];
    // uint32_t num_input_pages_in_row = 1;
    // uint32_t num_output_pages_in_row = 1;
    // uint32_t elements_per_output_page = output.logical_shape()[-1];
    // uint32_t elements_per_input_page = input.logical_shape()[-1];

    // if (input.is_sharded() && input.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
    //     uint32_t input_shard_width =
    //         (input.shard_spec().has_value() ? input.shard_spec().value().shape[1]
    //                                         : input.nd_shard_spec().value().shard_shape[-1]);
    //     num_input_pages_in_row = tt::div_up(elements_per_tensor_row, input_shard_width);
    //     elements_per_input_page = input_shard_width;
    // }
    // if (output.is_sharded() && output.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
    //     uint32_t output_shard_width =
    //         (output.shard_spec().has_value() ? output.shard_spec().value().shape[1]
    //                                          : output.nd_shard_spec().value().shard_shape[-1]);
    //     num_output_pages_in_row = tt::div_up(elements_per_tensor_row, output_shard_width);
    //     elements_per_output_page = output_shard_width;
    // }

    const auto input_pages_cb_index = tt::CBIndex::c_0;
    const auto output_page_cb_index = tt::CBIndex::c_16;

    // computation of core_grid. To be replaced by get_optimal_worker_cores_for_sharded_tensor API in PR #40452 once
    // that is merged
    std::vector<CoreCoord> ordered_cores_with_data;

    if (!output.memory_config().is_dram()) {
        ordered_cores_with_data = output_distribution_spec.cores_with_data();
    } else {
        TT_FATAL(output.device() != nullptr, "Device pointer must be valid when selecting optimal DRAM worker cores");
        auto all_dram_workers =
            output.device()->get_optimal_dram_bank_to_logical_worker_assignment(NOC::RISCV_0_default);
        const auto dram_banks = output_distribution_spec.cores_with_data();
        ordered_cores_with_data.reserve(dram_banks.size());
        for (const auto& dram_core : dram_banks) {
            const uint32_t dram_channel = output.device()->dram_channel_from_logical_core(dram_core);
            ordered_cores_with_data.push_back(all_dram_workers[dram_channel]);
        }
    }

    // end of core grid computation. Can add option to use custom core grid too (like default path to accomodate
    // subdevices op)

    const auto num_cores = ordered_cores_with_data.size();
    const auto& ordered_cores_with_data_range = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));

    // Configuring the CB that store input pages
    const auto aligned_input_page_size = input.buffer()->aligned_page_size();
    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::tt_metal::CircularBufferConfig input_pages_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * aligned_input_page_size, {{input_pages_cb_index, input_cb_data_format}})
            .set_page_size(input_pages_cb_index, aligned_input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, ordered_cores_with_data_range, input_pages_cb_config);

    // Configuring the CB that stores output pages
    const auto aligned_output_page_size =
        output.buffer()->aligned_page_size();  // Since we are double buffering, the output page_size must be aligned so
                                               // the noc_write reads from an aligned address in the CB
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::tt_metal::CircularBufferConfig output_pages_cb_config =
        tt::tt_metal::CircularBufferConfig(
            2 * aligned_output_page_size, {{output_page_cb_index, output_cb_data_format}})
            .set_page_size(output_page_cb_index, aligned_output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, ordered_cores_with_data_range, output_pages_cb_config);

    // Reader kernel config with compile-time args
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(input_pages_cb_index),
        static_cast<uint32_t>(num_output_shards),
        static_cast<uint32_t>(num_cores)};

    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(reader_compile_time_args);

    // Writer kernel config with compile-time args
    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(output_page_cb_index),
        static_cast<uint32_t>(num_output_shards),
        static_cast<uint32_t>(num_cores)};

    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/to_sharded_pages_row_major_reader.cpp",
        ordered_cores_with_data_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/to_sharded_pages_row_major_writer.cpp",
        ordered_cores_with_data_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // bool convert_dt = input_cb_data_format != output_cb_data_format;
    // tt::tt_metal::KernelHandle compute_kernel_id = 0;
    // if (convert_dt) {
    //     compute_kernel_id = tt::tt_metal::CreateKernel(
    //         program,
    //         "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp",
    //         ordered_cores_with_data_range,
    //         tt::tt_metal::ComputeConfig{});
    //         const auto page_mapping = output_distribution_spec.compute_page_mapping();

    // }
    // Set runtime args
    uint32_t start_shard_id = 0;
    for (const auto& core : ordered_cores_with_data) {
        // Reader run-time args
        std::vector<uint32_t> reader_run_time_args = {
            input.buffer()->address(), output.buffer()->address(), start_shard_id};
        std::vector<uint32_t> writer_run_time_args = {output.buffer()->address(), start_shard_id};
        // Set run-time arg
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_run_time_args);

        // if (convert_dt) {

        // }
        start_shard_id++;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, ordered_cores_with_data}};
}
void ToShardedTilizedProgramFactory::override_runtime_arguments(
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
        runtime_args_reader[1] = output_buffer->address();
        auto& runtime_args_writer = runtime_args_by_core_writer[core.x][core.y];
        runtime_args_writer[0] = output_buffer->address();
    }
}

}  // namespace ttnn::prim
