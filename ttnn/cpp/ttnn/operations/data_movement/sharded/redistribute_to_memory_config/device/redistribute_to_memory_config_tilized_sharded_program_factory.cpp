// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "redistribute_to_memory_config_tilized_sharded_program_factory.hpp"

#include <cmath>

#include "tt-metalium/buffer_distribution_spec.hpp"
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

RedistributeToMemoryConfigTilizedShardedProgramFactory::cached_program_t
RedistributeToMemoryConfigTilizedShardedProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    tt::tt_metal::Program program{};

    const auto& output_distribution_spec = output.buffer()->buffer_distribution_spec().value();
    const auto num_output_shards = output_distribution_spec.num_shards();

    // Computation of core_grid. TODO: To be replaced by get_optimal_worker_cores_for_sharded_tensor API in PR #40452
    // once that is merged
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

    // End of core grid computation.

    const auto num_cores = ordered_cores_with_data.size();
    const auto& ordered_cores_with_data_range = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));

    // Configuring the CB that store input pages

    const auto input_pages_cb_index = tt::CBIndex::c_0;
    auto output_page_cb_index = input_pages_cb_index;
    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    const auto aligned_input_page_size = input.buffer()->aligned_page_size();
    tt::tt_metal::CircularBufferConfig input_pages_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * aligned_input_page_size, {{input_pages_cb_index, input_cb_data_format}})
            .set_page_size(input_pages_cb_index, aligned_input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, ordered_cores_with_data_range, input_pages_cb_config);

    bool convert_df = input_cb_data_format != output_cb_data_format;

    if (convert_df) {
        // Configuring the CB that stores output pages if we need to convert data formats through the compute kernel
        output_page_cb_index = tt::CBIndex::c_16;
        const auto aligned_output_page_size =
            output.buffer()->aligned_page_size();  // Since we are double buffering, the output page_size must be
                                                   // aligned so the noc_write reads from an aligned address in the CB

        tt::tt_metal::CircularBufferConfig output_pages_cb_config =
            tt::tt_metal::CircularBufferConfig(
                2 * aligned_output_page_size, {{output_page_cb_index, output_cb_data_format}})
                .set_page_size(output_page_cb_index, aligned_output_page_size);
        tt::tt_metal::CreateCircularBuffer(program, ordered_cores_with_data_range, output_pages_cb_config);
    }

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
        "ttnn/cpp/ttnn/operations/data_movement/sharded/redistribute_to_memory_config/device/kernels/dataflow/"
        "to_sharded_pages_tilized_reader.cpp",
        ordered_cores_with_data_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/redistribute_to_memory_config/device/kernels/dataflow/"
        "to_sharded_pages_tilized_writer.cpp",
        ordered_cores_with_data_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    tt::tt_metal::UncompressedBufferPageMapping page_mapping;
    std::vector<CoreCoord> mapped_cores;
    if (convert_df) {
        compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp",
            ordered_cores_with_data_range,
            tt::tt_metal::ComputeConfig{});
        page_mapping = output_distribution_spec.compute_page_mapping();
        mapped_cores = page_mapping.all_cores;
    }

    // Set runtime args
    uint32_t start_shard_id = 0;
    std::vector<CoreCoord> ordered_memory_banks_with_data =
        output_distribution_spec.cores_with_data();  // need this for the maped_cored logic in the compute kernel
                                                     // runtime args block to be valid for DRAM sharded tensors
    for (const auto& core : ordered_cores_with_data) {
        // Reader run-time args
        std::vector<uint32_t> reader_run_time_args = {
            input.buffer()->address(), output.buffer()->address(), start_shard_id};
        std::vector<uint32_t> writer_run_time_args = {output.buffer()->address(), start_shard_id};
        // Set run-time arg
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_run_time_args);

        if (convert_df) {
            auto core_it =
                std::find(mapped_cores.begin(), mapped_cores.end(), ordered_memory_banks_with_data[start_shard_id]);
            uint32_t num_tiles_to_process = 0;

            if (core_it != mapped_cores.end()) {
                const size_t core_idx = std::distance(mapped_cores.begin(), core_it);
                const auto& host_page_indices = page_mapping.core_host_page_indices[core_idx];

                // Iterate through device pages in blocks of num_tiles_per_input_block.
                uint32_t page_offset = 0;
                const uint32_t total_pages = host_page_indices.size();

                while (page_offset < total_pages) {
                    if (host_page_indices[page_offset] != UncompressedBufferPageMapping::PADDING) {
                        num_tiles_to_process++;
                    } else if (page_offset == 0) {  // First page is PADDING means this core has no shards, no need to
                                                    // iterate further. This should never happen, as we are iterating
                                                    // over only cores with data.
                        break;
                    }
                    // Advance num_tiles_per_input_block
                    page_offset++;
                }
            }

            std::vector<uint32_t> compute_run_time_args = {num_tiles_to_process};
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_run_time_args);
        }
        start_shard_id++;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, ordered_cores_with_data}};
}
void RedistributeToMemoryConfigTilizedShardedProgramFactory::override_runtime_arguments(
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
