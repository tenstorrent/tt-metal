// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_to_memory_config_tilized_default_program_factory.hpp"

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

CopyToMemoryConfigTilizedDefaultProgramFactory::cached_program_t CopyToMemoryConfigTilizedDefaultProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    tt::tt_metal::Program program{};

    // Computation of core_grid
    auto* device = input.device();
    auto compute_with_storage_grid_size =
        device->compute_with_storage_grid_size();  // This can be replaced with get_worker_cores in subdevices

    const auto& logical_shape = input.logical_shape();
    const auto& tile = input.tensor_spec().tile();
    const uint32_t tile_height = tile.get_height();
    const uint32_t tile_width = tile.get_width();
    const uint32_t rank = logical_shape.rank();
    uint32_t total_tiles = 1;
    if (rank >= 1) {
        total_tiles = tt::div_up(logical_shape[-1], tile_width);
    }
    if (rank >= 2) {
        total_tiles *= tt::div_up(logical_shape[-2], tile_height);
    }
    for (uint32_t i = 0; i + 2 < rank; ++i) {
        total_tiles *= logical_shape[i];
    }
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_tiles);
    std::vector<CoreCoord> ordered_cores = corerange_to_cores(all_cores, num_cores, true);

    // Configuring the CB that store input pages

    const auto input_pages_cb_index = tt::CBIndex::c_0;
    auto output_page_cb_index = input_pages_cb_index;
    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    const auto aligned_input_page_size = input.buffer()->aligned_page_size();
    tt::tt_metal::CircularBufferConfig input_pages_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * aligned_input_page_size, {{input_pages_cb_index, input_cb_data_format}})
            .set_page_size(input_pages_cb_index, aligned_input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, input_pages_cb_config);

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
        tt::tt_metal::CreateCircularBuffer(program, all_cores, output_pages_cb_config);
    }

    // Reader kernel config with compile-time args
    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);

    // Writer kernel config with compile-time args
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(output_page_cb_index)};

    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    if (convert_df) {
        compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{});
    }

    // Set runtime args
    uint32_t start_tile_id = 0;
    for (const auto& core : ordered_cores) {
        uint32_t num_tiles_to_process = num_tiles_per_core_group_1;
        if (core_group_2.contains(core)) {
            num_tiles_to_process = num_tiles_per_core_group_2;
        }
        // Reader run-time args
        std::vector<uint32_t> reader_run_time_args = {input.buffer()->address(), num_tiles_to_process, start_tile_id};
        std::vector<uint32_t> writer_run_time_args = {output.buffer()->address(), num_tiles_to_process, start_tile_id};
        start_tile_id += num_tiles_to_process;
        // Set run-time arg
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_run_time_args);

        if (convert_df) {
            std::vector<uint32_t> compute_run_time_args = {num_tiles_to_process};
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_run_time_args);
        }
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, ordered_cores}};
}
void CopyToMemoryConfigTilizedDefaultProgramFactory::override_runtime_arguments(
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
