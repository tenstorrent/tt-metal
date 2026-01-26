// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <string>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/madd/device/madd_program_factory.hpp"

namespace ttnn::prim {

static inline auto get_tile_shape(const Tensor& x) {
    const auto tile = x.tensor_spec().tile();
    return std::make_pair(tile.get_height(), tile.get_width());
}

static inline auto get_tile_count(const Tensor& x) {
    const uint32_t input_tensor_width = x.padded_shape()[-1];
    const uint32_t input_tensor_height = x.physical_volume() / input_tensor_width;

    const auto [tile_height, tile_width] = get_tile_shape(x);

    const uint32_t num_input_tiles_in_row = input_tensor_width / tile_width;
    const uint32_t num_input_tiles_in_col = input_tensor_height / tile_height;

    return std::make_pair(num_input_tiles_in_row, num_input_tiles_in_col);
}

MAddProgramFactory::cached_program_t MAddProgramFactory::create(
    [[maybe_unused]] const MAddParams& operation_attributes, const MAddArgs& tensor_args, Tensor& output_tensor) {
    // unpack tensor args
    const ttnn::Tensor& a = tensor_args.a;
    const ttnn::Tensor& b = tensor_args.b;
    const ttnn::Tensor& c = tensor_args.c;
    const ttnn::Tensor& output = output_tensor;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // const ttnn::DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    // TODO: Verify that all inputs have the same data format

    // Output dimensions
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt::tt_metal::IDevice* const device = output.device();

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Declare variables that will be set based on layout
    uint32_t input_unit_size;
    uint32_t input_cb_required_pages;
    uint32_t aligned_input_unit_size;  // Size used for CB creation

    uint32_t output_unit_size;

    uint32_t work_units_to_split;

    const bool is_tiled_layout = (a.layout() == tt::tt_metal::Layout::TILE);
    if (is_tiled_layout) {
        // Tiled layout specific calculations
        input_unit_size = tt::tile_size(input_cb_data_format);
        aligned_input_unit_size = input_unit_size;

        output_unit_size = tt::tile_size(output_cb_data_format);

        const auto [num_input_tiles_in_row, num_input_tiles_in_col] = get_tile_count(a);

        /*
        For tiled layout, a unit of work (input wise) is a row of tiles
        */

        input_cb_required_pages = num_input_tiles_in_row;  // for CB sizing
        work_units_to_split = num_input_tiles_in_col;      // for work splitting
    } else {
        TT_THROW("Only tiled layout is supported for MAddOperation");
    }

    const auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_group_1, work_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, work_units_to_split);

    // Create circular buffers
    uint32_t next_cb_index = tt::CBIndex::c_0;
    uint32_t num_pages_in_input_cb;

    num_pages_in_input_cb = input_cb_required_pages;
    if (work_per_core_group_1 > 1) {
        // Double buffer if the core is processing 2+ blocks
        num_pages_in_input_cb *= 2;
    }

    // Create circular buffers for inputs
    const auto [cb_srcA_index, cb_srcA] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, aligned_input_unit_size, num_pages_in_input_cb, input_cb_data_format);

    const auto [cb_srcB_index, cb_srcB] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, aligned_input_unit_size, num_pages_in_input_cb, input_cb_data_format);

    const auto [cb_srcC_index, cb_srcC] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, aligned_input_unit_size, num_pages_in_input_cb, input_cb_data_format);

    // Zero tile CB
    const auto [cb_zero_index, cb_zero] =
        tt::tt_metal::create_cb(next_cb_index++, program, all_cores, aligned_input_unit_size, 1, input_cb_data_format);

    // Separate output CB for tiled
    const uint32_t num_pages_in_output_cb = num_pages_in_input_cb;  // double buffered if needed
    const auto [cb_output_index, cb_output] =
        create_cb(next_cb_index++, program, all_cores, output_unit_size, num_pages_in_output_cb, output_cb_data_format);

    auto* const a_buffer = a.buffer();
    auto* const b_buffer = b.buffer();
    auto* const c_buffer = c.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)cb_srcA_index,
        (std::uint32_t)cb_srcB_index,
        (std::uint32_t)cb_srcC_index,
        (std::uint32_t)cb_zero_index,
        (std::uint32_t)aligned_input_unit_size,
    };

    tt::tt_metal::TensorAccessorArgs(a_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(b_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(c_buffer).append_to(reader_compile_time_args);

    const tt::tt_metal::KernelHandle reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/madd/device/kernels/dataflow/"
        "reader_madd_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer compile time arguments

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)cb_output_index, (std::uint32_t)output_unit_size, (std::uint32_t)input_cb_required_pages};

    auto* const output_buffer = output.buffer();
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);

    const std::map<std::string, std::string> kernel_defines;
    const tt::tt_metal::KernelHandle writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/madd/device/kernels/dataflow/writer_madd_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    {  // Compute kernel (only for tiled layout)

        // Create compute kernel for core group 1 if it has cores
        if (core_group_1.num_cores() > 0) {
            const uint32_t pages_to_process = work_per_core_group_1 * input_cb_required_pages;

            const std::vector<uint32_t> compute_compile_time_args_group1 = {
                (uint32_t)pages_to_process,
                (uint32_t)cb_srcA_index,
                (uint32_t)cb_srcB_index,
                (uint32_t)cb_srcC_index,
                (uint32_t)cb_zero_index,
                (uint32_t)cb_output_index};

            tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/madd/device/kernels/compute/madd_compute.cpp",
                core_group_1,
                tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args_group1});
        }

        // Create compute kernel for core group 2 if it has cores
        if (core_group_2.num_cores() > 0) {
            const uint32_t pages_to_process = work_per_core_group_2 * input_cb_required_pages;

            const std::vector<uint32_t> compute_compile_time_args_group2 = {
                (uint32_t)pages_to_process,
                (uint32_t)cb_srcA_index,
                (uint32_t)cb_srcB_index,
                (uint32_t)cb_srcC_index,
                (uint32_t)cb_zero_index,
                (uint32_t)cb_output_index};

            tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/madd/device/kernels/compute/madd_compute.cpp",
                core_group_2,
                tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args_group2});
        }
    }

    // Set up runtime arguments
    std::vector<uint32_t> reader_rt_arguments{
        a_buffer->address(),
        b_buffer->address(),
        c_buffer->address(),
        0,  // set in loop, num of units on core
        0   // set in loop, start_id of unit in core
    };

    std::vector<uint32_t> writer_rt_arguments{
        output_buffer->address(),
        0,  // set in loop, num of units on core
        0   // set in loop, start_id of unit on core
    };

    for (uint32_t i = 0, blocks_processed = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};  // x, y, looks like a bug.
        uint32_t blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            blocks_per_core = work_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            blocks_per_core = work_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        reader_rt_arguments[3] = blocks_per_core * input_cb_required_pages;   // reader goes page by page
        reader_rt_arguments[4] = blocks_processed * input_cb_required_pages;  // offset in pages

        writer_rt_arguments[1] = blocks_per_core * input_cb_required_pages;
        writer_rt_arguments[2] = blocks_processed * input_cb_required_pages;

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, reader_rt_arguments);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core, writer_rt_arguments);

        blocks_processed += blocks_per_core;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel = reader_kernel,
            .writer_kernel = writer_kernel,
            .num_cores = num_cores,
            .num_cores_y = num_cores_y}};
}

void MAddProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MAddParams& /*operation_attributes*/,
    const MAddArgs& tensor_args,
    [[maybe_unused]] Tensor& output_tensor) {
    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel;
    const auto& num_cores = cached_program.shared_variables.num_cores;
    const auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    auto* const a_buffer = tensor_args.a.buffer();
    auto* const b_buffer = tensor_args.b.buffer();
    auto* const c_buffer = tensor_args.c.buffer();
    auto* const output_buffer = output_tensor.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = a_buffer->address();
            runtime_args[1] = b_buffer->address();
            runtime_args[2] = c_buffer->address();
        }
        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
