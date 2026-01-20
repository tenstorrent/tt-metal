// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/madd/device/madd_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::madd::program {

using FixedPoint = int32_t;
constexpr int32_t FIXED_POINT_SHIFT = 16;
constexpr int32_t FIXED_ONE = 1 << FIXED_POINT_SHIFT;

static FixedPoint float_to_fixed(float value) { return static_cast<FixedPoint>(value * FIXED_ONE); }

static inline auto get_tile_count(const Tensor& x) {
    const uint32_t input_tensor_width = x.padded_shape()[-1];
    const uint32_t input_tensor_height = x.physical_volume() / input_tensor_width;

    const auto& tile_shape = x.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_height = tile_shape[0];
    const uint32_t tile_width = tile_shape[1];

    const uint32_t num_input_tiles_in_row = input_tensor_width / tile_width;
    const uint32_t num_input_tiles_in_col = input_tensor_height / tile_height;

    return std::make_pair(num_input_tiles_in_row, num_input_tiles_in_col);
}

MAddProgramFactory::cached_program_t MAddProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const ttnn::Tensor& a = tensor_args.a;
    const ttnn::Tensor& b = tensor_args.b;
    const ttnn::Tensor& c = tensor_args.c;
    const ttnn::Tensor& output = output_tensor;

    const ttnn::DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Output dimensions
    const tt::tt_metal::Shape& output_shape = output.padded_shape();

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    // TODO: Verify that all inputs have the same data format

    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt::tt_metal::IDevice* const device = output.device();

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Declare variables that will be set based on layout
    uint32_t input_unit_size;
    uint32_t input_cb_required_pages;
    uint32_t aligned_input_unit_size;  // Size used for CB creation

    uint32_t output_unit_size;

    uint32_t work_units_to_split;

    const bool is_tiled_layout = (input.layout() == tt::tt_metal::Layout::TILE);
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

    // Separate output CB for tiled
    const uint32_t num_pages_in_output_cb = num_pages_in_input_cb;
    const auto [cb_output_index, cb_output] =
        create_cb(next_cb_index++, program, all_cores, output_unit_size, num_pages_in_output_cb, output_cb_data_format);

    auto* const a_buffer = a.buffer();
    auto* const b_buffer = b.buffer();
    auto* const c_buffer = c.buffer();
    auto* const dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)cb_srcA_index,
        (std::uint32_t)cb_srcB_index,
        (std::uint32_t)cb_srcC_index,
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

    // Set up runtime arguments
    std::vector<uint32_t> reader_rt_arguments{
        a_buffer->address(),
        b_buffer->address(),
        c_buffer->address(),
        0,  // set in loop, num of units on core
        0   // set in loop, start_id of unit in core
    };

    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
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

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, reader_rt_arguments);
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, writer_rt_arguments);

        blocks_processed += blocks_per_core;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel = reader_kernel,
            .writer_kernel = writer_kernel,
            .cb_srcA = cb_srcA,
            .cb_srcB = cb_srcB,
            .cb_srcC = cb_srcC,
            .out_cb = cb_output,
        }};
}

void MAddProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program& program = cached_program.program;
    tt::tt_metal::CBHandle& cb_srcA = cached_program.shared_variables.cb_srcA;
    tt::tt_metal::CBHandle& cb_srcB = cached_program.shared_variables.cb_srcB;
    tt::tt_metal::CBHandle& cb_srcC = cached_program.shared_variables.cb_srcC;
    tt::tt_metal::CBHandle& out_cb = cached_program.shared_variables.out_cb;

    const ttnn::Tensor& a = tensor_args.a;
    const ttnn::Tensor& b = tensor_args.b;
    const ttnn::Tensor& c = tensor_args.c;
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb_srcA, *a.buffer());
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb_srcB, *b.buffer());
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb_srcC, *c.buffer());
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, out_cb, *output_tensor.buffer());
}

}  // namespace ttnn::operations::madd::program
