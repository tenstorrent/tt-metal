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

    // Use the sliding window config passed from upsample.cpp (contains original input dimensions)
    TT_FATAL(
        operation_attributes.sliding_window_config.has_value(),
        "Bilinear upsample requires sliding_window_config to be provided");
    const sliding_window::SlidingWindowConfig sliding_window_config =
        operation_attributes.sliding_window_config.value();

    // Output dimensions
    const tt::tt_metal::Shape& output_shape = output.padded_shape();

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt::tt_metal::IDevice* const device = output.device();

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Declare variables that will be set based on layout
    uint32_t input_unit_size;
    uint32_t output_unit_size;
    uint32_t input_cb_required_pages;
    uint32_t aligned_input_unit_size;  // Size used for CB creation
    uint32_t work_units_to_split;

    const bool is_tiled_layout = (input.layout() == tt::tt_metal::Layout::TILE);
    if (is_tiled_layout) {
        // Tiled layout specific calculations
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        aligned_input_unit_size = input_unit_size;

        const uint32_t input_tensor_width = a.padded_shape()[-1];
        const uint32_t input_tensor_height = a.physical_volume() / input_tensor_width;

        const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
        const uint32_t tile_height = tile_shape[0];
        const uint32_t tile_width = tile_shape[1];

        const uint32_t num_input_tiles_in_row = input_tensor_width / tile_width;
        const uint32_t num_input_tiles_in_col = input_tensor_height / tile_height;

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
    if (work_per_core_group_1 != 1) {
        // Double buffer if the core is processing 2+ blocks
        num_pages_in_input_cb *= 2;
    }

    const auto [srcA_cb_index, cb_srcA] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, aligned_input_unit_size, num_pages_in_input_cb, input_cb_data_format);

    uint32_t output_cb_index = 0;
    if (is_tiled_layout) {
        // Separate output CB for tiled
        uint32_t num_pages_in_output_cb = num_pages_in_input_cb;
        const auto [out_cb_index, cb_output] = create_cb(
            next_cb_index++, program, all_cores, output_unit_size, num_pages_in_output_cb, output_cb_data_format);
        output_cb_index = out_cb_index;
    } else {
        // Same CB for input and output for row-major
        output_cb_index = src0_cb_index;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel = reader_kernel,
            .writer_kernel = writer_kernel,
            .cb_srcA = cb_srcA,
            .cb_srcB = cb_srcB,
            .cb_srcC = cb_srcC,
            .out_cb = out_cb,
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
