// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_program_factory.hpp"

#include <algorithm>  // for std::min
#include <cstdint>
#include <utility>  // for std::move
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "hostdevcommon/kernel_structs.h"
#include "tt_stl/assert.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization::softmax_backward {

SoftmaxBackwardProgramFactory::cached_program_t SoftmaxBackwardProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& softmax_output = tensor_args.softmax_output;
    const auto& upstream_grad = tensor_args.upstream_grad;
    const auto dim = operation_attributes.dim;

    auto* device = softmax_output.device();
    auto program = CreateProgram();

    // Get tensor properties
    const auto shape = softmax_output.padded_shape();
    const auto rank = shape.rank();

    // For simplicity, we'll implement for the last dimension (most common case)
    // This can be extended to support other dimensions
    TT_ASSERT(
        dim == rank - 1 || dim == static_cast<uint32_t>(-1),
        "Currently only supporting softmax_backward on last dimension");

    const auto height = shape[-2];
    const auto width = shape[-1];
    const auto height_tiles = height / constants::TILE_HEIGHT;

    // Calculate number of tiles to process
    auto num_outer_dims = softmax_output.physical_volume() / height / width;
    uint32_t num_rows = num_outer_dims * height_tiles;

    // Core configuration
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Distribute work across cores
    uint32_t num_cores = num_cores_x * num_cores_y;
    uint32_t tiles_per_core = (num_rows + num_cores - 1) / num_cores;

    // Data formats
    auto input_data_format = datatype_to_dataformat_converter(softmax_output.dtype());
    auto output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    auto intermed_data_format = DataFormat::Float16_b;  // Use bfloat16 for intermediate calculations

    uint32_t input_tile_size = tile_size(input_data_format);
    uint32_t output_tile_size = tile_size(output_data_format);
    uint32_t intermed_tile_size = tile_size(intermed_data_format);

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;        // softmax_output
    uint32_t src1_cb_index = tt::CBIndex::c_1;        // upstream_grad
    uint32_t out_cb_index = tt::CBIndex::c_16;        // output
    uint32_t intermed0_cb_index = tt::CBIndex::c_24;  // y * grad
    uint32_t intermed1_cb_index = tt::CBIndex::c_25;  // sum(y * grad)
    uint32_t intermed2_cb_index = tt::CBIndex::c_26;  // grad - sum(y * grad)

    // Create circular buffers for simplified implementation
    auto c_in0_config = CircularBufferConfig(1 * input_tile_size, {{src0_cb_index, input_data_format}})
                            .set_page_size(src0_cb_index, input_tile_size);
    CreateCircularBuffer(program, all_cores, c_in0_config);

    auto c_in1_config = CircularBufferConfig(1 * input_tile_size, {{src1_cb_index, input_data_format}})
                            .set_page_size(src1_cb_index, input_tile_size);
    CreateCircularBuffer(program, all_cores, c_in1_config);

    auto c_out_config = CircularBufferConfig(1 * output_tile_size, {{out_cb_index, output_data_format}})
                            .set_page_size(out_cb_index, output_tile_size);
    CreateCircularBuffer(program, all_cores, c_out_config);

    auto c_intermed0_config = CircularBufferConfig(1 * intermed_tile_size, {{intermed0_cb_index, intermed_data_format}})
                                  .set_page_size(intermed0_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed0_config);

    auto c_intermed1_config = CircularBufferConfig(1 * intermed_tile_size, {{intermed1_cb_index, intermed_data_format}})
                                  .set_page_size(intermed1_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed1_config);

    auto c_intermed2_config = CircularBufferConfig(1 * intermed_tile_size, {{intermed2_cb_index, intermed_data_format}})
                                  .set_page_size(intermed2_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed2_config);

    // Compile time arguments for kernels
    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index,
        src1_cb_index,
        1  // simplified to process one tile at a time
    };
    TensorAccessorArgs(softmax_output.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(upstream_grad.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {out_cb_index};
    TensorAccessorArgs(tensor_return_value.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        src0_cb_index,
        src1_cb_index,
        out_cb_index,
        intermed0_cb_index,
        intermed1_cb_index,
        intermed2_cb_index,
        1  // simplified to process one tile at a time
    };

    // Create kernels
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax_backward/device/kernels/dataflow/reader_softmax_backward.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax_backward/device/kernels/dataflow/writer_softmax_backward.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax_backward/device/kernels/compute/softmax_backward_kernel.cpp",
        all_cores,
        ComputeConfig{.compile_args = compute_compile_time_args});

    // Set runtime arguments
    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        CoreCoord core = {core_idx / num_cores_y, core_idx % num_cores_y};

        uint32_t start_tile = core_idx * tiles_per_core;
        uint32_t end_tile = std::min(start_tile + tiles_per_core, num_rows);
        uint32_t num_tiles_this_core = end_tile - start_tile;

        if (num_tiles_this_core == 0) {
            continue;
        }

        // Reader runtime args
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {softmax_output.buffer()->address(), upstream_grad.buffer()->address(), start_tile, num_tiles_this_core});

        // Writer runtime args
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {tensor_return_value.buffer()->address(), start_tile, num_tiles_this_core});

        // Compute runtime args
        SetRuntimeArgs(program, compute_kernel_id, core, {num_tiles_this_core});
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void SoftmaxBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const auto& softmax_output = tensor_args.softmax_output;
    const auto& upstream_grad = tensor_args.upstream_grad;

    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        CoreCoord core = {core_idx / num_cores_y, core_idx % num_cores_y};

        // Update reader runtime args
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        reader_runtime_args[0] = softmax_output.buffer()->address();
        reader_runtime_args[1] = upstream_grad.buffer()->address();

        // Update writer runtime args
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        writer_runtime_args[0] = tensor_return_value.buffer()->address();
    }
}

}  // namespace ttnn::operations::normalization::softmax_backward
