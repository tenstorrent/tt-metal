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
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/runtime_args_data.hpp"
#include "tt_stl/assert.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization::softmax_backward {

ComputeConfig precise(std::vector<uint32_t> compile_time_args, std::map<std::string, std::string> defines) {
    ComputeConfig config;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.math_fidelity = MathFidelity::HiFi4;
    config.compile_args = std::move(compile_time_args);
    config.defines = std::move(defines);
    return config;
}

SoftmaxBackwardProgramFactory::cached_program_t SoftmaxBackwardProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const ttnn::Tensor& softmax_output = tensor_args.softmax_output;
    const ttnn::Tensor& upstream_grad = tensor_args.upstream_grad;
    const uint32_t dim = operation_attributes.dim;

    distributed::MeshDevice* device = softmax_output.device();
    Program program = CreateProgram();

    // Get tensor properties
    const ttnn::Shape& shape = softmax_output.padded_shape();
    const ttnn::Shape& logical_shape = softmax_output.logical_shape();
    const size_t rank = shape.rank();

    // For simplicity, we'll implement for the last dimension (most common case)
    // This can be extended to support other dimensions
    TT_ASSERT(
        dim == rank - 1 || dim == static_cast<uint32_t>(-1),
        "Currently only supporting softmax_backward on last dimension");

    const uint32_t height = shape[-2];
    const uint32_t width = shape[-1];
    const uint32_t height_tiles = height / constants::TILE_HEIGHT;
    const uint32_t width_tiles = width / constants::TILE_WIDTH;

    // Get logical width to determine padding mask
    const uint32_t logical_width = logical_shape[-1];
    const uint32_t mask_w = logical_width % constants::TILE_WIDTH;  // Position where padding starts in last tile

    // Calculate number of tiles to process
    const auto num_outer_dims = softmax_output.physical_volume() / height / width;
    const uint32_t num_rows = num_outer_dims * height_tiles;

    // Core configuration
    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const CoreRange all_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Distribute work across cores
    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t tiles_per_core = (num_rows + num_cores - 1) / num_cores;

    // Data formats
    const DataFormat input_data_format = datatype_to_dataformat_converter(softmax_output.dtype());
    const DataFormat output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const DataFormat intermed_data_format = DataFormat::Float16_b;  // Use bfloat16 for intermediate calculations

    const uint32_t input_tile_size = tile_size(input_data_format);
    const uint32_t output_tile_size = tile_size(output_data_format);
    const uint32_t intermed_tile_size = tile_size(intermed_data_format);

    // Adjustable batch size - must match all kernels (reader, compute, writer)
    constexpr uint32_t tiles_per_batch = 4;

    // Create circular buffers
    const uint32_t src0_cb_index = tt::CBIndex::c_0;        // softmax_output
    const uint32_t src1_cb_index = tt::CBIndex::c_1;        // upstream_grad
    const uint32_t ones_cb_index = tt::CBIndex::c_2;        // ones vector for matmul reduction
    const uint32_t out_cb_index = tt::CBIndex::c_7;         // output
    const uint32_t intermed0_cb_index = tt::CBIndex::c_13;  // y * grad
    const uint32_t intermed1_cb_index = tt::CBIndex::c_14;  // sum(y * grad) - accumulated
    const uint32_t intermed2_cb_index = tt::CBIndex::c_15;  // batch sum temporary

    // Create circular buffers - ALL batch-sized for minimal L1 footprint!
    // Two-pass streaming algorithm: Pass 1 computes sum, Pass 2 computes output
    // Input data is read twice (once per pass), but eliminates L1 overflow

    auto c_in0_config = CircularBufferConfig(tiles_per_batch * input_tile_size, {{src0_cb_index, input_data_format}})
                            .set_page_size(src0_cb_index, input_tile_size);
    CreateCircularBuffer(program, all_cores, c_in0_config);

    auto c_in1_config = CircularBufferConfig(tiles_per_batch * input_tile_size, {{src1_cb_index, input_data_format}})
                            .set_page_size(src1_cb_index, input_tile_size);
    CreateCircularBuffer(program, all_cores, c_in1_config);

    auto c_scaler_config = CircularBufferConfig(1 * intermed_tile_size, {{ones_cb_index, intermed_data_format}})
                               .set_page_size(ones_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_scaler_config);

    auto c_out_config = CircularBufferConfig(tiles_per_batch * output_tile_size, {{out_cb_index, output_data_format}})
                            .set_page_size(out_cb_index, output_tile_size);
    CreateCircularBuffer(program, all_cores, c_out_config);

    // intermed0: batch-sized temporary buffer for y * grad products
    auto c_intermed0_config =
        CircularBufferConfig(tiles_per_batch * intermed_tile_size, {{intermed0_cb_index, intermed_data_format}})
            .set_page_size(intermed0_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed0_config);

    // intermed1: accumulated sum (single tile reused every batch)
    auto c_intermed1_config = CircularBufferConfig(1 * intermed_tile_size, {{intermed1_cb_index, intermed_data_format}})
                                  .set_page_size(intermed1_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed1_config);

    // intermed2: batch sum temporary (single tile)
    auto c_intermed2_config = CircularBufferConfig(1 * intermed_tile_size, {{intermed2_cb_index, intermed_data_format}})
                                  .set_page_size(intermed2_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed2_config);

    // Compile time arguments for kernels
    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index,
        src1_cb_index,
        ones_cb_index,
        width_tiles  // num_tiles_per_row
    };
    TensorAccessorArgs(softmax_output.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(upstream_grad.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {out_cb_index, width_tiles};
    TensorAccessorArgs(tensor_return_value.buffer()).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {
        src0_cb_index,       // 0: softmax_output
        src1_cb_index,       // 1: upstream_grad
        out_cb_index,        // 2: output
        intermed0_cb_index,  // 3: y * grad
        intermed1_cb_index,  // 4: sum(y * grad) - accumulated
        ones_cb_index,       // 5: ones vector for matmul reduction
        intermed2_cb_index,  // 6: batch sum temporary
        width_tiles,         // 7: num_tiles_per_row
        mask_w               // 8: padding mask position in last tile (0 if no padding)
    };

    // Defines for compute kernel
    const std::map<std::string, std::string> compute_defines = {{"BROADCAST_TYPE", "BroadcastType::COL"}};

    const ComputeConfig wconf = precise(compute_compile_time_args, compute_defines);

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
        wconf);

    // Set runtime arguments
    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        CoreCoord core = {core_idx / num_cores_y, core_idx % num_cores_y};

        uint32_t start_tile = core_idx * tiles_per_core;
        if (start_tile >= num_rows) {
            continue;
        }
        uint32_t end_tile = std::min(start_tile + tiles_per_core, num_rows);
        uint32_t num_tiles_this_core = end_tile - start_tile;

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
        SetRuntimeArgs(program, compute_kernel_id, core, {num_tiles_this_core, width_tiles});
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

    const Tensor& softmax_output = tensor_args.softmax_output;
    const Tensor& upstream_grad = tensor_args.upstream_grad;

    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);
    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const CoreCoord core = {core_idx / num_cores_y, core_idx % num_cores_y};

        // Update reader runtime args
        RuntimeArgsData& reader_runtime_args_per_core = reader_runtime_args[core.x][core.y];
        reader_runtime_args_per_core[0] = softmax_output.buffer()->address();
        reader_runtime_args_per_core[1] = upstream_grad.buffer()->address();

        // Update writer runtime args
        RuntimeArgsData& writer_runtime_args_per_core = writer_runtime_args[core.x][core.y];
        writer_runtime_args_per_core[0] = tensor_return_value.buffer()->address();
    }
}

}  // namespace ttnn::operations::normalization::softmax_backward
