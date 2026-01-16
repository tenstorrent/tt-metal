// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_non_streaming_program_factory.hpp"
#include "softmax_backward_program_factory_common.hpp"

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
// #include "tt-metalium/base_types.hpp"
// #include "tt_stl/assert.hpp"
// #include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization::softmax_backward {

SoftmaxBackwardNonStreamingFactory::cached_program_t SoftmaxBackwardNonStreamingFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const ttnn::Tensor& softmax_output = tensor_args.softmax_output;
    const ttnn::Tensor& upstream_grad = tensor_args.upstream_grad;

    distributed::MeshDevice* device = softmax_output.device();
    Program program = CreateProgram();

    uint32_t num_rows, width_tiles, mask_w;
    DataFormat input_data_format, output_data_format, intermed_data_format;
    uint32_t input_tile_size, output_tile_size, intermed_tile_size;

    get_tensor_properties(
        softmax_output,
        operation_attributes,
        num_rows,
        width_tiles,
        mask_w,
        input_data_format,
        output_data_format,
        intermed_data_format,
        input_tile_size,
        output_tile_size,
        intermed_tile_size,
        tensor_return_value);

    // Log kernel selection for debugging and testing verification
    const uint32_t total_tiles = num_rows * width_tiles;
    const auto [use_non_streaming_kernel_orig, estimated_memory_bytes] =
        should_use_non_streaming_kernel(width_tiles, intermed_tile_size);
    const uint32_t estimated_memory_kb = estimated_memory_bytes / 1024;

    log_info(
        tt::LogOp,
        "SoftmaxBackward: Using NON-STREAMING kernel | Shape: {}x{} tiles ({} total) | Estimated L1: {} KB",
        num_rows,
        width_tiles,
        total_tiles,
        estimated_memory_kb);

    // Core configuration
    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const CoreRange all_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Distribute work across cores
    const uint32_t num_cores = num_cores_x * num_cores_y;

    // Create circular buffers - row-by-row processing for non-streaming kernel
    const uint32_t src0_cb_index = tt::CBIndex::c_0;        // softmax_output (y)
    const uint32_t src1_cb_index = tt::CBIndex::c_1;        // upstream_grad (grad)
    const uint32_t ones_cb_index = tt::CBIndex::c_2;        // ones vector for matmul reduction
    const uint32_t out_cb_index = tt::CBIndex::c_7;         // output
    const uint32_t intermed0_cb_index = tt::CBIndex::c_13;  // y * grad
    const uint32_t intermed1_cb_index = tt::CBIndex::c_14;  // sum(y * grad)

    // Input buffers: sized for one row (width_tiles)
    auto c_in0_config = CircularBufferConfig(width_tiles * input_tile_size * 2, {{src0_cb_index, input_data_format}})
                            .set_page_size(src0_cb_index, input_tile_size);
    CreateCircularBuffer(program, all_cores, c_in0_config);

    auto c_in1_config = CircularBufferConfig(width_tiles * input_tile_size * 2, {{src1_cb_index, input_data_format}})
                            .set_page_size(src1_cb_index, input_tile_size);
    CreateCircularBuffer(program, all_cores, c_in1_config);

    // Ones vector for reduction
    auto c_scaler_config = CircularBufferConfig(1 * intermed_tile_size, {{ones_cb_index, intermed_data_format}})
                               .set_page_size(ones_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_scaler_config);

    // Output buffer: sized for one row (width_tiles)
    auto c_out_config = CircularBufferConfig(width_tiles * output_tile_size * 2, {{out_cb_index, output_data_format}})
                            .set_page_size(out_cb_index, output_tile_size);
    CreateCircularBuffer(program, all_cores, c_out_config);

    // intermed0: y * grad (one row)
    auto c_intermed0_config =
        CircularBufferConfig(width_tiles * intermed_tile_size * 2, {{intermed0_cb_index, intermed_data_format}})
            .set_page_size(intermed0_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed0_config);

    // intermed1: sum(y * grad) - single tile
    auto c_intermed1_config = CircularBufferConfig(1 * intermed_tile_size, {{intermed1_cb_index, intermed_data_format}})
                                  .set_page_size(intermed1_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed1_config);

    // Compile time arguments for kernels
    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index,
        src1_cb_index,
        ones_cb_index,
        width_tiles  // num_tiles_per_row
    };
    TensorAccessorArgs(softmax_output.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(upstream_grad.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        out_cb_index,
        width_tiles  // num_tiles_per_row
    };
    TensorAccessorArgs(tensor_return_value.buffer()).append_to(writer_compile_time_args);

    const std::vector<uint32_t> compute_compile_time_args = {
        src0_cb_index,       // 0: y_cb_id (softmax_output)
        src1_cb_index,       // 1: grad_cb_id (upstream_grad)
        out_cb_index,        // 2: out_cb_id (output)
        intermed0_cb_index,  // 3: mul_cb_id (y * grad)
        intermed1_cb_index,  // 4: sum_reduce_cb_id (sum(y * grad))
        ones_cb_index,       // 6: ones_cb_id (ones vector for matmul reduction)
        width_tiles          // 7: num_tiles_per_row
    };

    // Defines for compute kernel
    const std::map<std::string, std::string> compute_defines = {{"BROADCAST_TYPE", "BroadcastType::COL"}};

    const ComputeConfig wconf = precise(compute_compile_time_args, compute_defines);

    // Create kernels using _small variants
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax_backward/device/kernels/dataflow/"
        "reader_softmax_backward_small.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax_backward/device/kernels/dataflow/"
        "writer_softmax_backward_small.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax_backward/device/kernels/compute/"
        "softmax_backward_kernel_small.cpp",
        all_cores,
        wconf);

    // Set common runtime arguments for reader (addresses + grid info for coordinate calculation)
    SetCommonRuntimeArgs(
        program,
        reader_kernel_id,
        {softmax_output.buffer()->address(),
         upstream_grad.buffer()->address(),
         num_rows,       // total rows to process
         num_cores_x,    // grid width
         num_cores_y});  // grid height

    // Set common runtime arguments for writer (addresses + grid info for coordinate calculation)
    SetCommonRuntimeArgs(
        program,
        writer_kernel_id,
        {tensor_return_value.buffer()->address(),
         num_rows,       // total rows to process
         num_cores_x,    // grid width
         num_cores_y});  // grid height

    // Set per-core runtime arguments for compute only (compute kernels can't access coordinates)
    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        CoreCoord core = {core_idx / num_cores_y, core_idx % num_cores_y};

        const uint32_t rows_per_core = (num_rows + num_cores - 1) / num_cores;
        const uint32_t start_row = core_idx * rows_per_core;
        const uint32_t end_row = std::min(start_row + rows_per_core, num_rows);
        const uint32_t num_rows_this_core = start_row < num_rows ? end_row - start_row : 0;

        // Compute runtime args: (num_rows, width_tiles)
        SetRuntimeArgs(program, compute_kernel_id, core, {num_rows_this_core, width_tiles});
    }

    return {
        std::move(program), {.unary_reader_kernel_id = reader_kernel_id, .unary_writer_kernel_id = writer_kernel_id}};
}

void SoftmaxBackwardNonStreamingFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const Tensor& softmax_output = tensor_args.softmax_output;
    const Tensor& upstream_grad = tensor_args.upstream_grad;

    // Update common runtime args (shared across all cores)
    auto& reader_common_args = GetCommonRuntimeArgs(program, reader_kernel_id);
    reader_common_args[0] = softmax_output.buffer()->address();
    reader_common_args[1] = upstream_grad.buffer()->address();

    auto& writer_common_args = GetCommonRuntimeArgs(program, writer_kernel_id);
    writer_common_args[0] = tensor_return_value.buffer()->address();
}

}  // namespace ttnn::operations::normalization::softmax_backward
