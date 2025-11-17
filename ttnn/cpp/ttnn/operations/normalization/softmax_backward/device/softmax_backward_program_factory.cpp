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

// Estimate L1 memory usage for small tensor kernel
// Returns true if tensor fits in L1 memory for non-streaming approach
bool should_use_small_kernel(uint32_t num_rows, uint32_t width_tiles, uint32_t tile_size) {
    // L1 memory available for circular buffers (conservative estimate)
    constexpr uint32_t L1_AVAILABLE_FOR_CBS = 1024 * 1024;  // ~1MB available

    const uint32_t total_tiles = num_rows * width_tiles;

    // Memory requirements for small kernel:
    // - Y input: total_tiles
    // - grad input: total_tiles
    // - output: total_tiles
    // - mul_cb: width_tiles (one row at a time)
    // - sum_reduce_cb: 1 tile
    // - ones_cb: 1 tile

    const uint32_t memory_needed = (total_tiles * 2) * tile_size +  // Y and grad inputs (2x buffer)
                                   (total_tiles * 2) * tile_size +  // Output (2x buffer)
                                   (width_tiles * 2) * tile_size +  // mul_cb (2x buffer)
                                   tile_size +                      // sum_reduce_cb
                                   tile_size;                       // ones_cb

    return memory_needed < L1_AVAILABLE_FOR_CBS;
}

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
    TT_FATAL(
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

    // Data formats
    const DataFormat input_data_format = datatype_to_dataformat_converter(softmax_output.dtype());
    const DataFormat output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const DataFormat intermed_data_format = DataFormat::Float16_b;  // Use bfloat16 for intermediate calculations

    const uint32_t input_tile_size = tile_size(input_data_format);
    const uint32_t output_tile_size = tile_size(output_data_format);
    const uint32_t intermed_tile_size = tile_size(intermed_data_format);

    // Decide between small (non-streaming) and large (streaming) kernel
    const bool use_small_kernel = should_use_small_kernel(num_rows, width_tiles, intermed_tile_size);

    // Log kernel selection for debugging and testing verification
    const uint32_t total_tiles = num_rows * width_tiles;
    const uint32_t estimated_memory_kb =
        ((total_tiles * 2 * 2 + width_tiles * 2) * intermed_tile_size + 2 * intermed_tile_size) / 1024;

    log_info(
        tt::LogOp,
        "SoftmaxBackward: Using {} kernel | Shape: {}x{} tiles ({} total) | Estimated L1: {} KB",
        use_small_kernel ? "SMALL (non-streaming)" : "LARGE (streaming)",
        num_rows,
        width_tiles,
        total_tiles,
        estimated_memory_kb);

    if (use_small_kernel) {
        // Small kernel: tensor fits in L1, use non-streaming approach
        return SoftmaxBackwardProgramFactory::create_small_kernel_program(
            program,
            device,
            softmax_output,
            upstream_grad,
            tensor_return_value,
            num_rows,
            width_tiles,
            mask_w,
            input_data_format,
            output_data_format,
            intermed_data_format,
            input_tile_size,
            output_tile_size,
            intermed_tile_size);
    } else {
        // Large kernel: use streaming approach
        return SoftmaxBackwardProgramFactory::create_large_kernel_program(
            program,
            device,
            softmax_output,
            upstream_grad,
            tensor_return_value,
            num_rows,
            width_tiles,
            mask_w,
            input_data_format,
            output_data_format,
            intermed_data_format,
            input_tile_size,
            output_tile_size,
            intermed_tile_size);
    }
}

SoftmaxBackwardProgramFactory::cached_program_t SoftmaxBackwardProgramFactory::create_small_kernel_program(
    Program& program,
    distributed::MeshDevice* device,
    const ttnn::Tensor& softmax_output,
    const ttnn::Tensor& upstream_grad,
    ttnn::Tensor& tensor_return_value,
    uint32_t num_rows,
    uint32_t width_tiles,
    uint32_t mask_w,
    DataFormat input_data_format,
    DataFormat output_data_format,
    DataFormat intermed_data_format,
    uint32_t input_tile_size,
    uint32_t output_tile_size,
    uint32_t intermed_tile_size) {
    // Core configuration
    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const CoreRange all_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Distribute work across cores
    const uint32_t num_cores = num_cores_x * num_cores_y;

    // Create circular buffers - row-by-row processing for small kernel
    const uint32_t src0_cb_index = tt::CBIndex::c_0;        // softmax_output (y)
    const uint32_t src1_cb_index = tt::CBIndex::c_1;        // upstream_grad (grad)
    const uint32_t ones_cb_index = tt::CBIndex::c_2;        // ones vector for matmul reduction
    const uint32_t out_cb_index = tt::CBIndex::c_7;         // output
    const uint32_t intermed0_cb_index = tt::CBIndex::c_13;  // y * grad
    const uint32_t intermed1_cb_index = tt::CBIndex::c_14;  // sum(y * grad)
    const uint32_t intermed2_cb_index = tt::CBIndex::c_15;  // grad - sum(y * grad)

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

    // intermed2: grad - sum (single tile for intermediate subtraction result)
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
        intermed2_cb_index,  // 5: grad_minus_sum_cb_id (grad - sum)
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
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

SoftmaxBackwardProgramFactory::cached_program_t SoftmaxBackwardProgramFactory::create_large_kernel_program(
    Program& program,
    distributed::MeshDevice* device,
    const ttnn::Tensor& softmax_output,
    const ttnn::Tensor& upstream_grad,
    ttnn::Tensor& tensor_return_value,
    uint32_t num_rows,
    uint32_t width_tiles,
    uint32_t mask_w,
    DataFormat input_data_format,
    DataFormat output_data_format,
    DataFormat intermed_data_format,
    uint32_t input_tile_size,
    uint32_t output_tile_size,
    uint32_t intermed_tile_size) {
    // Adjustable block size - must match all kernels (reader, compute, writer)
    constexpr uint32_t tiles_per_block = 4;

    // Core configuration
    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const CoreRange all_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Distribute work across cores
    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t tiles_per_core = (num_rows + num_cores - 1) / num_cores;

    // Create circular buffers
    const uint32_t src0_cb_index = tt::CBIndex::c_0;        // softmax_output
    const uint32_t src1_cb_index = tt::CBIndex::c_1;        // upstream_grad
    const uint32_t ones_cb_index = tt::CBIndex::c_2;        // ones vector for matmul reduction
    const uint32_t out_cb_index = tt::CBIndex::c_7;         // output
    const uint32_t intermed0_cb_index = tt::CBIndex::c_13;  // y * grad
    const uint32_t intermed1_cb_index = tt::CBIndex::c_14;  // sum(y * grad) - accumulated
    const uint32_t intermed2_cb_index = tt::CBIndex::c_15;  // block sum temporary

    // Create circular buffers - ALL block-sized for minimal L1 footprint!
    // Two-pass streaming algorithm: Pass 1 computes sum, Pass 2 computes output
    // Input data is read twice (once per pass), but eliminates L1 overflow

    auto c_in0_config =
        CircularBufferConfig(tiles_per_block * input_tile_size * 2, {{src0_cb_index, input_data_format}})
            .set_page_size(src0_cb_index, input_tile_size);
    CreateCircularBuffer(program, all_cores, c_in0_config);

    auto c_in1_config =
        CircularBufferConfig(tiles_per_block * input_tile_size * 2, {{src1_cb_index, input_data_format}})
            .set_page_size(src1_cb_index, input_tile_size);
    CreateCircularBuffer(program, all_cores, c_in1_config);

    auto c_scaler_config = CircularBufferConfig(1 * intermed_tile_size, {{ones_cb_index, intermed_data_format}})
                               .set_page_size(ones_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_scaler_config);

    auto c_out_config =
        CircularBufferConfig(tiles_per_block * output_tile_size * 2, {{out_cb_index, output_data_format}})
            .set_page_size(out_cb_index, output_tile_size);
    CreateCircularBuffer(program, all_cores, c_out_config);

    // intermed0: block-sized temporary buffer for y * grad products
    auto c_intermed0_config =
        CircularBufferConfig(tiles_per_block * intermed_tile_size * 2, {{intermed0_cb_index, intermed_data_format}})
            .set_page_size(intermed0_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed0_config);

    // intermed1: accumulated sum (single tile reused every block)
    auto c_intermed1_config = CircularBufferConfig(1 * intermed_tile_size, {{intermed1_cb_index, intermed_data_format}})
                                  .set_page_size(intermed1_cb_index, intermed_tile_size);
    CreateCircularBuffer(program, all_cores, c_intermed1_config);

    // intermed2: block sum temporary (single tile)
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
        intermed2_cb_index,  // 6: block sum temporary
        width_tiles,         // 7: num_tiles_per_row
        mask_w               // 8: padding mask position in last tile (0 if no padding)
    };

    // Defines for compute kernel
    const std::map<std::string, std::string> compute_defines = {{"BROADCAST_TYPE", "BroadcastType::COL"}};

    const ComputeConfig wconf = precise(compute_compile_time_args, compute_defines);

    // Create kernels using _large variants
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax_backward/device/kernels/dataflow/"
        "reader_softmax_backward_large.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax_backward/device/kernels/dataflow/"
        "writer_softmax_backward_large.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/softmax_backward/device/kernels/compute/"
        "softmax_backward_kernel_large.cpp",
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

        uint32_t start_tile = core_idx * tiles_per_core;
        if (start_tile >= num_rows) {
            continue;
        }
        uint32_t end_tile = std::min(start_tile + tiles_per_core, num_rows);
        uint32_t num_tiles_this_core = end_tile - start_tile;

        // Compute runtime args: (num_tiles, width_tiles)
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
