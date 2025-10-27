// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optional"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cmath>
#include <algorithm>

#include "slice_op.hpp"
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

/**
 * Core allocation result structure for multi-core slice operations.
 * Contains all the computed values for core grid layout and allocation.
 */
struct CoreAllocation {
    uint32_t grid_x;               // Grid width (number of cores in x direction)
    uint32_t grid_y;               // Grid height (number of cores in y direction)
    uint32_t num_cores;            // Total number of cores used (grid_x * grid_y)
    uint32_t max_cores_available;  // Maximum cores available on device
    uint32_t num_cores_needed;     // Number of cores needed for optimal allocation
};

/**
 * Calculate total output rows based on tensor rank - this is what we distribute across cores.
 * Generalized for N-dimensional tensors. For rank R, rows = product(dims[0:R-1])
 * The last dimension is always processed as contiguous width data.
 */
inline uint32_t calculate_total_output_rows(const ttnn::Shape& output_shape) {
    auto rank = output_shape.rank();

    if (rank == 1) {
        return 1;  // 1D has only 1 "row" to process
    }

    // For N-dimensional tensors, calculate product of all dimensions except the last
    return output_shape.volume() / output_shape[-1];
}

/**
 * Core allocation and work distribution for multi-core slice operation.
 * This implements the key insight from Python: use only as many cores as we have work for.
 * Translated from Python get_multicore_slice_descriptor method.
 */
inline CoreAllocation calculate_core_allocation(tt::tt_metal::IDevice* device, uint32_t total_output_rows) {
    // Get hardware compute grid dimensions
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t max_cores_available = compute_grid.x * compute_grid.y;

    // CORE ALLOCATION DECISION:
    // Use min(available_cores, rows_to_process) to avoid over-parallelization
    // Each core processes at least one row, so more cores than rows is wasteful
    uint32_t num_cores_needed = std::min(max_cores_available, total_output_rows);

    // GRID LAYOUT OPTIMIZATION:
    // Arrange cores in optimal rectangular grid for memory access patterns
    uint32_t grid_x, grid_y;
    if (num_cores_needed <= compute_grid.x) {
        // Small workload: use single row of cores for simplicity
        grid_x = num_cores_needed;
        grid_y = 1;
    } else {
        // Large workload: use full grid width, calculate required height
        grid_x = compute_grid.x;
        grid_y = (num_cores_needed + grid_x - 1) / grid_x;  // Ceiling division
    }

    uint32_t num_cores = grid_x * grid_y;

    return {grid_x, grid_y, num_cores, max_cores_available, num_cores_needed};
}

/**
 * Prepare runtime arguments for multi-core slice kernels.
 * Handles both 4D kernels and maintains compatibility with 2D kernels.
 * Translated from Python get_multicore_slice_descriptor method.
 */
inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& slice_step,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t num_cores,
    uint32_t total_output_rows,
    const std::string& reader_kernel_path,
    const std::string& writer_kernel_path) {
    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();
    uint32_t element_size = input_tensor.element_size();

    // Extract dimensions for N-dimensional tensors
    uint32_t tensor_rank = input_shape.rank();

    // For backward compatibility, extract specific 4D dimensions using negative indexing
    // This avoids manual bounds checking since ttnn::Shape handles it automatically
    uint32_t input_n = input_shape[-4];  // Returns 1 if rank < 4
    uint32_t input_d = input_shape[-3];  // Returns 1 if rank < 3
    uint32_t input_h = input_shape[-2];  // Returns 1 if rank < 2
    uint32_t input_w = input_shape[-1];  // Always valid for rank >= 1

    uint32_t output_n = output_shape[-4];
    uint32_t output_d = output_shape[-3];
    uint32_t output_h = output_shape[-2];
    uint32_t output_w = output_shape[-1];

    uint32_t start_n = slice_start[-4];  // Returns 0 if rank < 4
    uint32_t start_d = slice_start[-3];  // Returns 0 if rank < 3
    uint32_t start_h = slice_start[-2];  // Returns 0 if rank < 2
    uint32_t start_w = slice_start[-1];  // Always valid

    uint32_t end_n = slice_end[-4];  // Returns appropriate default if rank < 4
    uint32_t end_d = slice_end[-3];  // Returns appropriate default if rank < 3
    uint32_t end_h = slice_end[-2];  // Returns appropriate default if rank < 2
    uint32_t end_w = slice_end[-1];  // Always valid

    uint32_t step_n = slice_step[-4];  // Returns 1 if rank < 4
    uint32_t step_d = slice_step[-3];  // Returns 1 if rank < 3
    uint32_t step_h = slice_step[-2];  // Returns 1 if rank < 2
    uint32_t step_w = slice_step[-1];  // Always valid

    // WORK DISTRIBUTION ALGORITHM:
    // Distribute rows as evenly as possible across cores
    // Some cores may get one extra row to handle remainder
    uint32_t base_rows_per_core = total_output_rows / num_cores;  // Minimum rows per core
    uint32_t extra_rows = total_output_rows % num_cores;          // Remainder rows to distribute

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

    uint32_t row_start_id = 0;
    uint32_t extra_rows_remaining = extra_rows;

    // Determine kernel type
    bool using_4d_kernels = (reader_kernel_path.find("4d") != std::string::npos);

    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        // Calculate work distribution for this core
        uint32_t rows_for_this_core = base_rows_per_core;
        if (extra_rows_remaining > 0) {
            rows_for_this_core += 1;
            extra_rows_remaining -= 1;
        }

        // Prepare runtime arguments based on kernel type
        std::vector<uint32_t> reader_args, writer_args;

        if (using_4d_kernels) {
            // 4D kernels expect 25 arguments for reader, 9 for writer
            reader_args = {
                input_tensor.buffer()->address(),  // 0: src_addr - input tensor memory address
                tensor_rank,                       // 1: tensor_rank - 1D, 2D, 3D, or 4D
                input_w,                           // 2: input_w - input width
                input_h,                           // 3: input_h - input height
                input_d,                           // 4: input_d - input depth
                input_n,                           // 5: input_n - input batch
                output_w,                          // 6: output_w - output width
                output_h,                          // 7: output_h - output height
                output_d,                          // 8: output_d - output depth
                output_n,                          // 9: output_n - output batch
                start_w,                           // 10: slice_start_w - start index for width
                end_w,                             // 11: slice_end_w - end index for width
                step_w,                            // 12: slice_step_w - step size for width
                start_h,                           // 13: slice_start_h - start index for height
                end_h,                             // 14: slice_end_h - end index for height
                step_h,                            // 15: slice_step_h - step size for height
                start_d,                           // 16: slice_start_d - start index for depth
                end_d,                             // 17: slice_end_d - end index for depth
                step_d,                            // 18: slice_step_d - step size for depth
                start_n,                           // 19: slice_start_n - start index for batch
                end_n,                             // 20: slice_end_n - end index for batch
                step_n,                            // 21: slice_step_n - step size for batch
                element_size,                      // 22: element_size - bytes per element
                rows_for_this_core,                // 23: num_rows - rows for this core
                row_start_id                       // 24: start_row - starting row for this core
            };

            writer_args = {
                output_tensor.buffer()->address(),  // 0: dst_addr - output tensor memory address
                tensor_rank,                        // 1: tensor_rank - 1D, 2D, 3D, or 4D
                output_w,                           // 2: output_w - output width
                output_h,                           // 3: output_h - output height
                output_d,                           // 4: output_d - output depth
                output_n,                           // 5: output_n - output batch
                element_size,                       // 6: element_size - bytes per element
                rows_for_this_core,                 // 7: num_rows - rows for this core
                row_start_id                        // 8: start_row - starting row for this core
            };
        } else {
            // N-dimensional kernels for 5D+ tensors
            // Reader arguments: src_addr, tensor_rank, element_size, num_rows, start_row, then dimension arrays
            reader_args = {
                input_tensor.buffer()->address(),  // 0: src_addr - input tensor memory address
                tensor_rank,                       // 1: tensor_rank - tensor dimensionality
                element_size,                      // 2: element_size - bytes per element
                rows_for_this_core,                // 3: num_rows - rows for this core
                row_start_id                       // 4: start_row - starting row for this core
            };
            // Append dimension arrays: input_dims, output_dims, slice_starts, slice_ends, slice_steps
            reader_args.insert(reader_args.end(), input_shape.cbegin(), input_shape.cend());
            reader_args.insert(reader_args.end(), output_shape.cbegin(), output_shape.cend());
            reader_args.insert(reader_args.end(), slice_start.cbegin(), slice_start.cend());
            reader_args.insert(reader_args.end(), slice_end.cbegin(), slice_end.cend());
            reader_args.insert(reader_args.end(), slice_step.cbegin(), slice_step.cend());

            // Writer arguments: dst_addr, tensor_rank, element_size, num_rows, start_row, then output_dims
            writer_args = {
                output_tensor.buffer()->address(),  // 0: dst_addr - output tensor memory address
                tensor_rank,                        // 1: tensor_rank - tensor dimensionality
                element_size,                       // 2: element_size - bytes per element
                rows_for_this_core,                 // 3: num_rows - rows for this core
                row_start_id                        // 4: start_row - starting row for this core
            };
            // Append output dimensions
            writer_args.insert(writer_args.end(), output_shape.cbegin(), output_shape.cend());
        }

        ret_val[core_idx] = {reader_args, writer_args};
        row_start_id += rows_for_this_core;
    }

    return ret_val;
}

/**
 * Main multi-core slice program factory.
 * Implements the intelligent multi-core slice strategy from kernel_bench.
 * Translated from Python MulticoreSlice.__call__ and get_multicore_slice_descriptor methods.
 */
operation::ProgramWithCallbacks slice_rm_multi_core_stride(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& slice_step) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::tt_metal::IDevice* device = input_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();
    uint32_t element_size = input_tensor.element_size();

    // Calculate total output rows based on tensor rank - this is what we distribute across cores
    uint32_t total_output_rows = calculate_total_output_rows(output_shape);

    // MULTI-CORE ALLOCATION STRATEGY:
    // The key insight is to use only as many cores as we have work for
    // This avoids the overhead of idle cores and optimizes resource utilization
    auto core_allocation = calculate_core_allocation(device, total_output_rows);
    uint32_t grid_x = core_allocation.grid_x;
    uint32_t grid_y = core_allocation.grid_y;
    uint32_t num_cores = core_allocation.num_cores;

    // DEBUG OUTPUT: Show core allocation decisions for performance analysis
    log_debug(
        tt::LogOp,
        "Multi-core slice allocation - Total output rows: {}, Available cores: {}",
        total_output_rows,
        core_allocation.max_cores_available);
    log_debug(tt::LogOp, "Tensor shape: {}D {}", input_shape.rank(), output_shape);
    log_debug(tt::LogOp, "Using grid: {}x{} = {} cores", grid_x, grid_y, num_cores);

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {grid_x - 1, grid_y - 1};
    CoreRange full_grid_range(start_core, end_core);
    CoreRangeSet core_grid({full_grid_range});

    // Select kernels based on tensor rank for optimal performance
    std::string reader_kernel_path, writer_kernel_path;
    if (input_shape.rank() <= 4) {
        // Use optimized 4D kernels for 1D-4D tensors
        reader_kernel_path =
            "ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp";
        writer_kernel_path =
            "ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_4d.cpp";
    } else {
        // Use generic N-dimensional kernels for 5D+ tensors
        reader_kernel_path =
            "ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp";
        writer_kernel_path =
            "ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_nd.cpp";
    }

    // Circular buffer configuration - same as single-core but for multiple cores
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t actual_output_w = output_shape[-1];
    uint32_t output_bytes_per_row = actual_output_w * element_size;
    uint32_t cb_page_size = output_bytes_per_row;

    // Align to 32-byte boundaries for optimal DRAM memory controller performance
    auto src_buffer_alignment = input_tensor.buffer()->alignment();
    auto dst_buffer_alignment = output_tensor.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    uint32_t cb_page_size_aligned = tt::round_up(cb_page_size, alignment);
    // Double buffering allows continuous data flow (producer/consumer overlap)
    uint32_t cb_total_size = 2 * cb_page_size_aligned;

    // Circular buffer indices following TTNN conventions
    constexpr uint32_t in_cb = 0;  // Input data buffer (reader -> writer)

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_total_size, {{in_cb, cb_data_format}})
            .set_page_size(in_cb, cb_page_size_aligned);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, cb_src0_config);

    // Prepare kernel compilation arguments using TensorAccessor
    std::vector<uint32_t> reader_compile_time_args = {in_cb, element_size};
    TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {in_cb, element_size};
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

    // Create kernels
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, core_grid, tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_path, core_grid, tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Get runtime arguments for all cores
    auto all_runtime_args = get_slice_runtime_args(
        input_tensor,
        output_tensor,
        slice_start,
        slice_end,
        slice_step,
        grid_x,
        grid_y,
        num_cores,
        total_output_rows,
        reader_kernel_path,
        writer_kernel_path);

    // Set runtime arguments for each core
    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        uint32_t x = core_idx % grid_x;
        uint32_t y = core_idx / grid_x;
        CoreCoord core = {x, y};

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[core_idx].first);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[core_idx].second);
    }

    // Runtime arguments override callback for buffer address updates
    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, grid_x, grid_y, num_cores](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        const auto& src_tensor = input_tensors.at(0);
        const auto& dst_tensor = output_tensors.at(0);

        // Update buffer addresses in runtime arguments for all cores
        for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
            uint32_t x = core_idx % grid_x;
            uint32_t y = core_idx / grid_x;
            CoreCoord core = {x, y};

            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
            reader_runtime_args[0] = src_tensor.buffer()->address();

            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
            writer_runtime_args[0] = dst_tensor.buffer()->address();
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
