// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optional"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
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
 * Get element size in bytes for different TTNN data types.
 * Translated from Python _get_element_size method.
 */
inline uint32_t get_element_size_kb(const DataType& dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return 2;   // 16-bit brain floating point
        case DataType::FLOAT32: return 4;    // 32-bit IEEE floating point
        case DataType::INT32: return 4;      // 32-bit signed integer
        case DataType::UINT32: return 4;     // 32-bit unsigned integer
        case DataType::BFLOAT8_B: return 1;  // 8-bit brain floating point
        default: TT_THROW("Unsupported data type for KB slice operation");
    }
}

/**
 * Calculate total output rows based on tensor rank - this is what we distribute across cores.
 * Translated from Python get_multicore_slice_descriptor method.
 */
inline uint32_t calculate_total_output_rows_kb(const ttnn::Shape& output_shape) {
    auto rank = output_shape.rank();

    if (rank == 1) {
        return 1;
    } else if (rank == 2) {
        return output_shape[-2];  // output_h
    } else if (rank == 3) {
        return output_shape[-3] * output_shape[-2];  // output_d * output_h
    } else if (rank == 4) {
        return output_shape[-4] * output_shape[-3] * output_shape[-2];  // output_n * output_d * output_h
    } else {
        TT_THROW("KB slice operation supports only 1D-4D tensors");
    }
}

/**
 * Core allocation and work distribution for KB multi-core slice operation.
 * This implements the key insight from Python: use only as many cores as we have work for.
 * Translated from Python get_multicore_slice_descriptor method.
 */
inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> calculate_core_allocation_kb(
    tt::tt_metal::IDevice* device, uint32_t total_output_rows) {
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

    return std::make_tuple(grid_x, grid_y, num_cores, max_cores_available, num_cores_needed);
}

/**
 * Prepare runtime arguments for KB multi-core slice kernels.
 * Handles both 4D kernels and maintains compatibility with 2D kernels.
 * Translated from Python get_multicore_slice_descriptor method.
 */
inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args_kb(
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
    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();
    uint32_t element_size = get_element_size_kb(input_tensor.dtype());

    // Extract dimensions with proper defaults for lower-rank tensors
    uint32_t tensor_rank = input_shape.rank();

    // Input dimensions (padded to 4D)
    uint32_t input_n = (tensor_rank >= 4) ? input_shape[-4] : 1;
    uint32_t input_d = (tensor_rank >= 3) ? input_shape[-3] : 1;
    uint32_t input_h = (tensor_rank >= 2) ? input_shape[-2] : 1;
    uint32_t input_w = input_shape[-1];

    // Output dimensions (padded to 4D)
    uint32_t output_n = (tensor_rank >= 4) ? output_shape[-4] : 1;
    uint32_t output_d = (tensor_rank >= 3) ? output_shape[-3] : 1;
    uint32_t output_h = (tensor_rank >= 2) ? output_shape[-2] : 1;
    uint32_t output_w = output_shape[-1];

    // Slice parameters (padded to 4D)
    uint32_t start_n = (tensor_rank >= 4) ? slice_start[-4] : 0;
    uint32_t start_d = (tensor_rank >= 3) ? slice_start[-3] : 0;
    uint32_t start_h = (tensor_rank >= 2) ? slice_start[-2] : 0;
    uint32_t start_w = slice_start[-1];

    uint32_t end_n = (tensor_rank >= 4) ? slice_end[-4] : 1;
    uint32_t end_d = (tensor_rank >= 3) ? slice_end[-3] : 1;
    uint32_t end_h = (tensor_rank >= 2) ? slice_end[-2] : 1;
    uint32_t end_w = slice_end[-1];

    uint32_t step_n = (tensor_rank >= 4) ? slice_step[-4] : 1;
    uint32_t step_d = (tensor_rank >= 3) ? slice_step[-3] : 1;
    uint32_t step_h = (tensor_rank >= 2) ? slice_step[-2] : 1;
    uint32_t step_w = slice_step[-1];

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
            // Original 2D multicore kernels for backward compatibility
            reader_args = {
                input_tensor.buffer()->address(),  // 0: src_addr - input tensor memory address
                input_h,                           // 1: input_h - input height in elements
                input_w,                           // 2: input_w - input width in elements
                output_h,                          // 3: output_h - output height in elements
                output_w,                          // 4: output_w - output width in elements
                start_h,                           // 5: slice_start_h - start index for height
                end_h,                             // 6: slice_end_h - end index for height
                step_h,                            // 7: slice_step_h - step size for height
                start_w,                           // 8: slice_start_w - start index for width
                end_w,                             // 9: slice_end_w - end index for width
                step_w,                            // 10: slice_step_w - step size for width
                element_size,                      // 11: element_size - bytes per element
                rows_for_this_core,                // 12: num_rows - rows for this core
                row_start_id                       // 13: start_row - starting row for this core
            };

            writer_args = {
                output_tensor.buffer()->address(),  // 0: dst_addr - output tensor memory address
                output_h,                           // 1: output_h - output height in elements
                output_w,                           // 2: output_w - output width in elements
                element_size,                       // 3: element_size - bytes per element
                rows_for_this_core,                 // 4: num_rows - rows for this core
                row_start_id                        // 5: start_row - starting row for this core
            };
        }

        ret_val[core_idx] = {reader_args, writer_args};
        row_start_id += rows_for_this_core;
    }

    return ret_val;
}

/**
 * Optimized KB multi-core slice program factory using production's approach
 */
operation::ProgramWithCallbacks slice_rm_multi_core_kb(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& slice_step) {
    std::cout << "LLONG slice_rm_multi_core_kb optimized" << std::endl;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::IDevice* device = input_tensor.device();

    // Calculate work distribution like production - use sticks instead of rows
    uint32_t num_unpadded_sticks = output_tensor.physical_volume() / output_tensor.padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    // Use production's work splitting
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    tt::tt_metal::Buffer* src0_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    constexpr uint32_t src0_cb_index = 0;

    // Use production's CB size calculation
    const auto [cb_page_size, num_read_per_barrier, misalignment] = compute_cb_size(
        input_tensor, output_tensor, slice_start, num_sticks_per_core_group_1, num_sticks_per_core_group_2);

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_read_per_barrier * 2 * cb_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    // Use production kernels instead of KB kernels
    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args_vec);

    std::vector<uint32_t> reader_compile_time_args_vec;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args_vec);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_writer_unary_stick_layout_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));

    // Use production's runtime args calculation
    auto all_runtime_args = get_slice_runtime_args_rm(
        input_tensor,
        output_tensor,
        slice_start,
        num_cores_total,
        num_cores,
        num_cores_y,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        MAX_READ_SIZE);

    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
    }

    // Production-style callback
    auto override_runtime_args_callback = [unary_reader_kernel_id,
                                           unary_writer_kernel_id,
                                           compute_with_storage_grid_size,
                                           src0_cb_index,
                                           cb_src0](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        const auto& src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t num_cores_total = num_cores_x * num_cores_y;
        uint32_t num_unpadded_sticks = dst_tensor.physical_volume() / dst_tensor.padded_shape()[-1];

        auto
            [num_cores,
             all_cores,
             core_group_1,
             core_group_2,
             num_sticks_per_core_group_1,
             num_sticks_per_core_group_2] =
                tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

        const auto tensor_start =
            static_cast<const ttnn::operations::data_movement::SliceDeviceOperation*>(operation)->slice_start;

        const auto [cb_page_size, num_read_per_barrier, misalignment] = compute_cb_size(
            src_tensor, dst_tensor, tensor_start, num_sticks_per_core_group_1, num_sticks_per_core_group_2);

        const uint32_t cb_size_bytes = num_read_per_barrier * 2 * cb_page_size;
        UpdateCircularBufferTotalSize(program, cb_src0, cb_size_bytes);
        UpdateCircularBufferPageSize(program, cb_src0, src0_cb_index, cb_page_size);

        auto all_runtime_args = get_slice_runtime_args_rm(
            src_tensor,
            dst_tensor,
            tensor_start,
            num_cores_total,
            num_cores,
            num_cores_y,
            core_group_1,
            core_group_2,
            num_sticks_per_core_group_1,
            num_sticks_per_core_group_2,
            MAX_READ_SIZE);

        for (uint32_t i = 0; i < num_cores_total; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            std::copy(all_runtime_args[i].first.begin(), all_runtime_args[i].first.end(), reader_runtime_args.data());
            auto& writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            std::copy(all_runtime_args[i].second.begin(), all_runtime_args[i].second.end(), writer_runtime_args.data());
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
