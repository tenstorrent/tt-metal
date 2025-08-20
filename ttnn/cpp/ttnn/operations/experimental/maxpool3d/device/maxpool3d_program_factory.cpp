// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "maxpool3d_device_operation.hpp"
#include "maxpool3d_program_factory.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <algorithm>

namespace ttnn::operations::experimental::maxpool3d::detail {

tt::tt_metal::operation::ProgramWithCallbacks maxpool3d_factory(
    const Tensor& input_tensor,
    const MaxPool3dConfig& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    auto grid_size = config.compute_with_storage_grid_size;
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();

    auto input_tensor_shape = input_tensor.logical_shape();
    uint32_t N = input_tensor_shape[0];
    uint32_t T_in = input_tensor_shape[1];
    uint32_t H_in = input_tensor_shape[2];
    uint32_t W_in = input_tensor_shape[3];
    uint32_t C = input_tensor_shape[4];

    auto [T_out, H_out, W_out] =
        detail::compute_output_dims(T_in, H_in, W_in, config.padding, config.stride, config.kernel_size);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto dtype_bytes = input_tensor.element_size();

    // Calculate window size and memory requirements
    uint32_t window_size = config.kernel_size[0] * config.kernel_size[1] * config.kernel_size[2];
    uint32_t channels_bytes = C * dtype_bytes;
    uint32_t window_bytes = window_size * channels_bytes;

    log_debug(tt::LogOp, "MaxPool3D configuration:");
    log_debug(tt::LogOp, "Input shape: N={}, T={}, H={}, W={}, C={}", N, T_in, H_in, W_in, C);
    log_debug(tt::LogOp, "Output shape: T={}, H={}, W={}, C={}", T_out, H_out, W_out, C);
    log_debug(tt::LogOp, "Kernel size: {}x{}x{}", config.kernel_size[0], config.kernel_size[1], config.kernel_size[2]);
    log_debug(tt::LogOp, "Stride: {}x{}x{}", config.stride[0], config.stride[1], config.stride[2]);
    log_debug(tt::LogOp, "Padding: {}x{}x{}", config.padding[0], config.padding[1], config.padding[2]);
    log_debug(tt::LogOp, "Window size: {} sticks", window_size);
    log_debug(tt::LogOp, "Channels: {}, channels_bytes: {}", C, channels_bytes);

    // Create circular buffers
    uint32_t next_cb_index = tt::CBIndex::c_0;

    // CB for input 3D window (reader -> compute)
    uint32_t cb_input_window_id = next_cb_index++;
    tt::tt_metal::create_cb(cb_input_window_id, program, core_grid, channels_bytes, window_size, data_format);

    // CB for output result (compute -> writer)
    uint32_t cb_output_id = next_cb_index++;
    tt::tt_metal::create_cb(cb_output_id, program, core_grid, channels_bytes, 1, data_format);

    log_debug(
        tt::LogOp,
        "CB input_window: id={}, page_size={} bytes, num_pages={}",
        cb_input_window_id,
        channels_bytes,
        window_size);
    log_debug(tt::LogOp, "CB output: id={}, page_size={} bytes, num_pages={}", cb_output_id, channels_bytes, 1);

    bool is_padding_zeros = config.padding_mode == "zeros";

    uint32_t in_page_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t out_page_size_bytes = output_tensor.buffer()->aligned_page_size();

    // Reader kernel compile-time args
    std::vector<uint32_t> reader_compile_time_args = {
        cb_input_window_id,
        N,
        T_in,
        H_in,
        W_in,
        C,
        T_out,
        H_out,
        W_out,
        config.padding[0],
        config.padding[1],
        config.padding[2],
        config.kernel_size[0],
        config.kernel_size[1],
        config.kernel_size[2],
        config.stride[0],
        config.stride[1],
        config.stride[2],
        in_page_size_bytes,
        channels_bytes,
        is_padding_zeros};

    // Compute kernel compile-time args
    std::vector<uint32_t> compute_compile_time_args = {
        cb_input_window_id,
        cb_output_id,
        config.kernel_size[0],
        config.kernel_size[1],
        config.kernel_size[2],
        C,
        1  // is_max_pool = true
    };

    // Writer kernel compile-time args
    std::vector<uint32_t> writer_compile_time_args = {cb_output_id, out_page_size_bytes, H_out, W_out};

    // Create kernels
    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/maxpool3d/device/kernels/reader_maxpool3d.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Get compute kernel config args
    auto device = input_tensor.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/maxpool3d/device/kernels/compute_maxpool3d.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = {}});

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/maxpool3d/device/kernels/writer_maxpool3d.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t input_addr = input_tensor.buffer()->address();
    uint32_t output_addr = output_tensor.buffer()->address();

    // Calculate optimal core usage based on output size
    uint32_t total_output_elements = T_out * H_out * W_out;
    uint32_t optimal_cores = std::min(static_cast<uint32_t>(num_cores), total_output_elements);

    // Calculate parallelization factors for T, H, W dimensions
    uint32_t t_parallel = 1, h_parallel = 1, w_parallel = 1;

    if (optimal_cores > 1) {
        // Try to distribute cores evenly across spatial dimensions
        // Priority: W > H > T (similar to Conv3D approach)

        // First, try parallelizing W dimension
        w_parallel = std::min(optimal_cores, W_out);
        uint32_t remaining_cores = optimal_cores / w_parallel;

        if (remaining_cores > 1) {
            // Then H dimension
            h_parallel = std::min(remaining_cores, H_out);
            remaining_cores = remaining_cores / h_parallel;

            if (remaining_cores > 1) {
                // Finally T dimension
                t_parallel = std::min(remaining_cores, T_out);
            }
        }
    }

    uint32_t actual_cores_used = t_parallel * h_parallel * w_parallel;

    log_info(
        tt::LogOp,
        "MaxPool3D Core allocation: {} total cores, {} optimal, {} used",
        num_cores,
        optimal_cores,
        actual_cores_used);
    log_info(tt::LogOp, "MaxPool3D Parallelization: T={}, H={}, W={}", t_parallel, h_parallel, w_parallel);

    // Set runtime args for used cores only
    for (uint32_t core_idx = 0; core_idx < actual_cores_used; ++core_idx) {
        CoreCoord core = {core_idx % grid_size.x, core_idx / grid_size.x};

        // Calculate work assignment for this core
        uint32_t w_idx = core_idx % w_parallel;
        uint32_t remaining = core_idx / w_parallel;
        uint32_t h_idx = remaining % h_parallel;
        uint32_t t_idx = remaining / h_parallel;

        // Calculate output ranges for this core
        uint32_t t_out_start = (t_idx * T_out) / t_parallel;
        uint32_t t_out_end = ((t_idx + 1) * T_out) / t_parallel;

        uint32_t h_out_start = (h_idx * H_out) / h_parallel;
        uint32_t h_out_end = ((h_idx + 1) * H_out) / h_parallel;

        uint32_t w_out_start = (w_idx * W_out) / w_parallel;
        uint32_t w_out_end = ((w_idx + 1) * W_out) / w_parallel;

        log_info(
            tt::LogOp,
            "MaxPool3D Core ({},{}): T=[{},{}), H=[{},{}), W=[{},{})",
            core.x,
            core.y,
            t_out_start,
            t_out_end,
            h_out_start,
            h_out_end,
            w_out_start,
            w_out_end);

        // Reader args
        std::vector<uint32_t> reader_args = {
            input_addr, t_out_start, t_out_end, h_out_start, h_out_end, w_out_start, w_out_end};

        // Writer args
        std::vector<uint32_t> writer_args = {
            output_addr, t_out_start, t_out_end, h_out_start, h_out_end, w_out_start, w_out_end};

        // Compute args - number of windows this core processes
        uint32_t windows_per_core = (t_out_end - t_out_start) * (h_out_end - h_out_start) * (w_out_end - w_out_start);
        std::vector<uint32_t> compute_args = {windows_per_core};

        SetRuntimeArgs(program, reader_kernels_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernels_id, core, writer_args);
        SetRuntimeArgs(program, compute_kernels_id, core, compute_args);
    }

    // Set dummy args for unused cores to avoid issues
    for (uint32_t i = actual_cores_used; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        std::vector<uint32_t> dummy_reader_args = {
            input_addr, 0, 0, 0, 0, 0, 0  // Empty ranges
        };

        std::vector<uint32_t> dummy_writer_args = {
            output_addr, 0, 0, 0, 0, 0, 0  // Empty ranges
        };

        std::vector<uint32_t> dummy_compute_args = {0};  // No windows to process

        SetRuntimeArgs(program, reader_kernels_id, core, dummy_reader_args);
        SetRuntimeArgs(program, writer_kernels_id, core, dummy_writer_args);
        SetRuntimeArgs(program, compute_kernels_id, core, dummy_compute_args);
    }

    auto override_runtime_arguments_callback =
        [actual_cores_used, grid_size, reader_kernels_id, writer_kernels_id](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
            auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);

            auto input_addr = input_tensors.at(0).buffer()->address();
            auto output_addr = output_tensors.at(0).buffer()->address();

            // Only update args for cores that are actually used
            for (uint32_t i = 0; i < actual_cores_used; ++i) {
                CoreCoord core = {i % grid_size.x, i / grid_size.x};

                auto& reader_args = reader_args_by_core[core.x][core.y];
                reader_args[0] = input_addr;
                // Keep the spatial ranges (args 1-6) unchanged

                auto& writer_args = writer_args_by_core[core.x][core.y];
                writer_args[0] = output_addr;
                // Keep the spatial ranges (args 1-6) unchanged
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::maxpool3d::detail
