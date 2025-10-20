// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/circular_buffer_config.hpp>
#include "ema_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::operations::reduction::accumulation {

constexpr auto ema_buffer_depth = 2;

tt::tt_metal::operation::ProgramWithCallbacks ema_multi_core(
    const Tensor& a,
    Tensor& output,
    float alpha,
    CoreCoord grid_size,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    // Grid sizing
    // ------------
    // If empty grid size, use all cores
    if ((grid_size.x == 0) && (grid_size.y == 0)) {
        grid_size = a.device()->compute_with_storage_grid_size();
    }
    auto num_cores_available = grid_size.x * grid_size.y;

    // Compute total_tiles to determine core split
    auto a_shape = a.padded_shape();
    auto num_batches = a_shape[1];
    auto num_channels = a_shape[2];

    auto num_channel_tiles = num_channels / a.tensor_spec().tile().get_height();
    auto tiles_per_channel = a_shape[3] / a.tensor_spec().tile().get_width();

    auto total_tiles = num_batches * num_channel_tiles * tiles_per_channel;

    // We pick the maximum number of cores (from the available) that divides total_tiles equally
    uint32_t num_cores = num_cores_available;
    while ((num_cores > 1) && (total_tiles % num_cores != 0)) {
        --num_cores;
    }

    // We now have the number of cores to use, compute per core parameters
    auto all_cores = CoreRangeSet(grid_to_cores(num_cores, grid_size.x, grid_size.y, false));
    log_warning(tt::LogOp, "Using {} cores out of {} available", num_cores, num_cores_available);
    log_warning(tt::LogOp, "Provided grid size: {}x{}", grid_size.x, grid_size.y);

    auto total_tiles_per_core = total_tiles / num_cores;
    auto total_batches_per_core = total_tiles_per_core / tiles_per_channel;

    // Premcompute the alpha and beta bits
    // Used by the EMA SFPU instructions
    // ----------------------------------
    auto alpha_bits = std::bit_cast<uint32_t>(alpha);
    auto beta_bits = std::bit_cast<uint32_t>(1.0f - alpha);

    // Create program
    // --------------
    auto program = Program();

    // Circular buffer config
    // ----------------------
    auto src_cb_index = tt::CBIndex::c_0;
    auto dst_cb_index = tt::CBIndex::c_1;
    auto prev_cb_index = tt::CBIndex::c_2;

    auto src_data_format = datatype_to_dataformat_converter(a.dtype());
    auto dst_data_format = datatype_to_dataformat_converter(output.dtype());

    auto src_tile_size = a.tensor_spec().tile().get_tile_size(src_data_format);
    auto dst_tile_size = output.tensor_spec().tile().get_tile_size(dst_data_format);

    auto src_cb_size = src_tile_size * ema_buffer_depth;
    auto dst_cb_size = dst_tile_size * ema_buffer_depth;
    auto prev_cb_size = src_tile_size;

    auto src_cb_cfg = tt::tt_metal::CircularBufferConfig(src_cb_size, {{src_cb_index, src_data_format}})
                          .set_page_size(src_cb_index, src_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, src_cb_cfg);

    auto dst_cb_cfg = tt::tt_metal::CircularBufferConfig(dst_cb_size, {{dst_cb_index, dst_data_format}})
                          .set_page_size(dst_cb_index, dst_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, dst_cb_cfg);

    auto prev_cb_cfg = tt::tt_metal::CircularBufferConfig(prev_cb_size, {{prev_cb_index, src_data_format}})
                           .set_page_size(prev_cb_index, src_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, prev_cb_cfg);

    // Compile time args for the kernels
    // ---------------------------------
    std::vector<uint32_t> reader_compile_args = {total_tiles_per_core, src_tile_size};
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {total_tiles_per_core, dst_tile_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_args);

    std::vector<uint32_t> compute_compile_args = {
        total_batches_per_core,
        tiles_per_channel,
        alpha_bits,
        beta_bits,
    };

    // Create kernels
    // --------------
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_reader.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_compile_args});

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_writer.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), compute_kernel_config);
    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/compute/ema_compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_args});

    // Set runtime args
    // ---------------
    std::vector<uint32_t> reader_runtime_args = {
        a.buffer()->address(),
        0,  // Placeholder for src_start_tile
    };
    std::vector<uint32_t> writer_runtime_args = {
        output.buffer()->address(),
        0,  // Placeholder for dst_start_tile
    };

    uint32_t src_start_tile = 0;
    uint32_t dst_start_tile = 0;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            reader_runtime_args[1] = src_start_tile;
            writer_runtime_args[1] = dst_start_tile;
            src_start_tile += total_tiles_per_core;
            dst_start_tile += total_tiles_per_core;

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }
    }

    // runtime_args_callback: If only the tensor changes, invoke this callback and skip the rest
    // -----------------------------------------------------------------------------------------
    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, all_cores](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer()->address();
        auto dst_buffer = output_tensors.at(0).buffer()->address();

        // Update buffer addresses for all cores
        for (const auto& range : all_cores.ranges()) {
            for (const auto& core : range) {
                auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                reader_runtime_args[0] = src_buffer;
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                writer_runtime_args[0] = dst_buffer;
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::reduction::accumulation
