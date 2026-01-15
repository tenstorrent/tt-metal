// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "ema_program_factory.hpp"

#include "ttnn/operations/math.hpp"

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <bit>

namespace ttnn::operations::reduction::ema::program {

using namespace tt::tt_metal;

constexpr auto ema_buffer_depth = 2;

EmaProgramFactory::cached_program_t EmaProgramFactory::create(
    const EmaParams& operation_attributes, const EmaInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;

    // Grid sizing
    // -----------
    // If empty grid size, use all cores
    auto grid_size = operation_attributes.grid_size;
    if ((grid_size.x == 0) && (grid_size.y == 0)) {
        grid_size = input.device()->compute_with_storage_grid_size();
    }
    auto num_cores_available = grid_size.x * grid_size.y;

    // Compute total_tiles to determine core split
    auto input_shape = input.padded_shape();
    auto num_batches = input_shape[1];
    auto num_channels = input_shape[2];
    auto num_samples_per_channel = input_shape[3];

    auto num_channel_tiles = num_channels / input.tensor_spec().tile().get_height();
    auto tiles_per_channel = num_samples_per_channel / input.tensor_spec().tile().get_width();

    auto total_batch_channel_tiles = num_batches * num_channel_tiles;

    // We pick the maximum number of cores (from the available) that divides total_tiles equally
    auto [num_cores, total_batch_channel_tiles_per_core] = get_max_cores_divisible_by_tiles_per_core_tiles(
        total_batch_channel_tiles, num_cores_available, /*request_even=*/false);

    // We now have the number of cores to use, compute per core parameters
    auto all_cores = CoreRangeSet(grid_to_cores(num_cores, grid_size.x, grid_size.y, false));

    log_debug(
        tt::LogOp,
        "EmaProgramFactory: grid_size=({}, {}), num_cores={}, total_batch_channel_tiles={}",
        grid_size.y,
        grid_size.x,
        num_cores,
        total_batch_channel_tiles);

    auto total_tiles_per_core = total_batch_channel_tiles_per_core * tiles_per_channel;

    // Precompute the alpha and beta bits
    // Used by the EMA SFPU instructions
    // ----------------------------------
    auto alpha_bits = std::bit_cast<uint32_t>(operation_attributes.alpha);
    auto beta_bits = std::bit_cast<uint32_t>(1.0f - operation_attributes.alpha);

    // Create program
    // --------------
    auto program = Program();

    // Circular buffer config
    // ----------------------
    constexpr auto src_cb_index = tt::CBIndex::c_0;
    constexpr auto dst_cb_index = tt::CBIndex::c_1;
    constexpr auto prev_cb_index = tt::CBIndex::c_2;

    auto src_data_format = datatype_to_dataformat_converter(input.dtype());
    auto dst_data_format = datatype_to_dataformat_converter(output.dtype());

    auto src_tile_size = input.tensor_spec().tile().get_tile_size(src_data_format);
    auto dst_tile_size = output.tensor_spec().tile().get_tile_size(dst_data_format);

    auto src_cb_size = src_tile_size * ema_buffer_depth;
    auto dst_cb_size = dst_tile_size * ema_buffer_depth;
    auto prev_cb_size = src_tile_size;

    auto src_cb_cfg =
        CircularBufferConfig(src_cb_size, {{src_cb_index, src_data_format}}).set_page_size(src_cb_index, src_tile_size);
    CreateCircularBuffer(program, all_cores, src_cb_cfg);

    auto dst_cb_cfg =
        CircularBufferConfig(dst_cb_size, {{dst_cb_index, dst_data_format}}).set_page_size(dst_cb_index, dst_tile_size);
    CreateCircularBuffer(program, all_cores, dst_cb_cfg);

    auto prev_cb_cfg = CircularBufferConfig(prev_cb_size, {{prev_cb_index, src_data_format}})
                           .set_page_size(prev_cb_index, src_tile_size);
    CreateCircularBuffer(program, all_cores, prev_cb_cfg);

    // Compile time args for the kernels
    // ---------------------------------
    std::vector<uint32_t> reader_compile_args = {total_tiles_per_core};
    TensorAccessorArgs(input.buffer()).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {total_tiles_per_core};
    TensorAccessorArgs(output.buffer()).append_to(writer_compile_args);

    std::vector<uint32_t> compute_compile_args = {
        total_batch_channel_tiles_per_core,
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
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_args});

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_writer.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);
    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/compute/ema_compute.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_args});

    // Set runtime args
    // ---------------
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_runtime_args = {
        src_buffer->address(),
        0,  // Placeholder for src_start_tile, populated below
    };
    std::vector<uint32_t> writer_runtime_args = {
        dst_buffer->address(),
        0,  // Placeholder for dst_start_tile, populated below
    };

    uint32_t src_start_tile = 0;
    uint32_t dst_start_tile = 0;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            reader_runtime_args[1] = src_start_tile;
            writer_runtime_args[1] = dst_start_tile;
            src_start_tile += total_tiles_per_core;
            dst_start_tile += total_tiles_per_core;

            SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .all_cores = std::move(all_cores)}};
}

void EmaProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const EmaParams& /*operation_attributes*/,
    const EmaInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_variables = cached_program.shared_variables;

    auto src_buffer_address = tensor_args.input.buffer()->address();
    auto dst_buffer_address = tensor_return_value.buffer()->address();

    // Update buffer addresses for all cores
    for (const auto& range : shared_variables.all_cores.ranges()) {
        for (const auto& core : range) {
            GetRuntimeArgs(program, shared_variables.reader_kernel_id, core)[0] = src_buffer_address;
            GetRuntimeArgs(program, shared_variables.writer_kernel_id, core)[0] = dst_buffer_address;
        }
    }
}

}  // namespace ttnn::operations::reduction::ema::program
