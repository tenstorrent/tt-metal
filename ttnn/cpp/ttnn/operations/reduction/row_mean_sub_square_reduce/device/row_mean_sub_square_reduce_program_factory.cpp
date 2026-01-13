// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "row_mean_sub_square_reduce_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/bfloat16.hpp>

namespace ttnn::operations::reduction::row_mean_sub_square_reduce::program {

using namespace tt;
using namespace tt::tt_metal;

RowMeanSubSquareReduceProgramFactory::cached_program_t RowMeanSubSquareReduceProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;

    tt::tt_metal::Program program{};

    // Data formats
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t tile_size = tt::tile_size(input_cb_data_format);

    // Shape extraction
    const auto& input_shape = input.logical_shape();
    uint32_t N = input_shape[0];
    uint32_t C = input_shape[1];
    uint32_t H = input_shape[2];
    uint32_t W = input_shape[3];

    // Tile dimensions
    uint32_t Ht = (H + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    uint32_t Wt = (W + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;

    // Calculate padded width (must be multiple of TILE_WIDTH=32)
    uint32_t padded_W = Wt * tt::constants::TILE_WIDTH;

    // Work distribution: parallelize over tile-rows
    uint32_t num_tile_rows = N * C * Ht;

    tt::tt_metal::IDevice* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tile_rows);

    // Create circular buffers from spec
    // CB c_0: Row-major input staging (sized for full row width)
    uint32_t cb_rm_in_idx = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_rm_in_config =
        tt::tt_metal::CircularBufferConfig(Wt * tile_size, {{cb_rm_in_idx, input_cb_data_format}})
            .set_page_size(cb_rm_in_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_rm_in_config);

    // CB c_1: Tilized input (sized for full row width)
    uint32_t cb_tilized_idx = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_tilized_config =
        tt::tt_metal::CircularBufferConfig(Wt * tile_size, {{cb_tilized_idx, input_cb_data_format}})
            .set_page_size(cb_tilized_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_tilized_config);

    // CB c_2: Scaler tile (1/W for mean computation)
    uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_scaler_config =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_scaler_idx, input_cb_data_format}})
            .set_page_size(cb_scaler_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    // CB c_3: Mean tile (single tile for broadcast subtraction)
    uint32_t cb_mean_idx = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_mean_config =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_mean_idx, input_cb_data_format}})
            .set_page_size(cb_mean_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_mean_config);

    // CB c_4: Intermediate (squared differences, sized for full row width)
    uint32_t cb_intermediate_idx = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(Wt * tile_size, {{cb_intermediate_idx, input_cb_data_format}})
            .set_page_size(cb_intermediate_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermediate_config);

    // CB c_5: Output tiled (variance tile before untilize)
    uint32_t cb_out_tiled_idx = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_out_tiled_config =
        tt::tt_metal::CircularBufferConfig(tile_size, {{cb_out_tiled_idx, output_cb_data_format}})
            .set_page_size(cb_out_tiled_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_tiled_config);

    // CB c_16: Row-major output staging (double-buffered)
    uint32_t cb_rm_out_idx = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_rm_out_config =
        tt::tt_metal::CircularBufferConfig(2 * tile_size, {{cb_rm_out_idx, output_cb_data_format}})
            .set_page_size(cb_rm_out_idx, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_rm_out_config);

    // Prepare kernel arguments
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    // Calculate stick sizes
    uint32_t input_stick_size = padded_W * input.element_size();
    uint32_t output_stick_size = tt::constants::TILE_WIDTH * output.element_size();  // Output width is always 32

    // Calculate scaler value: 1/W packed as two bfloat16 in uint32
    // Use padded_W for the scaler since that's what we're actually reducing
    float scaler_value = 1.0f / static_cast<float>(padded_W);
    bfloat16 scaler_bf16 = bfloat16(scaler_value);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({scaler_bf16, scaler_bf16});

    // Compile-time args for reader: stick_size, Wt, packed_scaler_value, TensorAccessorArgs
    std::vector<uint32_t> reader_compile_time_args = {input_stick_size, Wt, packed_scaler_value};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    // Compile-time args for writer: output_stick_size, TensorAccessorArgs
    std::vector<uint32_t> writer_compile_time_args = {output_stick_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Create reader kernel (RISCV_0 / BRISC)
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/row_mean_sub_square_reduce/device/kernels/dataflow/"
        "reader_row_mean_sub_square_reduce.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Create writer kernel (RISCV_1 / NCRISC)
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/row_mean_sub_square_reduce/device/kernels/dataflow/"
        "writer_row_mean_sub_square_reduce.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create compute kernel with per-core-group args
    // Compile args: Wt, num_rows_per_core
    std::vector<uint32_t> compute_args_group_1 = {Wt, num_rows_per_core_group_1};
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/row_mean_sub_square_reduce/device/kernels/compute/"
        "row_mean_sub_square_reduce_compute.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,  // Enable FP32 accumulation for better precision
            .compile_args = compute_args_group_1});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_group_2 = {Wt, num_rows_per_core_group_2};
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/row_mean_sub_square_reduce/device/kernels/compute/"
            "row_mean_sub_square_reduce_compute.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .compile_args = compute_args_group_2});
    }

    // Build cores vector for shared_variables
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    std::vector<CoreCoord> cores;
    for (uint32_t i = 0; i < num_cores; i++) {
        cores.push_back({i / num_cores_y, i % num_cores_y});
    }

    // Set runtime args for each core
    for (uint32_t i = 0, tile_rows_processed = 0; i < num_cores; i++) {
        CoreCoord core = cores[i];
        uint32_t num_rows_this_core =
            (i < core_group_1.num_cores()) ? num_rows_per_core_group_1 : num_rows_per_core_group_2;

        uint32_t start_tile_row = tile_rows_processed;

        // Reader runtime args: src_addr, num_tile_rows, start_tile_row
        tt::tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, {src_buffer->address(), num_rows_this_core, start_tile_row});

        // Writer runtime args: dst_addr, num_tile_rows, start_tile_row
        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {dst_buffer->address(), num_rows_this_core, start_tile_row});

        tile_rows_processed += num_rows_this_core;
    }

    return {
        std::move(program),
        RowMeanSubSquareReduceSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .all_cores = all_cores,
            .num_cores = num_cores}};
}

void RowMeanSubSquareReduceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;

    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& all_cores = cached_program.shared_variables.all_cores;

    const uint32_t src_addr = input.buffer()->address();
    const uint32_t dst_addr = output.buffer()->address();

    // Update buffer addresses for all cores
    for (const auto& core_range : all_cores.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord core = {x, y};
                {
                    auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src_addr;
                }
                {
                    auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = dst_addr;
                }
            }
        }
    }
}

}  // namespace ttnn::operations::reduction::row_mean_sub_square_reduce::program
