// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_avg_w_rm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/bfloat16.hpp>

namespace ttnn::operations::reduction::reduce_avg_w_rm::program {

using namespace tt;
using namespace tt::tt_metal;

ReduceAvgWRmProgramFactory::cached_program_t ReduceAvgWRmProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;

    tt::tt_metal::Program program{};

    // Data formats
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t input_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_tile_size = tt::tile_size(output_cb_data_format);

    // Extract shape dimensions
    const auto& input_shape = input.padded_shape();
    uint32_t N = input_shape[0];
    uint32_t C = input_shape[1];
    uint32_t H = input_shape[2];
    uint32_t W = input_shape[3];

    // Compute tile dimensions
    uint32_t Ht = H / constants::TILE_HEIGHT;
    uint32_t Wt = W / constants::TILE_WIDTH;

    // Work distribution - work unit is one row of tiles (one value of h)
    uint32_t num_tile_rows = N * C * Ht;

    tt::tt_metal::IDevice* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tile_rows);

    // Create circular buffers from spec's "Circular Buffer Requirements" table
    // CB c_0: Input row-major sticks (for stub, size it to hold 32 sticks worth of data)
    uint32_t cb_in_idx = tt::CBIndex::c_0;
    uint32_t stick_size = W * sizeof(uint16_t);
    uint32_t cb_in_size = constants::TILE_HEIGHT * stick_size;  // 32 sticks
    tt::tt_metal::CircularBufferConfig cb_in_config =
        tt::tt_metal::CircularBufferConfig(cb_in_size, {{cb_in_idx, input_cb_data_format}})
            .set_page_size(cb_in_idx, cb_in_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_config);

    // CB c_1: Intermediate tilized tiles (Wt tiles)
    uint32_t cb_tilized_idx = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_tilized_config =
        tt::tt_metal::CircularBufferConfig(Wt * input_tile_size, {{cb_tilized_idx, input_cb_data_format}})
            .set_page_size(cb_tilized_idx, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_tilized_config);

    // CB c_2: Scaler tile (1/W) - 1 tile
    uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_scaler_config =
        tt::tt_metal::CircularBufferConfig(1 * input_tile_size, {{cb_scaler_idx, input_cb_data_format}})
            .set_page_size(cb_scaler_idx, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    // CB c_3: Intermediate reduced tiles (1 tile per row)
    uint32_t cb_reduced_idx = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_reduced_config =
        tt::tt_metal::CircularBufferConfig(1 * output_tile_size, {{cb_reduced_idx, output_cb_data_format}})
            .set_page_size(cb_reduced_idx, output_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_reduced_config);

    // CB c_16: Output row-major sticks (for stub, same size as input for passthrough)
    uint32_t cb_out_idx = tt::CBIndex::c_16;
    uint32_t cb_out_size = constants::TILE_HEIGHT * (32 * sizeof(uint16_t));  // 32 sticks of width 32
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(cb_out_size, {{cb_out_idx, output_cb_data_format}})
            .set_page_size(cb_out_idx, cb_out_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    // Create kernels
    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    // Compute scaler value (1/W) - pack as two bfloat16 values into uint32
    // generate_reduce_scaler expects scaler as "16-bit value double packed into a u32"
    float scaler = 1.0f / static_cast<float>(W);
    bfloat16 bfloat_scaler_value = bfloat16::truncate(scaler);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    uint32_t input_stick_size = W * sizeof(uint16_t);    // bfloat16 = 2 bytes
    uint32_t output_stick_size = 32 * sizeof(uint16_t);  // Output width = 32

    // Compile-time args for reader
    std::vector<uint32_t> reader_compile_time_args = {packed_scaler_value, input_stick_size};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    // Compile-time args for writer
    std::vector<uint32_t> writer_compile_time_args = {cb_out_idx, output_stick_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Compile-time args for compute (group 1)
    std::vector<uint32_t> compute_args_group_1 = {num_rows_per_core_group_1, Wt};

    // Create reader kernel (RISCV_0 / BRISC)
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/reduce_avg_w_rm/device/kernels/dataflow/reader_reduce_avg_w_rm.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Create writer kernel (RISCV_1 / NCRISC)
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/reduce_avg_w_rm/device/kernels/dataflow/writer_reduce_avg_w_rm.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create compute kernel for core_group_1
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/reduce_avg_w_rm/device/kernels/compute/compute_reduce_avg_w_rm.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args_group_1});

    // Create compute kernel for core_group_2 if exists
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_group_2 = {num_rows_per_core_group_2, Wt};
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/reduce_avg_w_rm/device/kernels/compute/compute_reduce_avg_w_rm.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args_group_2});
    }

    // Set runtime args for each core
    uint32_t stick_offset = 0;

    // Set runtime args for core_group_1
    for (const auto& core_coord : core_group_1.ranges()) {
        for (uint32_t x = core_coord.start_coord.x; x <= core_coord.end_coord.x; x++) {
            for (uint32_t y = core_coord.start_coord.y; y <= core_coord.end_coord.y; y++) {
                CoreCoord core = {x, y};
                uint32_t num_sticks_per_core = num_rows_per_core_group_1 * constants::TILE_HEIGHT;

                // Reader runtime args: src_addr, num_sticks, start_stick_id
                tt::tt_metal::SetRuntimeArgs(
                    program, reader_kernel_id, core, {src_buffer->address(), num_sticks_per_core, stick_offset});

                // Compute runtime args: num_blocks
                tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {num_rows_per_core_group_1});

                // Writer runtime args: dst_addr, num_sticks, start_stick_id
                tt::tt_metal::SetRuntimeArgs(
                    program, writer_kernel_id, core, {dst_buffer->address(), num_sticks_per_core, stick_offset});

                stick_offset += num_sticks_per_core;
            }
        }
    }

    // Set runtime args for core_group_2 if it has cores
    if (!core_group_2.ranges().empty()) {
        for (const auto& core_coord : core_group_2.ranges()) {
            for (uint32_t x = core_coord.start_coord.x; x <= core_coord.end_coord.x; x++) {
                for (uint32_t y = core_coord.start_coord.y; y <= core_coord.end_coord.y; y++) {
                    CoreCoord core = {x, y};
                    uint32_t num_sticks_per_core = num_rows_per_core_group_2 * constants::TILE_HEIGHT;

                    // Reader runtime args: src_addr, num_sticks, start_stick_id
                    tt::tt_metal::SetRuntimeArgs(
                        program, reader_kernel_id, core, {src_buffer->address(), num_sticks_per_core, stick_offset});

                    // Compute runtime args: num_blocks
                    tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {num_rows_per_core_group_2});

                    // Writer runtime args: dst_addr, num_sticks, start_stick_id
                    tt::tt_metal::SetRuntimeArgs(
                        program, writer_kernel_id, core, {dst_buffer->address(), num_sticks_per_core, stick_offset});

                    stick_offset += num_sticks_per_core;
                }
            }
        }
    }

    // Build cores vector for shared_variables
    std::vector<CoreCoord> cores_vec;
    for (uint32_t i = 0; i < num_cores; i++) {
        cores_vec.push_back({i % compute_with_storage_grid_size.x, i / compute_with_storage_grid_size.x});
    }

    return {
        std::move(program),
        ReduceAvgWRmSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .all_cores = all_cores,
            .num_cores = num_cores}};
}

void ReduceAvgWRmProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    uint32_t num_cores = cached_program.shared_variables.num_cores;

    // Extract shape dimensions
    const auto& input_shape = input.padded_shape();
    uint32_t N = input_shape[0];
    uint32_t C = input_shape[1];
    uint32_t H = input_shape[2];
    uint32_t Ht = H / constants::TILE_HEIGHT;

    uint32_t num_tile_rows = N * C * Ht;

    tt::tt_metal::IDevice* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [_, __, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tile_rows);

    // Update runtime args for each core
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i % compute_with_storage_grid_size.x, i / compute_with_storage_grid_size.x};

        // Update reader runtime args
        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
        reader_runtime_args[0] = src_buffer->address();
        // Keep num_sticks and start_stick_id unchanged

        // Update writer runtime args
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        writer_runtime_args[0] = dst_buffer->address();
        // Keep num_sticks and start_stick_id unchanged
    }
}

}  // namespace ttnn::operations::reduction::reduce_avg_w_rm::program
