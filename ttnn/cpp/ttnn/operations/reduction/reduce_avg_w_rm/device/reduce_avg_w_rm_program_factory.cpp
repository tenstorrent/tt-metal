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

    // Get shape dimensions
    const auto& input_shape = input.logical_shape();
    uint32_t N = input_shape[0];
    uint32_t C = input_shape[1];
    uint32_t H = input_shape[2];
    uint32_t W = input_shape[3];

    // Calculate tile dimensions
    uint32_t Ht = H / 32;  // Height in tiles
    uint32_t Wt = W / 32;  // Width in tiles

    // Data formats
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t input_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_tile_size = tt::tile_size(output_cb_data_format);

    // Work distribution: work unit is one tile row
    uint32_t NC = N * C;
    uint32_t num_tile_rows = NC * Ht;

    tt::tt_metal::IDevice* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tile_rows);

    // Create circular buffers according to spec

    // CB c_0: CB_rm_in - Raw row-major input sticks (double-buffered, 2 * Wt tiles)
    uint32_t cb_rm_in_idx = tt::CBIndex::c_0;
    uint32_t num_rm_in_tiles = 2 * Wt;  // Double-buffered, Wt tiles per block
    tt::tt_metal::CircularBufferConfig cb_rm_in_config =
        tt::tt_metal::CircularBufferConfig(num_rm_in_tiles * input_tile_size, {{cb_rm_in_idx, input_cb_data_format}})
            .set_page_size(cb_rm_in_idx, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_rm_in_config);

    // CB c_1: CB_tilized - Tilized intermediate data
    // Must hold Wt tiles since tilize pushes all at once before reduce streams them
    uint32_t cb_tilized_idx = tt::CBIndex::c_1;
    uint32_t num_tilized_tiles = Wt;  // Sized to hold full tilize output
    tt::tt_metal::CircularBufferConfig cb_tilized_config =
        tt::tt_metal::CircularBufferConfig(
            num_tilized_tiles * input_tile_size, {{cb_tilized_idx, input_cb_data_format}})
            .set_page_size(cb_tilized_idx, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_tilized_config);

    // CB c_2: CB_scaler - Scaler tile (1/W) (single-buffered, 1 tile)
    uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    uint32_t num_scaler_tiles = 1;  // Single-buffered, persists for entire program
    tt::tt_metal::CircularBufferConfig cb_scaler_config =
        tt::tt_metal::CircularBufferConfig(num_scaler_tiles * input_tile_size, {{cb_scaler_idx, input_cb_data_format}})
            .set_page_size(cb_scaler_idx, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    // CB c_3: CB_reduced - Reduced tiled data (double-buffered, 2 tiles)
    uint32_t cb_reduced_idx = tt::CBIndex::c_3;
    uint32_t num_reduced_tiles = 2;  // Double-buffered
    tt::tt_metal::CircularBufferConfig cb_reduced_config =
        tt::tt_metal::CircularBufferConfig(
            num_reduced_tiles * output_tile_size, {{cb_reduced_idx, output_cb_data_format}})
            .set_page_size(cb_reduced_idx, output_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_reduced_config);

    // CB c_16: CB_rm_out - Row-major output sticks (double-buffered, 2 tiles)
    uint32_t cb_rm_out_idx = tt::CBIndex::c_16;
    uint32_t num_rm_out_tiles = 2;  // Double-buffered
    tt::tt_metal::CircularBufferConfig cb_rm_out_config =
        tt::tt_metal::CircularBufferConfig(
            num_rm_out_tiles * output_tile_size, {{cb_rm_out_idx, output_cb_data_format}})
            .set_page_size(cb_rm_out_idx, output_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_rm_out_config);

    // Get buffers
    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    // Calculate compile-time args
    uint32_t stick_size = W * input.element_size();           // Size of one input row in bytes
    uint32_t output_stick_size = 32 * output.element_size();  // Size of one output row in bytes (width=32)

    // Pack scaler value (1/W) as two bfloat16 values
    float scaler_value = 1.0f / static_cast<float>(W);
    bfloat16 bfloat_scaler(scaler_value);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler, bfloat_scaler});

    // Compile-time args for reader
    std::vector<uint32_t> reader_compile_time_args = {stick_size, packed_scaler_value};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    // Compile-time args for writer
    std::vector<uint32_t> writer_compile_time_args = {output_stick_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Compile-time args for compute (per core group)
    std::vector<uint32_t> compute_compile_time_args_group_1 = {Wt};
    std::vector<uint32_t> compute_compile_time_args_group_2 = {Wt};

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

    // Enable FP32 accumulation for FLOAT32 inputs
    bool fp32_dest_acc_en = (input.dtype() == DataType::FLOAT32);

    // Create compute kernel for core_group_1
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::KernelHandle();
    if (!core_group_1.ranges().empty()) {
        compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/reduce_avg_w_rm/device/kernels/compute/reduce_avg_w_rm_compute.cpp",
            core_group_1,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_compile_time_args_group_1});
    }

    // Create compute kernel for core_group_2 (if it has cores)
    if (!core_group_2.ranges().empty()) {
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/reduce_avg_w_rm/device/kernels/compute/reduce_avg_w_rm_compute.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_compile_time_args_group_2});
    }

    // Set runtime args for each core
    uint32_t tile_row_idx = 0;

    // Iterate through all_cores and set runtime args
    for (const auto& core_range : all_cores.ranges()) {
        for (uint32_t x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (uint32_t y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord core = {x, y};

                // Determine which core group this core belongs to
                uint32_t num_tile_rows_per_core =
                    core_group_1.contains(core) ? num_tile_rows_per_core_group_1 : num_tile_rows_per_core_group_2;

                // Reader runtime args: src_addr, num_tile_rows, start_tile_row
                tt::tt_metal::SetRuntimeArgs(
                    program, reader_kernel_id, core, {src_buffer->address(), num_tile_rows_per_core, tile_row_idx});

                // Compute runtime args: num_tile_rows
                tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {num_tile_rows_per_core});

                // Writer runtime args: dst_addr, num_tile_rows, start_tile_row
                tt::tt_metal::SetRuntimeArgs(
                    program, writer_kernel_id, core, {dst_buffer->address(), num_tile_rows_per_core, tile_row_idx});

                tile_row_idx += num_tile_rows_per_core;
            }
        }
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
    // Stub - update runtime arguments for cached program
    // This will update tensor buffer addresses when program is reused
    // Will be implemented in Stage 6 after kernels are created
}

}  // namespace ttnn::operations::reduction::reduce_avg_w_rm::program
