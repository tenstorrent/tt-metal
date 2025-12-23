// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_untilize_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::reduction::program {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

TilizeUntilizeProgramFactory::cached_program_t TilizeUntilizeProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;

    tt::tt_metal::Program program{};

    // Get buffers
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // Data formats
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t input_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_dest_acc_en = input.dtype() == DataType::FLOAT32;

    // Get tensor dimensions
    // Shape is [N, C, H, W]
    auto shape = input.padded_shape();
    uint32_t tensor_height = shape[-2];
    uint32_t tensor_width = shape[-1];

    // Calculate tiles
    uint32_t num_tiles_per_row = tensor_width / TILE_WIDTH;
    uint32_t num_tile_rows = tensor_height / TILE_HEIGHT;

    // Calculate batch dimensions (N * C)
    uint32_t batch_size = 1;
    if (shape.rank() >= 3) {
        batch_size = shape[-3];
    }
    if (shape.rank() >= 4) {
        batch_size *= shape[-4];
    }

    // Total number of tile blocks (each block = 1 tile row = 32 rows × width)
    uint32_t num_blocks = num_tile_rows * batch_size;

    // Stick (row) dimensions
    uint32_t stick_size = tensor_width * input.element_size();
    uint32_t num_sticks = (tensor_height * batch_size);

    // Device and core setup - single core for template simplicity
    CoreCoord core = {0, 0};
    CoreRange core_range(core, core);
    CoreRangeSet all_cores(core_range);

    // ============================================================
    // CIRCULAR BUFFER CREATION (from spec "Circular Buffer Requirements" table)
    // CB_in (c_0): Row-major input staging - num_tiles_per_row tiles
    // CB_tiled (c_1): Tiled intermediate - num_tiles_per_row tiles
    // CB_out (c_16): Row-major output staging - num_tiles_per_row tiles
    // ============================================================

    // CB_in (c_0): Input circular buffer for row-major data from reader
    uint32_t cb_in_id = tt::CBIndex::c_0;
    uint32_t cb_in_num_tiles = num_tiles_per_row;
    tt::tt_metal::CircularBufferConfig cb_in_config =
        tt::tt_metal::CircularBufferConfig(cb_in_num_tiles * input_tile_size, {{cb_in_id, input_cb_data_format}})
            .set_page_size(cb_in_id, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_config);

    // CB_tiled (c_1): Intermediate circular buffer for tiled data
    uint32_t cb_tiled_id = tt::CBIndex::c_1;
    uint32_t cb_tiled_num_tiles = num_tiles_per_row;
    tt::tt_metal::CircularBufferConfig cb_tiled_config =
        tt::tt_metal::CircularBufferConfig(cb_tiled_num_tiles * input_tile_size, {{cb_tiled_id, input_cb_data_format}})
            .set_page_size(cb_tiled_id, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_tiled_config);

    // CB_out (c_16): Output circular buffer for row-major data to writer
    uint32_t cb_out_id = tt::CBIndex::c_16;
    uint32_t cb_out_num_tiles = num_tiles_per_row;
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(cb_out_num_tiles * output_tile_size, {{cb_out_id, output_cb_data_format}})
            .set_page_size(cb_out_id, output_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    // ============================================================
    // KERNEL CREATION (Stage 6)
    // ============================================================

    // Compile-time args for reader kernel
    std::vector<uint32_t> reader_compile_time_args = {stick_size};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    // Compile-time args for writer kernel
    std::vector<uint32_t> writer_compile_time_args = {cb_out_id, stick_size, TILE_HEIGHT};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Compile-time args for compute kernel
    std::vector<uint32_t> compute_compile_time_args = {num_blocks, num_tiles_per_row};

    // Create reader kernel (RISCV_0 / BRISC / NOC0)
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/tilize_untilize/device/kernels/dataflow/"
        "reader_tilize_untilize_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Create writer kernel (RISCV_1 / NCRISC / NOC1)
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/tilize_untilize/device/kernels/dataflow/"
        "writer_tilize_untilize_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create compute kernel
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/tilize_untilize/device/kernels/compute/tilize_untilize_compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_compile_time_args});

    // Set runtime args
    // Reader: src_addr, num_sticks, start_stick_id
    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address(), num_sticks, 0});

    // Writer: dst_addr, num_blocks, start_stick_id
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, {dst_buffer->address(), num_blocks, 0});

    return {
        std::move(program),
        TilizeUntilizeSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .all_cores = all_cores,
            .num_cores = 1}};
}

void TilizeUntilizeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    const auto& input = tensor_args.input;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    // Update reader kernel runtime args (address at index 0)
    auto cores = corerange_to_cores(shared_vars.all_cores);
    for (const auto& core : cores) {
        {
            auto& runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, shared_vars.writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::reduction::program
