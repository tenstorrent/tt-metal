// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rm_scaled_add_program_factory.hpp"
#include "rm_scaled_add_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/bfloat16.hpp>

namespace ttnn::experimental::prim {

RmScaledAddProgramFactory::cached_program_t RmScaledAddProgramFactory::create(
    const RmScaledAddParams& operation_attributes,
    const RmScaledAddInputs& tensor_args,
    Tensor& tensor_return_value) {

    tt::tt_metal::Program program{};

    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;

    // Get buffers
    auto* src0_buffer = input_a.buffer();
    auto* src1_buffer = input_b.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    // For RM tensors, buffer page_size is typically the row width
    // We read page-by-page using proper interleaved addressing
    const auto& input_shape = input_a.padded_shape();
    uint32_t total_elements = input_shape.volume();
    uint32_t total_bytes = total_elements * input_a.element_size();

    // Number of "tiles" for compute (each tile = 1024 elements = 2048 bytes for bf16)
    uint32_t n_tiles = total_elements / 1024;

    // Tile size in bytes for bfloat16 (for compute processing)
    constexpr uint32_t tile_size_bytes = 2048;  // 32 * 32 * 2 bytes

    // Get the actual buffer page size (for DRAM reads)
    uint32_t buffer_page_size = src0_buffer->page_size();
    uint32_t num_buffer_pages = total_bytes / buffer_page_size;

    // Data format
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_a.dtype());

    // Single core operation
    CoreCoord core = {0, 0};
    CoreRange core_range(core, core);

    // Create circular buffers
    // For this experimental op, we read entire tensor into L1, then process tile-by-tile
    // CBs need to hold all the data (n_tiles * tile_size_bytes)

    // cb_in0: Input A - holds all tiles
    uint32_t cb_in0_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_in0_config =
        tt::tt_metal::CircularBufferConfig(n_tiles * tile_size_bytes, {{cb_in0_index, data_format}})
            .set_page_size(cb_in0_index, tile_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range, cb_in0_config);

    // cb_in1: Input B - holds all tiles
    uint32_t cb_in1_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_in1_config =
        tt::tt_metal::CircularBufferConfig(n_tiles * tile_size_bytes, {{cb_in1_index, data_format}})
            .set_page_size(cb_in1_index, tile_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range, cb_in1_config);

    // cb_scalar: Scalar tile for broadcast - single buffered
    uint32_t cb_scalar_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_scalar_config =
        tt::tt_metal::CircularBufferConfig(tile_size_bytes, {{cb_scalar_index, data_format}})
            .set_page_size(cb_scalar_index, tile_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range, cb_scalar_config);

    // cb_out0: Output - holds all tiles
    uint32_t cb_out0_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_out0_config =
        tt::tt_metal::CircularBufferConfig(n_tiles * tile_size_bytes, {{cb_out0_index, data_format}})
            .set_page_size(cb_out0_index, tile_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range, cb_out0_config);

    // Pack scalar as bfloat16
    bfloat16 scalar_bf16(operation_attributes.scale);
    uint32_t packed_scalar = pack_two_bfloat16_into_uint32({scalar_bf16, scalar_bf16});

    // Determine buffer types
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool src1_is_dram = src1_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // Create reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        cb_in0_index,
        cb_in1_index,
        cb_scalar_index,
        src0_is_dram,
        src1_is_dram
    };

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rm_scaled_add/device/kernels/dataflow/reader_rm_scaled_add.cpp",
        core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Create writer kernel
    std::vector<uint32_t> writer_compile_time_args = {
        cb_out0_index,
        dst_is_dram
    };

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rm_scaled_add/device/kernels/dataflow/writer_rm_scaled_add.cpp",
        core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create compute kernel
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rm_scaled_add/device/kernels/compute/rm_scaled_add_compute.cpp",
        core_range,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    // Set runtime arguments
    tt::tt_metal::SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {src0_buffer->address(),
         src1_buffer->address(),
         n_tiles,
         packed_scalar,
         buffer_page_size,
         num_buffer_pages});

    tt::tt_metal::SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        {dst_buffer->address(),
         n_tiles,
         buffer_page_size,
         num_buffer_pages});

    tt::tt_metal::SetRuntimeArgs(
        program,
        compute_kernel_id,
        core,
        {n_tiles});

    return cached_program_t{
        std::move(program),
        {/* reader_kernel_id = */ reader_kernel_id,
         /* writer_kernel_id = */ writer_kernel_id,
         /* compute_kernel_id = */ compute_kernel_id,
         /* core = */ core}};
}

void RmScaledAddProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RmScaledAddParams& /* operation_attributes */,
    const RmScaledAddInputs& tensor_args,
    Tensor& tensor_return_value) {

    auto* src0_buffer = tensor_args.input_a.buffer();
    auto* src1_buffer = tensor_args.input_b.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    auto& program = cached_program.program;
    const auto& core = cached_program.shared_variables.core;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
    reader_runtime_args[0] = src0_buffer->address();
    reader_runtime_args[1] = src1_buffer->address();

    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
    writer_runtime_args[0] = dst_buffer->address();
}

}  // namespace ttnn::experimental::prim
