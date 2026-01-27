// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_width_sharded_program_factory.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeMultiCoreWidthShardedProgramFactory::cached_program_t TilizeMultiCoreWidthShardedProgramFactory::create(
    const ttnn::prim::TilizeParams& /*operation_attributes*/,
    const ttnn::prim::TilizeInputs& tensor_args,
    const Tensor& output_tensor) {
    tt::tt_metal::Program program{};
    auto input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    bool fp32_llk_acc = input.dtype() == DataType::FLOAT32;

    auto shard_spec = input.shard_spec().value();
    uint32_t num_tiles_per_shard = shard_spec.shape[0] * shard_spec.shape[1] / TILE_HW;
    uint32_t num_tiles_per_row = shard_spec.shape[1] / TILE_WIDTH;
    auto all_cores = shard_spec.grid;

    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_cores,
        input_single_tile_size,
        num_tiles_per_shard,
        input_cb_data_format,
        input.buffer());

    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        all_cores,
        output_single_tile_size,
        num_tiles_per_shard,
        output_cb_data_format,
        output.buffer());

    // Calculate row width in bytes
    uint32_t row_width = shard_spec.shape[1] * input.element_size();

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index, row_width};

    // Add TensorAccessor compile-time args for the input tensor
    auto input_tensor_args = TensorAccessorArgs<2>();
    tt::tt_metal::add_tensor_accessor_args(reader_compile_time_args, input, input_tensor_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_width_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args = {
        uint32_t(num_tiles_per_shard / num_tiles_per_row), uint32_t(num_tiles_per_row)};

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_llk_acc,
            .compile_args = compute_args,
        });

    // Calculate runtime arguments for reader kernel
    uint32_t src_addr = input.buffer()->address();                 // correct ?
    uint32_t height_per_core = shard_spec.shape[0] / TILE_HEIGHT;  // Number of tile-height rows per core
    uint32_t start_chunk_id = 0;                                   // Will need to be updated per core if needed

    std::vector<uint32_t> reader_runtime_args = {src_addr, height_per_core, num_tiles_per_row, start_chunk_id};

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, reader_runtime_args);
    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, all_cores, {num_tiles_per_shard});
    return TilizeMultiCoreWidthShardedProgramFactory::cached_program_t{
        std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cb_output}};
}

void TilizeMultiCoreWidthShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::TilizeParams& /*operation_attributes*/,
    const ttnn::prim::TilizeInputs& tensor_args,
    const Tensor& output_tensor) {
    auto& input = tensor_args.input_tensor;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output_tensor.buffer();

    // Update circular buffer addresses
    UpdateDynamicCircularBufferAddress(
        cached_program.program, cached_program.shared_variables.input_cb_handle, *src_buffer);
    UpdateDynamicCircularBufferAddress(
        cached_program.program, cached_program.shared_variables.output_cb_handle, *dst_buffer);

    // Update reader kernel runtime arguments
    auto shard_spec = input.shard_spec().value();
    uint32_t num_tiles_per_row = shard_spec.shape[1] / TILE_WIDTH;
    uint32_t src_addr = src_buffer->address();
    uint32_t height_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t start_chunk_id = 0;

    std::vector<uint32_t> reader_runtime_args = {src_addr, height_per_core, num_tiles_per_row, start_chunk_id};

    auto all_cores = shard_spec.grid;
    tt::tt_metal::SetRuntimeArgs(
        cached_program.program, cached_program.shared_variables.unary_reader_kernel_id, all_cores, reader_runtime_args);
}
}  // namespace ttnn::prim
