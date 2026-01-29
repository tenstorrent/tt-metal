// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_single_core_sharded_program_factory.hpp"

#include <cmath>
#include "ttnn/operations/cb_utils.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeWithValPaddingSingleCoreShardedFactory::cached_program_t TilizeWithValPaddingSingleCoreShardedFactory::create(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor, const Tensor& output_tensor) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const Tensor& input = input_tensor;
    const Tensor& output = output_tensor;
    auto pad_value = operation_attributes.pad_value;

    // Data formats and tile sizes
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tile_size(output_cb_data_format);

    bool fp32_llk_acc = input.dtype() == DataType::FLOAT32;

    // Dimensions
    auto input_shape = input.padded_shape();
    auto output_shape = output.padded_shape();

    uint32_t input_height = input_shape[-2];
    uint32_t input_width = input_shape[-1];
    uint32_t output_height = output_shape[-2];
    uint32_t output_width = output_shape[-1];

    uint32_t num_batches = output.physical_volume() / (output_height * output_width);

    // Tile calculations
    uint32_t tiles_per_row = output_width / TILE_WIDTH;
    uint32_t tile_rows = output_height / TILE_HEIGHT;
    uint32_t total_tiles = tiles_per_row * tile_rows * num_batches;

    // Pick first core from output shard grid for single-core execution
    auto shard_grid = output.shard_spec().value().grid;
    CoreCoord core = corerange_to_cores(shard_grid).at(0);
    CoreRange core_range(core, core);

    // Create CBs.
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0,
        program,
        core_range,
        input_single_tile_size,
        tiles_per_row * 2,  // Double buffer
        input_cb_data_format);

    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        core_range,
        output_single_tile_size,
        tiles_per_row * 2,  // Double buffer
        output_cb_data_format);

    // Prepare compile-time arguments for reader with ShardedAddrGen
    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(src0_cb_index),
        static_cast<uint32_t>(input.element_size()),
        static_cast<uint32_t>(TILE_HEIGHT),
        static_cast<uint32_t>(TILE_WIDTH)};

    // Add ShardedAddrGen compile-time args
    shard_builder::extend_sharding_compile_time_args(input, reader_ct_args);

    // Define for sharded path
    std::map<std::string, std::string> reader_defines;
    reader_defines["SHARDED"] = "1";

    // Create reader kernel
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "reader_unary_pad_height_sharded.cpp",
        core_range,
        ReaderDataMovementConfig(reader_ct_args, reader_defines));

    // Create writer kernel with ShardedAddrGen support
    std::vector<uint32_t> writer_ct_args = {output_cb_index};

    // Add ShardedAddrGen compile-time args for writer
    shard_builder::extend_sharding_compile_time_args(output, writer_ct_args);

    std::map<std::string, std::string> writer_defines;
    writer_defines["SHARDED"] = "1";

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "writer_tilize_sharded.cpp",
        core_range,
        WriterDataMovementConfig(writer_ct_args, writer_defines));

    uint32_t num_tiles_per_block = tiles_per_row;
    uint32_t num_blocks = total_tiles / num_tiles_per_block;

    std::vector<uint32_t> compute_args = {
        num_blocks,          // per_core_block_cnt
        num_tiles_per_block  // per_core_block_tile_cnt
    };

    CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
        core_range,
        ComputeConfig{.fp32_dest_acc_en = fp32_llk_acc, .compile_args = compute_args});

    // Set runtime arguments for reader
    uint32_t packed_pad_value = detail::get_packed_value(input, pad_value);
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_rt_args = {
        src_buffer->address(),  // Base address for ShardedAddrGen
        input_width,            // logical_width
        output_width,           // padded_width
        input_height,           // logical_height
        output_height,          // padded_height
        tiles_per_row,
        tile_rows,
        num_batches,
        packed_pad_value};

    // Add ShardedAddrGen runtime args (mapping table)
    shard_builder::extend_sharding_run_time_args(input, reader_rt_args);

    SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

    // Set runtime arguments for writer
    std::vector<uint32_t> writer_rt_args = {
        dst_buffer->address(),  // Base address for ShardedAddrGen
        total_tiles};

    // Add ShardedAddrGen runtime args (mapping table) for writer
    shard_builder::extend_sharding_run_time_args(output, writer_rt_args);

    SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);

    return cached_program_t(
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .cb_src0 = cb_src0,
            .cb_output = cb_output,
            .core = core});
}

void TilizeWithValPaddingSingleCoreShardedFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const Tensor& input_tensor,
    const Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    // Reuse core used in create()
    const CoreCoord core = shared_variables.core.start_coord;  // Extract CoreCoord from CoreRange

    // Override buffer addresses in runtime args (index 0 for both kernels)
    auto& reader_rt_args = GetRuntimeArgs(program, shared_variables.reader_kernel_id, core);
    reader_rt_args[0] = src_buffer->address();

    auto& writer_rt_args = GetRuntimeArgs(program, shared_variables.writer_kernel_id, core);
    writer_rt_args[0] = dst_buffer->address();
}

}  // namespace ttnn::prim
