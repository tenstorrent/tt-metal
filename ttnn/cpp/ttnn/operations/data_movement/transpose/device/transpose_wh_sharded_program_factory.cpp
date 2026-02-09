// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include <algorithm>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::transpose::program {

TransposeWHShardedProgramFactory::cached_program_t TransposeWHShardedProgramFactory::create(
    const transpose::TransposeParams& /*operation_attributes*/,
    const transpose::TransposeInputs& tensor_args,
    transpose::tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = tensor_return_value;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    Program program = CreateProgram();

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const auto tile = input_tensor.tensor_spec().tile();
    const uint32_t tile_hw = tile.get_tile_hw();

    IDevice* device = input_tensor.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto shard_spec = input_tensor.shard_spec().value();
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores = shard_spec.grid;
    uint32_t num_tiles_per_shard = shard_spec.numel() / tile_hw;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = num_tiles_per_shard;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size)
            .set_globally_allocated_address(*input_tensor.buffer());
    auto cb_src0 = CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = num_tiles_per_shard;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size)
            .set_globally_allocated_address(*output_tensor.buffer());
    auto cb_output = CreateCircularBuffer(program, total_cores, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index};
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    std::vector<uint32_t> compute_compile_time_args = {src0_cb_index, output_cb_index};

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        total_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
        total_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_sharded.cpp",
        total_cores,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_compile_time_args});

    auto padded_shape = input_tensor.padded_shape();
    auto shard_shape = shard_spec.shape;

    uint32_t H = padded_shape[2];
    uint32_t Hs = shard_shape[0], Ws = shard_shape[1];

    uint32_t Hts = Hs / tile.get_height();
    uint32_t Wts = Ws / tile.get_width();

    uint32_t Ht = H / tile.get_height();
    uint32_t Ht_per_shard = std::min(Ht, Hts);

    uint32_t num_hw_blocks_per_shard = Hts > Ht ? Hts / Ht : 1;

    uint32_t HtWt_tile_size = Ht_per_shard * Wts;
    uint32_t num_blocks = num_hw_blocks_per_shard * HtWt_tile_size;

    auto bbox = all_cores.bounding_box();
    std::vector<CoreCoord> cores =
        grid_to_cores_with_noop(bbox.end_coord.x, bbox.end_coord.y, num_cores_x, num_cores_y, row_major);

    std::vector<std::vector<uint32_t>> unary_reader_args = {cores.size(), std::vector<uint32_t>(1)};
    std::vector<std::vector<uint32_t>> unary_compute_args = {cores.size(), std::vector<uint32_t>(5)};
    std::vector<std::vector<uint32_t>> unary_writer_args = {cores.size(), std::vector<uint32_t>(1)};
    std::fill(
        unary_reader_args.begin(),
        unary_reader_args.begin() + all_cores.num_cores(),
        std::vector<uint32_t>{num_blocks});
    std::fill(
        unary_compute_args.begin(),
        unary_compute_args.begin() + all_cores.num_cores(),
        std::vector<uint32_t>{num_blocks, HtWt_tile_size, num_hw_blocks_per_shard, Ht_per_shard, Wts});
    std::fill(
        unary_writer_args.begin(),
        unary_writer_args.begin() + all_cores.num_cores(),
        std::vector<uint32_t>{num_blocks});

    SetRuntimeArgs(program, reader_kernel_id, cores, unary_reader_args);
    SetRuntimeArgs(program, compute_kernel_id, cores, unary_compute_args);
    SetRuntimeArgs(program, writer_kernel_id, cores, unary_writer_args);

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cb_src0 = cb_src0,
         .cb_output = cb_output,
         .src0_single_tile_size = src0_single_tile_size,
         .dst_single_tile_size = dst_single_tile_size,
         .num_cores_x = num_cores_x,
         .num_cores_y = num_cores_y}};
}

void TransposeWHShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const transpose::TransposeParams& /*operation_attributes*/,
    const transpose::TransposeInputs& tensor_args,
    transpose::tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    const auto& src_tensor = tensor_args.input;
    auto& dst_tensor = tensor_return_value;

    auto* const src_buffer = src_tensor.buffer();
    auto* const dst_buffer = dst_tensor.buffer();

    bool src0_sharded = src_tensor.is_sharded();
    bool out_sharded = dst_tensor.is_sharded();

    auto shard_spec = src_tensor.shard_spec().value();

    const auto tile = src_tensor.tensor_spec().tile();
    const uint32_t tile_hw = tile.get_tile_hw();

    uint32_t num_tiles_per_shard = shard_spec.numel() / tile_hw;

    if (src0_sharded) {
        UpdateDynamicCircularBufferAddressAndTotalSize(
            program,
            shared_variables.cb_src0,
            *src_buffer,
            num_tiles_per_shard * shared_variables.src0_single_tile_size);
    }

    if (out_sharded) {
        UpdateDynamicCircularBufferAddressAndTotalSize(
            program,
            shared_variables.cb_output,
            *dst_buffer,
            num_tiles_per_shard * shared_variables.dst_single_tile_size);
    }

    auto padded_shape = src_tensor.padded_shape();
    auto shard_shape = shard_spec.shape;

    uint32_t H = padded_shape[2];
    uint32_t Hs = shard_shape[0], Ws = shard_shape[1];

    uint32_t Hts = Hs / tile.get_height();
    uint32_t Wts = Ws / tile.get_width();

    uint32_t Ht = H / tile.get_height();
    uint32_t Ht_per_shard = std::min(Ht, Hts);

    uint32_t num_hw_blocks_per_shard = Hts > Ht ? Hts / Ht : 1;

    uint32_t HtWt_tile_size = Ht_per_shard * Wts;
    uint32_t num_blocks = num_hw_blocks_per_shard * HtWt_tile_size;

    const auto& all_cores = shard_spec.grid;
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto bbox = all_cores.bounding_box();
    std::vector<CoreCoord> cores = grid_to_cores_with_noop(
        bbox.end_coord.x, bbox.end_coord.y, shared_variables.num_cores_x, shared_variables.num_cores_y, row_major);

    std::vector<std::vector<uint32_t>> unary_reader_args = {cores.size(), std::vector<uint32_t>(1)};
    std::vector<std::vector<uint32_t>> unary_compute_args = {cores.size(), std::vector<uint32_t>(5)};
    std::vector<std::vector<uint32_t>> unary_writer_args = {cores.size(), std::vector<uint32_t>(1)};
    std::fill(
        unary_reader_args.begin(),
        unary_reader_args.begin() + all_cores.num_cores(),
        std::vector<uint32_t>{num_blocks});
    std::fill(
        unary_compute_args.begin(),
        unary_compute_args.begin() + all_cores.num_cores(),
        std::vector<uint32_t>{num_blocks, HtWt_tile_size, num_hw_blocks_per_shard, Ht_per_shard, Wts});
    std::fill(
        unary_writer_args.begin(),
        unary_writer_args.begin() + all_cores.num_cores(),
        std::vector<uint32_t>{num_blocks});

    SetRuntimeArgs(program, shared_variables.reader_kernel_id, cores, unary_reader_args);
    SetRuntimeArgs(program, shared_variables.compute_kernel_id, cores, unary_compute_args);
    SetRuntimeArgs(program, shared_variables.writer_kernel_id, cores, unary_writer_args);
}

}  // namespace ttnn::operations::data_movement::transpose::program
