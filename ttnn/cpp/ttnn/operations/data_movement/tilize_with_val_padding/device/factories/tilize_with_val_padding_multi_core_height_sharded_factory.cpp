// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_height_sharded_factory.hpp"

#include <cmath>
#include <map>
#include <tt-logger/tt-logger.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeWithValPaddingMultiCoreHeightShardedFactory::cached_program_t
TilizeWithValPaddingMultiCoreHeightShardedFactory::create(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor, const Tensor& output_tensor) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const Tensor& input = input_tensor;
    const Tensor& output = output_tensor;
    auto pad_value = operation_attributes.pad_value;

    const auto buf_shard_spec = input.buffer()->shard_spec();
    log_info(
        tt::LogTest,
        "Tilize height-sharded input pages: aligned_page_size={} page_size={} tensor2d_shape_in_pages=[{}, {}]",
        input.buffer()->aligned_page_size(),
        input.buffer()->page_size(),
        buf_shard_spec.tensor2d_shape_in_pages[0],
        buf_shard_spec.tensor2d_shape_in_pages[1]);

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = input.dtype() == DataType::FLOAT32;

    auto input_shard_spec = input.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();

    auto all_cores = output_shard_spec.grid;

    auto output_shape = output.padded_shape();
    uint32_t output_height = output_shape[-2];
    uint32_t output_width = output_shape[-1];
    uint32_t num_batches = output.physical_volume() / (output_height * output_width);

    log_info(
        tt::LogTest,
        "Tilize height-sharded shapes: input_logical={} input_padded={} output_logical={} output_padded={} "
        "num_batches={}",
        input.logical_shape(),
        input.padded_shape(),
        output.logical_shape(),
        output.padded_shape(),
        num_batches);

    const uint32_t input_shard_height = input_shard_spec.shape[0];
    const uint32_t input_shard_width = input_shard_spec.shape[1];
    const uint32_t output_shard_height = output_shard_spec.shape[0];
    const uint32_t output_shard_width = output_shard_spec.shape[1];

    const uint32_t output_shard_count = shard_builder::get_sharding_core_count(output);
    const uint32_t global_logical_height = input.padded_shape()[-2];  // Need for batch row stride.

    TT_FATAL(num_batches > 0, "num_batches must be > 0");
    const uint32_t total_logical_height = global_logical_height * num_batches;
    const bool shard_is_batch_partition = (num_batches > 1) && (output_shard_height == global_logical_height) &&
                                          (input_shard_height == output_shard_height) &&
                                          (output_shard_count == num_batches);
    TT_FATAL(
        shard_is_batch_partition || output_shard_height % num_batches == 0,
        "Output shard height must be divisible by num_batches when sharding within a batch");
    const uint32_t effective_num_batches = shard_is_batch_partition ? 1 : num_batches;
    const uint32_t output_shard_height_per_batch =
        shard_is_batch_partition ? output_shard_height : (output_shard_height / num_batches);

    TT_FATAL(output_shard_height % TILE_HEIGHT == 0, "Output shard height must be TILE_HEIGHT-aligned");
    TT_FATAL(output_shard_width % TILE_WIDTH == 0, "Output shard width must be TILE_WIDTH-aligned");

    const uint32_t tiles_per_row = output_shard_width / TILE_WIDTH;
    const uint32_t tile_rows = output_shard_height_per_batch / TILE_HEIGHT;
    const uint32_t total_tiles_per_core = tiles_per_row * tile_rows * effective_num_batches;

    log_info(
        tt::LogTest,
        "Tilize height-sharded params: input_shard_h={} input_shard_w={} output_shard_h={} output_shard_h_per_batch={} "
        "output_shard_w={} tiles_per_row={} tile_rows={} total_tiles_per_core={} global_logical_height={} "
        "effective_num_batches={} shard_is_batch_partition={} total_logical_height={}",
        input_shard_height,
        input_shard_width,
        output_shard_height,
        output_shard_height_per_batch,
        output_shard_width,
        tiles_per_row,
        tile_rows,
        total_tiles_per_core,
        global_logical_height,
        effective_num_batches,
        shard_is_batch_partition,
        total_logical_height);

    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0, program, all_cores, input_single_tile_size, tiles_per_row * 2, input_cb_data_format);

    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16, program, all_cores, output_single_tile_size, tiles_per_row * 2, output_cb_data_format);

    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(src0_cb_index),
        static_cast<uint32_t>(input.element_size()),
        static_cast<uint32_t>(TILE_HEIGHT),
        static_cast<uint32_t>(TILE_WIDTH)};

    shard_builder::extend_sharding_compile_time_args(input, reader_ct_args);

    std::map<std::string, std::string> reader_defines;
    reader_defines["SHARDED"] = "1";

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "reader_unary_pad_height_sharded_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args, reader_defines));

    std::vector<uint32_t> writer_ct_args = {output_cb_index};
    shard_builder::extend_sharding_compile_time_args(output, writer_ct_args);

    std::map<std::string, std::string> writer_defines;
    writer_defines["SHARDED"] = "1";

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "writer_tilize_sharded_multicore.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args, writer_defines));

    const uint32_t num_tiles_per_block = tiles_per_row;
    const uint32_t num_blocks = total_tiles_per_core / num_tiles_per_block;

    std::vector<uint32_t> compute_args = {
        num_blocks,          // per_core_block_cnt
        num_tiles_per_block  // per_core_block_tile_cnt
    };

    CreateKernel(
        program,
        "ttnn/cpp/ttnn/kernel/compute/tilize.cpp",
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = fp32_llk_acc, .compile_args = compute_args});

    uint32_t packed_pad_value = detail::get_packed_value(input, pad_value);
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    const tt::tt_metal::IDevice* device = output.device();
    const auto core_ranges = output.buffer()->shard_spec().grid().ranges();
    const bool shard_grid_transposed =
        ((output.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
          output_shard_spec.orientation == ShardOrientation::ROW_MAJOR) ||
         ((output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
           output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) &&
          output_shard_spec.orientation == ShardOrientation::COL_MAJOR));
    const bool is_dram = output.memory_config().is_dram();

    std::vector<CoreCoord> logical_cores;
    std::vector<CoreCoord> shard_cores;
    for (const auto& core_range : core_ranges) {
        if (shard_grid_transposed) {
            for (uint32_t x_index = core_range.start_coord.x; x_index <= core_range.end_coord.x; x_index++) {
                for (uint32_t y_index = core_range.start_coord.y; y_index <= core_range.end_coord.y; y_index++) {
                    CoreCoord logical_core(x_index, y_index);
                    logical_cores.push_back(logical_core);
                    shard_cores.push_back(is_dram ? logical_core : device->worker_core_from_logical_core(logical_core));
                }
            }
        } else {
            for (uint32_t y_index = core_range.start_coord.y; y_index <= core_range.end_coord.y; y_index++) {
                for (uint32_t x_index = core_range.start_coord.x; x_index <= core_range.end_coord.x; x_index++) {
                    CoreCoord logical_core(x_index, y_index);
                    logical_cores.push_back(logical_core);
                    shard_cores.push_back(is_dram ? logical_core : device->worker_core_from_logical_core(logical_core));
                }
            }
        }
    }
    const uint32_t shard_count = shard_cores.size();
    const uint32_t input_shard_count = shard_builder::get_sharding_core_count(input);
    TT_FATAL(
        input_shard_count == shard_count && output_shard_count == shard_count,
        "Input/output shard core counts must match for height-sharded tilize (input={}, output={}, program={})",
        input_shard_count,
        output_shard_count,
        shard_count);

    const auto output_shard_map = shard_builder::generate_run_time_args(output);
    auto get_core_index = [](const std::vector<uint32_t>& shard_map, const CoreCoord& core, uint32_t core_count) {
        const uint32_t packed = ((core.x & 0xFF) << 8) | (core.y & 0xFF);
        uint32_t core_num = 0;
        for (uint32_t idx = 0; idx < shard_map.size() && core_num < core_count; ++idx) {
            const uint32_t word = shard_map[idx];
            const uint32_t high = (word >> 16) & 0xFFFF;
            if (high == packed) {
                return core_num;
            }
            ++core_num;
            if (core_num >= core_count) {
                break;
            }
            const uint32_t low = word & 0xFFFF;
            if (low == packed) {
                return core_num;
            }
            ++core_num;
        }
        TT_FATAL(false, "Core ({},{}) not found in shard map", core.x, core.y);
        return 0u;
    };

    // Per-core RT args.
    for (uint32_t i = 0; i < shard_count; ++i) {
        const auto& core = shard_cores[i];
        const auto& logical_core = logical_cores[i];
        const uint32_t writer_core_index = get_core_index(output_shard_map, core, shard_count);

        // Shard start row per batch
        const uint32_t shard_start_row = shard_is_batch_partition ? (writer_core_index * output_shard_height)
                                                                  : (writer_core_index * output_shard_height_per_batch);
        const uint32_t logical_height_core =
            shard_is_batch_partition
                ? output_shard_height_per_batch
                : (shard_start_row < global_logical_height
                       ? std::min(output_shard_height_per_batch, global_logical_height - shard_start_row)
                       : 0);
        const uint32_t padded_height_core = output_shard_height_per_batch;

        // For block Sharding support: Column offset in bytes. For pure HEIGHT, 0.
        const uint32_t start_col_bytes = 0;  // !TODO: later support.

        // for now assume they all the same.
        // const uint32_t logical_height_core = input_shard_spec.shape[0];
        // const uint32_t padded_height_core = output_shard_spec.shape[0];

        std::vector<uint32_t> reader_rt_args = {
            src_buffer->address(),  // Base address for ShardedAddrGen
            input_shard_width,      // logical_width
            output_shard_width,     // padded_width
            logical_height_core,    // logical_height
            padded_height_core,     // padded_height
            shard_is_batch_partition ? total_logical_height : global_logical_height,
            shard_start_row,
            start_col_bytes,  // For block support.
            tiles_per_row,
            tile_rows,
            effective_num_batches,
            packed_pad_value};

        shard_builder::extend_sharding_run_time_args(input, reader_rt_args);
        SetRuntimeArgs(program, reader_kernel_id, logical_core, reader_rt_args);

        // Tile offset per core.
        const uint32_t shard_start_tile = writer_core_index * total_tiles_per_core;
        std::vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),  // Base address for ShardedAddrGen
            total_tiles_per_core,
            shard_start_tile};

        shard_builder::extend_sharding_run_time_args(output, writer_rt_args);
        SetRuntimeArgs(program, writer_kernel_id, logical_core, writer_rt_args);
    }

    return cached_program_t(
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .cb_src0 = cb_src0,
            .cb_output = cb_output,
            .cores = logical_cores});
}

void TilizeWithValPaddingMultiCoreHeightShardedFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const Tensor& input_tensor,
    const Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    for (const auto& core : shared_variables.cores) {
        auto& reader_rt_args = GetRuntimeArgs(program, shared_variables.reader_kernel_id, core);
        reader_rt_args[0] = src_buffer->address();

        auto& writer_rt_args = GetRuntimeArgs(program, shared_variables.writer_kernel_id, core);
        writer_rt_args[0] = dst_buffer->address();
    }
}
}  // namespace ttnn::prim
