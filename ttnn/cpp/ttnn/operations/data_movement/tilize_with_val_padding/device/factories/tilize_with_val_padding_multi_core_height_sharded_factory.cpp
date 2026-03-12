// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_height_sharded_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

#include <tt-logger/tt-logger.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

uint32_t get_flattened_row_major_height(const ttnn::Shape& shape) {
    TT_FATAL(shape.rank() >= 1, "Shape rank must be >= 1");
    const uint32_t width = shape[-1];
    TT_FATAL(width > 0, "Last dimension must be > 0");
    return shape.volume() / width;
}

uint32_t get_core_index_from_shard_map(
    const std::vector<uint32_t>& shard_map, const CoreCoord& core, uint32_t expected_core_count) {
    const uint32_t packed = ((core.x & 0xFF) << 8) | (core.y & 0xFF);

    uint32_t logical_index = 0;
    for (uint32_t i = 0; i < shard_map.size() && logical_index < expected_core_count; ++i) {
        const uint32_t word = shard_map[i];

        const uint32_t hi = (word >> 16) & 0xFFFF;
        if (hi == packed) {
            return logical_index;
        }
        ++logical_index;
        if (logical_index >= expected_core_count) {
            break;
        }

        const uint32_t lo = word & 0xFFFF;
        if (lo == packed) {
            return logical_index;
        }
        ++logical_index;
    }

    TT_FATAL(false, "Core ({}, {}) not found in shard map", core.x, core.y);
    return 0;
}

}  // namespace

TilizeWithValPaddingMultiCoreHeightShardedFactory::cached_program_t
TilizeWithValPaddingMultiCoreHeightShardedFactory::create(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor, const Tensor& output_tensor) {
    Program program = CreateProgram();

    const Tensor& input = input_tensor;
    const Tensor& output = output_tensor;

    TT_FATAL(input.memory_config().is_sharded(), "Input must be sharded");
    TT_FATAL(output.memory_config().is_sharded(), "Output must be sharded");

    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "This factory only supports HEIGHT_SHARDED input");
    TT_FATAL(
        output.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "This factory only supports HEIGHT_SHARDED output");

    TT_FATAL(input.shard_spec().has_value(), "Input must have legacy shard_spec");
    TT_FATAL(output.shard_spec().has_value(), "Output must have legacy shard_spec");

    TT_FATAL(
        !input.memory_config().created_with_nd_shard_spec(),
        "Input created with ND shard spec is not supported by this factory");
    TT_FATAL(
        !output.memory_config().created_with_nd_shard_spec(),
        "Output created with ND shard spec is not supported by this factory");

    auto pad_value = operation_attributes.pad_value;

    const auto input_shard_spec = input.shard_spec().value();
    const auto output_shard_spec = output.shard_spec().value();

    const auto all_cores = output_shard_spec.grid;

    const uint32_t input_shard_height = input_shard_spec.shape[0];
    const uint32_t input_shard_width = input_shard_spec.shape[1];
    const uint32_t output_shard_height = output_shard_spec.shape[0];
    const uint32_t output_shard_width = output_shard_spec.shape[1];

    TT_FATAL(input_shard_width > 0 && output_shard_width > 0, "Shard widths must be > 0");
    TT_FATAL(output_shard_height > 0, "Output shard height must be > 0");

    TT_FATAL(output_shard_height % TILE_HEIGHT == 0, "Output shard height must be TILE_HEIGHT aligned");
    TT_FATAL(output_shard_width % TILE_WIDTH == 0, "Output shard width must be TILE_WIDTH aligned");

    const uint32_t flattened_input_logical_rows = get_flattened_row_major_height(input.logical_shape());
    const uint32_t flattened_output_padded_rows = get_flattened_row_major_height(output.padded_shape());

    const uint32_t output_shard_count = shard_builder::get_sharding_core_count(output);
    const uint32_t input_shard_count = shard_builder::get_sharding_core_count(input);

    TT_FATAL(
        input_shard_count == output_shard_count,
        "Input/output shard counts must match for height-sharded tilize (input={}, output={})",
        input_shard_count,
        output_shard_count);

    TT_FATAL(
        output_shard_height * output_shard_count == flattened_output_padded_rows,
        "For height-sharded output, shard_height * shard_count must equal flattened padded rows "
        "(shard_height={}, shard_count={}, flattened_output_padded_rows={})",
        output_shard_height,
        output_shard_count,
        flattened_output_padded_rows);

    TT_FATAL(
        input_shard_width == input.logical_shape()[-1],
        "Height-sharded input is expected to cover full row width per shard. Got input_shard_width={} logical_width={}",
        input_shard_width,
        input.logical_shape()[-1]);

    TT_FATAL(
        output_shard_width == output.padded_shape()[-1],
        "Height-sharded output is expected to cover full row width per shard. Got output_shard_width={} "
        "padded_width={}",
        output_shard_width,
        output.padded_shape()[-1]);

    const uint32_t tiles_per_row = output_shard_width / TILE_WIDTH;
    const uint32_t tile_rows_per_core = output_shard_height / TILE_HEIGHT;
    const uint32_t total_tiles_per_core = tiles_per_row * tile_rows_per_core;

    log_info(
        tt::LogTest,
        "height-sharded tilize factory:"
        " input_logical={} input_padded={} output_logical={} output_padded={}"
        " flattened_input_rows={} flattened_output_rows={}"
        " input_shard=[{}, {}] output_shard=[{}, {}]"
        " shard_count={} tiles_per_row={} tile_rows_per_core={} total_tiles_per_core={}",
        input.logical_shape(),
        input.padded_shape(),
        output.logical_shape(),
        output.padded_shape(),
        flattened_input_logical_rows,
        flattened_output_padded_rows,
        input_shard_height,
        input_shard_width,
        output_shard_height,
        output_shard_width,
        output_shard_count,
        tiles_per_row,
        tile_rows_per_core,
        total_tiles_per_core);

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const uint32_t input_single_tile_size = tile_size(input_cb_data_format);
    const uint32_t output_single_tile_size = tile_size(output_cb_data_format);

    const bool fp32_llk_acc = input.dtype() == DataType::FLOAT32;

    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0, program, all_cores, input_single_tile_size, tiles_per_row * 2, input_cb_data_format);

    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16, program, all_cores, output_single_tile_size, tiles_per_row * 2, output_cb_data_format);

    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(src0_cb_index),
        static_cast<uint32_t>(input.element_size()),
        static_cast<uint32_t>(TILE_HEIGHT),
        static_cast<uint32_t>(TILE_WIDTH),
    };
    shard_builder::extend_sharding_compile_time_args(input, reader_ct_args);

    std::map<std::string, std::string> reader_defines;
    reader_defines["SHARDED"] = "1";

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "reader_unary_pad_height_sharded_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args, reader_defines));

    std::vector<uint32_t> writer_ct_args = {static_cast<uint32_t>(output_cb_index)};
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
    const uint32_t num_blocks = tile_rows_per_core;

    std::vector<uint32_t> compute_args = {
        num_blocks,          // per_core_block_cnt
        num_tiles_per_block  // per_core_block_tile_cnt
    };

    CreateKernel(
        program,
        "ttnn/cpp/ttnn/kernel/compute/tilize.cpp",
        all_cores,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_llk_acc,
            .compile_args = compute_args,
        });

    uint32_t packed_pad_value = detail::get_packed_value(input, pad_value);

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    const IDevice* device = output.device();
    const auto core_ranges = output.buffer()->shard_spec().grid().ranges();
    const bool is_dram = output.memory_config().is_dram();

    std::vector<CoreCoord> logical_cores;
    std::vector<CoreCoord> physical_cores;
    for (const auto& core_range : core_ranges) {
        for (uint32_t y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
            for (uint32_t x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                CoreCoord logical_core{x, y};
                logical_cores.push_back(logical_core);
                physical_cores.push_back(is_dram ? logical_core : device->worker_core_from_logical_core(logical_core));
            }
        }
    }

    TT_FATAL(
        logical_cores.size() == output_shard_count,
        "Number of enumerated cores ({}) must match output shard count ({})",
        logical_cores.size(),
        output_shard_count);

    const auto output_shard_map = shard_builder::generate_run_time_args(output);

    for (uint32_t i = 0; i < logical_cores.size(); ++i) {
        const CoreCoord& logical_core = logical_cores[i];
        const CoreCoord& physical_core = physical_cores[i];

        const uint32_t output_core_index =
            get_core_index_from_shard_map(output_shard_map, physical_core, output_shard_count);

        const uint32_t shard_start_row = output_core_index * output_shard_height;

        const uint32_t logical_height_core =
            (shard_start_row < flattened_input_logical_rows)
                ? std::min(output_shard_height, flattened_input_logical_rows - shard_start_row)
                : 0;

        const uint32_t padded_height_core = output_shard_height;

        // Height-only factory: each shard spans full row width, so no column offset.
        const uint32_t start_col_bytes = 0;

        // Height-only flattened handling: reader kernel can treat whole tensor as one flattened batch.
        const uint32_t flattened_num_batches = 1;

        std::vector<uint32_t> reader_rt_args = {
            src_buffer->address(),         // src_base_addr
            input_shard_width,             // logical_width
            output_shard_width,            // padded_width
            logical_height_core,           // logical_height_core
            padded_height_core,            // padded_height_core
            flattened_input_logical_rows,  // global_logical_height
            shard_start_row,               // shard_start_row
            start_col_bytes,               // start_col_bytes
            tiles_per_row,                 // tiles_per_row
            tile_rows_per_core,            // tile_rows_core
            flattened_num_batches,         // num_batches
            packed_pad_value               // packed_pad_value
        };
        shard_builder::extend_sharding_run_time_args(input, reader_rt_args);
        SetRuntimeArgs(program, reader_kernel_id, logical_core, reader_rt_args);

        const uint32_t shard_start_tile = output_core_index * total_tiles_per_core;

        std::vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),  // dst_base_addr
            total_tiles_per_core,   // num_tiles_core
            shard_start_tile        // shard_start_tile
        };
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
            .cores = logical_cores,
        });
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
