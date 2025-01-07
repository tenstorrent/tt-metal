// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/core.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_log.h"

using namespace tt;

struct CoreSplitResult {
    uint32_t required_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
};

CoreSplitResult split_work_to_cores_aligned(
    const CoreCoord grid_size, const uint32_t units_to_divide, const uint32_t alignment) {
    ZoneScoped;

    uint32_t num_cores_x = grid_size.x, num_cores_y = grid_size.y;
    uint32_t total_cores = num_cores_x * num_cores_y;

    // Initialize units_per_core and required_cores
    uint32_t units_per_core = alignment;
    uint32_t required_cores = (units_to_divide + units_per_core - 1) / units_per_core;

    // find units per core and required cores
    if (required_cores > total_cores) {
        units_per_core = ((units_to_divide + total_cores - 1) / total_cores + alignment - 1) / alignment * alignment;
        required_cores = (units_to_divide + units_per_core - 1) / units_per_core;
    }

    // Core set for all active cores
    CoreRangeSet all_cores = num_cores_to_corerangeset(required_cores, grid_size, false);

    // Calculate remaining units for the last core
    uint32_t evenly_distributed_units = (required_cores - 1) * units_per_core;
    uint32_t remaining_units = units_to_divide - evenly_distributed_units;

    // Create core groups
    CoreRangeSet core_group_1 = all_cores;
    CoreRangeSet core_group_2;

    // Handle the last core if remaining units are less than units_per_core
    if (remaining_units > 0 && remaining_units < units_per_core) {
        uint32_t last_core_x = (required_cores - 1) % num_cores_x;
        uint32_t last_core_y = (required_cores - 1) / num_cores_x;

        core_group_2 =
            CoreRangeSet(CoreRange(CoreCoord(last_core_x, last_core_y), CoreCoord(last_core_x, last_core_y)));
        core_group_1 = num_cores_to_corerangeset(required_cores - 1, grid_size, false);
    }

    // Adjust the units per core for each group
    uint32_t units_per_core_group_1 = units_per_core;
    uint32_t units_per_core_group_2 = remaining_units < units_per_core ? remaining_units : 0;

    return CoreSplitResult{
        required_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2};
}

namespace ttnn::operations::embedding::detail {

operation::ProgramWithCallbacks embeddings_fused(
    const Tensor& a,
    const Tensor& weights,
    Tensor& output,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token) {
    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer* a_buffer = a.buffer();
    tt_metal::Buffer* weights_buffer = weights.buffer();
    tt_metal::Buffer* out_buffer = output.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device* device = a.device();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    bool in0_is_dram = a.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_is_dram = weights.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    bool output_sharded = is_sharded(output.buffer()->buffer_layout());

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();

    // row major, page size is last dim
    uint32_t input_page_size = a.get_legacy_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.get_legacy_shape()[-1] * weights_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]
    uint32_t num_embeddings = weights.get_legacy_shape()[-2];

    uint32_t batch_size = a.get_legacy_shape()[0];
    uint32_t num_output_rows_per_batch = a.get_legacy_shape()[-1];
    uint32_t num_output_rows = num_output_rows_per_batch * batch_size;
    // Note: num_blocks is just blocks along height
    uint32_t num_blocks = num_output_rows / TILE_HEIGHT;
    uint32_t num_blocks_per_batch = num_output_rows_per_batch / TILE_HEIGHT;
    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2, num_tiles_per_block;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    bool row_major;
    if (output_sharded) {
        const auto& shard_spec = output.shard_spec().value();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_cores = all_cores.num_cores();
        num_blocks_per_core_group_1 = shard_spec.shape[0] / TILE_HEIGHT;
        num_blocks_per_core_group_2 = 0;
        num_tiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
        row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    } else {
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
        num_tiles_per_block = weights.get_legacy_shape()[-1] / TILE_WIDTH;
        row_major = false;
    }
    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    // Create Buffers
    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    EmbeddingsIndexType embeddings_index_type;
    if (a.get_dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    tt::DataFormat weights_cb_data_format = tt_metal::datatype_to_dataformat_converter(weights.get_dtype());
    uint32_t weights_single_tile_size = tt_metal::detail::TileSize(weights_cb_data_format);
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    // Hardcoded limit to reduce L1 usage. Should be updated to be tuned based on overall L1 usage
    constexpr uint32_t max_double_buffer_tiles = 64;
    uint32_t buffering = num_tiles_per_block > max_double_buffer_tiles ? 1 : 2;

    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(
            buffering * num_tiles_per_block * weights_single_tile_size, {{src0_cb_index, weights_cb_data_format}})
            .set_page_size(src0_cb_index, weights_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(TILE_HEIGHT * input_element_size_bytes, {{src1_cb_index, input_cb_data_format}})
            .set_page_size(src1_cb_index, TILE_HEIGHT * input_element_size_bytes);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    constexpr uint32_t output_cb_index = CBIndex::c_2;
    uint32_t output_cb_size;
    if (output_sharded) {
        output_cb_size = output.buffer()->aligned_size_per_bank();
    } else {
        output_cb_size = buffering * num_tiles_per_block * output_single_tile_size;
    }
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(output_cb_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    if (output_sharded) {
        cb_output_config.set_globally_allocated_address(*out_buffer);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    constexpr uint32_t src2_cb_index = CBIndex::c_3;
    if (embeddings_type == EmbeddingsType::PADDED) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    } else if (embeddings_type == EmbeddingsType::BINARY) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(2 * cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    }
    uint32_t weight_block_size;
    if (output_sharded) {
        weight_block_size = output.shard_spec().value().shape[1] * weights_element_size_bytes;
    } else {
        weight_block_size = weight_page_size;
    }

    // TODO: Can increase size for larger reads
    uint32_t input_block_size_bytes = TILE_HEIGHT * input_element_size_bytes;
    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)input_page_size,
        (std::uint32_t)weights_is_dram,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)weight_block_size,
        (std::uint32_t)num_tiles_per_block,
        (std::uint32_t)input_block_size_bytes};

    std::map<string, string> embedding_defines = {
        {magic_enum::enum_name(embeddings_type).data(), "1"},
        {magic_enum::enum_name(embeddings_index_type).data(), "1"}};

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_tilize.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(embedding_compile_time_args, embedding_defines));

    if (num_blocks_per_core_group_1 > 0) {
        std::vector<uint32_t> compute_args_1 = {
            uint32_t(src0_cb_index),                // input embeddings_cb_index
            uint32_t(output_cb_index),              // output_cb_index
            uint32_t(num_blocks_per_core_group_1),  // per_core_block_cnt
            uint32_t(num_tiles_per_block)           // per_core_block_tile_cnt
        };
        auto tilize_kernel_id_1 = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp",
            core_group_1,
            tt_metal::ComputeConfig{.compile_args = compute_args_1});
    }

    if (num_blocks_per_core_group_2 > 0) {
        std::vector<uint32_t> compute_args_2 = {
            uint32_t(src0_cb_index),                // input embeddings_cb_index
            uint32_t(output_cb_index),              // output_cb_index
            uint32_t(num_blocks_per_core_group_2),  // per_core_block_cnt
            uint32_t(num_tiles_per_block)           // per_core_block_tile_cnt
        };
        auto tilize_kernel_id_2 = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.compile_args = compute_args_2});
    }
    KernelHandle writer_kernel_id = 0;
    // TODO: We can use the second risc to do more work in parallel
    if (!output_sharded) {
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)out_is_dram};

        // Tilized writer
        writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }

    auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);

    std::vector<uint32_t> reader_runtime_args = {
        (std::uint32_t)a.buffer()->address(),
        (std::uint32_t)weights.buffer()->address(),
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
    };
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_runtime_args.push_back(pad_token.value());
    }

    std::vector<uint32_t> writer_runtime_args = {
        (std::uint32_t)output.buffer()->address(), (std::uint32_t)0, (std::uint32_t)0};

    uint32_t input_offset = 0;
    uint32_t weight_offset = 0;
    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        {
            reader_runtime_args[2] = input_offset / num_blocks_per_batch;
            reader_runtime_args[3] = input_offset % num_blocks_per_batch * input_block_size_bytes;
            reader_runtime_args[4] = weight_offset;
            reader_runtime_args[5] = local_num_blocks;
            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        }

        // Writer
        if (!output_sharded) {
            writer_runtime_args[1] = num_tiles_per_block * local_num_blocks;
            writer_runtime_args[2] = tile_offset;
            tile_offset += local_num_blocks * num_tiles_per_block;
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
            input_offset += local_num_blocks;
        } else {
            weight_offset += weight_block_size;
            if (weight_offset == weight_page_size) {
                weight_offset = 0;
                input_offset += local_num_blocks;
            }
        }
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_id, writer_kernel_id, cores, cb_output](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto output_buffer = output_tensors.at(0).buffer();
            auto output_buffer_address = output_buffer->address();
            auto input_buffer_address = input_tensors.at(0).buffer()->address();
            auto weights_buffer_address = input_tensors.at(1).buffer()->address();

            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);
            const bool output_sharded = is_sharded(output_buffer->buffer_layout());
            if (output_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_output, *output_buffer);
            }

            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args[core.x][core.y];
                    runtime_args[0] = input_buffer_address;
                    runtime_args[1] = weights_buffer_address;
                }

                if (!output_sharded) {
                    auto& runtime_args = writer_runtime_args[core.x][core.y];
                    runtime_args[0] = output_buffer_address;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks embeddings_rm(
    const Tensor& a,
    const Tensor& weights,
    Tensor& output,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token) {
    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer* a_buffer = a.buffer();
    tt_metal::Buffer* weights_buffer = weights.buffer();
    tt_metal::Buffer* out_buffer = output.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device* device = a.device();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    bool in0_is_dram = a.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_is_dram = weights.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();
    uint32_t output_element_size_bytes = output.element_size();

    // row major, page size is last dim
    uint32_t input_page_size = a.get_legacy_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.get_legacy_shape()[-1] * weights_element_size_bytes;
    uint32_t output_page_size = output.get_legacy_shape()[-1] * output_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]
    uint32_t num_embeddings = weights.get_legacy_shape()[-2];

    uint32_t batch_size = a.get_legacy_shape()[0];
    uint32_t num_output_rows_per_batch = a.get_legacy_shape()[-1];
    uint32_t num_output_rows = num_output_rows_per_batch * batch_size;
    auto alignment = a.buffer()->alignment();
    uint32_t block_height = (alignment / input_element_size_bytes);
    uint32_t num_blocks = num_output_rows;
    uint32_t num_blocks_per_batch = num_output_rows_per_batch;

    // setup problem and grid size
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t problem_size = num_blocks;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, problem_size);
    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    // Create Buffers
    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    tt::DataFormat weights_cb_data_format = tt_metal::datatype_to_dataformat_converter(weights.get_dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t rounded_weight_page_size = round_up_to_mul32(weight_page_size);
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * rounded_weight_page_size, {{src0_cb_index, weights_cb_data_format}})
            .set_page_size(src0_cb_index, rounded_weight_page_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(block_height * index_page_size, {{src1_cb_index, input_cb_data_format}})
            .set_page_size(src1_cb_index, block_height * index_page_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    constexpr uint32_t output_cb_index = src0_cb_index;

    constexpr uint32_t src2_cb_index = CBIndex::c_2;
    if (embeddings_type == EmbeddingsType::PADDED) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    } else if (embeddings_type == EmbeddingsType::BINARY) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(2 * cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    }

    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)input_page_size,
        (std::uint32_t)weights_is_dram,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)block_height,
        (std::uint32_t)block_height * input_element_size_bytes};

    EmbeddingsIndexType embeddings_index_type;
    if (a.get_dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    std::map<string, string> embedding_defines = {
        {magic_enum::enum_name(embeddings_type).data(), "1"},
        {magic_enum::enum_name(embeddings_index_type).data(), "1"}};

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(embedding_compile_time_args, embedding_defines));

    bool output_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_page_size);
    uint32_t output_log2_stick_size =
        output_stick_size_is_power_of_two ? (std::uint32_t)std::log2(output_page_size) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)out_is_dram,
        (std::uint32_t)output_stick_size_is_power_of_two,
        (std::uint32_t)output_log2_stick_size};

    // Tilized writer
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t input_offset = 0;
    uint32_t weight_offset = 0;

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    std::vector<uint32_t> reader_runtime_args = {
        (std::uint32_t)a.buffer()->address(),
        (std::uint32_t)weights.buffer()->address(),
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
    };
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_runtime_args.push_back(pad_token.value());
    }
    std::vector<uint32_t> writer_runtime_args = {
        (std::uint32_t)output.buffer()->address(), (std::uint32_t)output_page_size, (std::uint32_t)0, (std::uint32_t)0};

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        {
            reader_runtime_args[2] = input_offset / num_blocks_per_batch;
            reader_runtime_args[3] =
                round_down(input_offset % num_blocks_per_batch, block_height) * input_element_size_bytes;
            reader_runtime_args[4] = local_num_blocks;
            reader_runtime_args[5] = input_offset % num_blocks_per_batch % block_height;
            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        }

        // Writer
        {
            writer_runtime_args[2] = local_num_blocks;
            writer_runtime_args[3] = input_offset;
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }

        input_offset += local_num_blocks;
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_id, writer_kernel_id, cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto output_buffer_address = output_tensors.at(0).buffer()->address();
            auto input_buffer_address = input_tensors.at(0).buffer()->address();
            auto weights_buffer_address = input_tensors.at(1).buffer()->address();

            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args[core.x][core.y];
                    runtime_args[0] = input_buffer_address;
                    runtime_args[1] = weights_buffer_address;
                }

                {
                    auto& runtime_args = writer_runtime_args[core.x][core.y];
                    runtime_args[0] = output_buffer_address;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks embeddings_tilized_indices(
    const Tensor& a,
    const Tensor& weights,
    Tensor& output,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token) {
    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer* a_buffer = a.buffer();
    tt_metal::Buffer* weights_buffer = weights.buffer();
    tt_metal::Buffer* out_buffer = output.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device* device = a.device();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    bool in0_is_dram = a.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_is_dram = weights.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();
    uint32_t output_element_size_bytes = output.element_size();

    // row major, page size is last dim
    uint32_t input_page_size = a.get_logical_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.get_legacy_shape()[-1] * weights_element_size_bytes;
    uint32_t output_page_size = output.get_legacy_shape()[-1] * output_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]
    uint32_t num_embeddings = weights.get_legacy_shape()[-2];

    uint32_t batch_size = a.get_logical_shape()[0];  // num rows
    uint32_t num_cols = a.get_logical_shape()[-1];
    uint32_t volume = num_cols * batch_size;

    auto num_embedding_dims = weights.get_legacy_shape()[-1];

    // setup problem and grid size
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t problem_size = volume;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreSplitResult work = split_work_to_cores_aligned(compute_with_storage_grid_size, problem_size, FACE_HEIGHT);

    uint32_t num_cores = work.required_cores;
    CoreRangeSet all_cores = work.all_cores;
    CoreRangeSet core_group_1 = work.core_group_1;
    CoreRangeSet core_group_2 = work.core_group_2;
    uint32_t num_blocks_per_core_group_1 = work.units_per_core_group_1;
    uint32_t num_blocks_per_core_group_2 = work.units_per_core_group_2;

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    // Create Buffers
    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    tt::DataFormat weights_cb_data_format = tt_metal::datatype_to_dataformat_converter(weights.get_dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t rounded_weight_page_size = round_up_to_mul32(weight_page_size);
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * rounded_weight_page_size, {{src0_cb_index, weights_cb_data_format}})
            .set_page_size(src0_cb_index, rounded_weight_page_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(FACE_HEIGHT * index_page_size, {{src1_cb_index, input_cb_data_format}})
            .set_page_size(src1_cb_index, FACE_HEIGHT * index_page_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    constexpr uint32_t src2_cb_index = CBIndex::c_2;
    if (embeddings_type == EmbeddingsType::PADDED) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    } else if (embeddings_type == EmbeddingsType::BINARY) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(2 * cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    }

    uint32_t output_cb_index = src0_cb_index;

    bool input_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_page_size);
    uint32_t input_log2_stick_size = input_stick_size_is_power_of_two ? (std::uint32_t)std::log2(input_page_size) : 0;
    bool weight_stick_size_is_power_of_two = is_power_of_two_at_least_32(weight_page_size);
    uint32_t weight_log2_stick_size =
        weight_stick_size_is_power_of_two ? (std::uint32_t)std::log2(weight_page_size) : 0;

    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)input_page_size,
        (std::uint32_t)weights_is_dram,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)a.get_logical_shape()[-1],  // width/length of a row
        (std::uint32_t)FACE_HEIGHT};

    EmbeddingsIndexType embeddings_index_type;
    if (a.get_dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    std::map<string, string> embedding_defines = {
        {magic_enum::enum_name(embeddings_type).data(), "1"},
        {magic_enum::enum_name(embeddings_index_type).data(), "1"}};

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embedding_ind_tilized.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(embedding_compile_time_args, embedding_defines));

    bool output_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_page_size);
    uint32_t output_log2_stick_size =
        output_stick_size_is_power_of_two ? (std::uint32_t)std::log2(output_page_size) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)out_is_dram,
        (std::uint32_t)output_stick_size_is_power_of_two,
        (std::uint32_t)output_log2_stick_size};

    // Tilized writer
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t col_offset = 0;
    uint32_t weight_offset = 0;

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    std::vector<uint32_t> reader_runtime_args = {
        (std::uint32_t)a.buffer()->address(),
        (std::uint32_t)weights.buffer()->address(),
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
    };
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_runtime_args.push_back(pad_token.value());
    }
    std::vector<uint32_t> writer_runtime_args = {
        (std::uint32_t)output.buffer()->address(), (std::uint32_t)output_page_size, (std::uint32_t)0, (std::uint32_t)0};

    uint32_t row = 0;
    uint32_t tiles_per_tile_row = (num_cols + TILE_HEIGHT - 1) / TILE_HEIGHT;

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        col_offset = weight_offset % num_cols;
        row = weight_offset / num_cols;

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;
        uint32_t r_f_offset = ((row % TILE_HEIGHT) / FACE_HEIGHT) * 2 * FACE_HW + (row % FACE_HEIGHT) * FACE_HEIGHT;
        // Offset by one face size if we are in the right half of the tile + where we are in the row
        uint32_t c_f_offset = ((col_offset % TILE_HEIGHT) / FACE_HEIGHT) * FACE_HW;
        uint32_t face_offset = r_f_offset + c_f_offset;
        uint32_t curr_tile = (row / TILE_HEIGHT) * tiles_per_tile_row + (col_offset / TILE_HEIGHT);

        // Reader
        {
            reader_runtime_args[2] = curr_tile;
            reader_runtime_args[3] = face_offset;
            reader_runtime_args[4] = local_num_blocks;
            reader_runtime_args[5] = col_offset;
            reader_runtime_args[6] = (col_offset % FACE_HEIGHT);  // starting col in the face row
            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        }

        // Writer
        {
            writer_runtime_args[2] = local_num_blocks;
            writer_runtime_args[3] = weight_offset;
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }

        weight_offset += local_num_blocks;
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_id, writer_kernel_id, cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto output_buffer_address = output_tensors.at(0).buffer()->address();
            auto input_buffer_address = input_tensors.at(0).buffer()->address();
            auto weights_buffer_address = input_tensors.at(1).buffer()->address();

            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args[core.x][core.y];
                    runtime_args[0] = input_buffer_address;
                    runtime_args[1] = weights_buffer_address;
                }

                {
                    auto& runtime_args = writer_runtime_args[core.x][core.y];
                    runtime_args[0] = output_buffer_address;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks embeddings_(
    const Tensor& a,
    const Tensor& weights,
    Tensor& output,
    bool tilized,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token) {
    if (a.get_layout() == ttnn::TILE_LAYOUT) {
        return embeddings_tilized_indices(a, weights, output, embeddings_type, pad_token);
    } else if (tilized) {
        return embeddings_fused(a, weights, output, embeddings_type, pad_token);
    } else {
        return embeddings_rm(a, weights, output, embeddings_type, pad_token);
    }
}
}  // namespace ttnn::operations::embedding::detail
