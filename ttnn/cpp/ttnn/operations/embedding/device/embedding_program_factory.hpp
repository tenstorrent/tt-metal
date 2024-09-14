// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_log.h"

using namespace tt;

namespace ttnn::operations::embedding::detail {

operation::ProgramWithCallbacks embeddings_tilized(
    const Tensor &a,
    const Tensor &weights,
    Tensor &output,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token) {
    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer *a_buffer = a.buffer();
    tt_metal::Buffer *weights_buffer = weights.buffer();
    tt_metal::Buffer *out_buffer = output.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = a.device();
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

    // row major, page size is last dim
    uint32_t input_page_size = a.get_legacy_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.get_legacy_shape()[-1] * weights_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]
    uint32_t num_embeddings = weights.get_legacy_shape()[-2];

    uint32_t batch_size = a.get_legacy_shape()[0];
    uint32_t num_output_rows_per_batch = a.get_legacy_shape()[-1];
    uint32_t num_output_rows = num_output_rows_per_batch * batch_size;
    uint32_t num_blocks = num_output_rows / TILE_HEIGHT;
    uint32_t num_blocks_per_batch = num_output_rows_per_batch / TILE_HEIGHT;

    auto num_embedding_dims = weights.get_legacy_shape()[-1];

    // setup problem and grid size
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t problem_size = num_blocks;

    auto compute_with_storage_grid_size = DeviceComputeWithStorageGridSize(device);
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, problem_size);
    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    // Create Buffers
    uint32_t num_tiles_per_block = weights.get_legacy_shape()[-1] / TILE_WIDTH;
    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    EmbeddingsIndexType embeddings_index_type;
    if(a.get_dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    }
    else{
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }


    tt::DataFormat weights_cb_data_format = tt_metal::datatype_to_dataformat_converter(weights.get_dtype());
    uint32_t weights_single_tile_size = tt_metal::detail::TileSize(weights_cb_data_format);
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t buffering = weights.get_legacy_shape()[-1] > 2048 ? 1 : 2;

    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(
            buffering * num_tiles_per_block * weights_single_tile_size, {{src0_cb_index, weights_cb_data_format}})
            .set_page_size(src0_cb_index, weights_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(TILE_HEIGHT * input_element_size_bytes, {{src1_cb_index, input_cb_data_format}})
            .set_page_size(src1_cb_index, TILE_HEIGHT * input_element_size_bytes);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    if (embeddings_type == EmbeddingsType::PADDED) {
        uint32_t src2_cb_index = 2;
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    } else if (embeddings_type == EmbeddingsType::BINARY) {
        uint32_t src2_cb_index = 2;
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(2 * cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    }

    uint32_t output_cb_index = 16;  // output operands start at index 16
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            buffering * num_tiles_per_block * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    bool input_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_page_size);
    uint32_t input_log2_stick_size = input_stick_size_is_power_of_two ? (std::uint32_t)log2(input_page_size) : 0;
    bool weight_stick_size_is_power_of_two = is_power_of_two_at_least_32(weight_page_size);
    uint32_t weight_log2_stick_size = weight_stick_size_is_power_of_two ? (std::uint32_t)log2(weight_page_size) : 0;

    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)input_stick_size_is_power_of_two,
        (std::uint32_t)input_page_size,
        (std::uint32_t)input_log2_stick_size,
        (std::uint32_t)weights_is_dram,
        (std::uint32_t)weight_stick_size_is_power_of_two,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)weight_log2_stick_size,
        (std::uint32_t)num_tiles_per_block,
        (std::uint32_t)TILE_HEIGHT * input_element_size_bytes};

    std::map<string, string> embedding_defines = {{magic_enum::enum_name(embeddings_type).data(), "1"}, {magic_enum::enum_name(embeddings_index_type).data(), "1"}};

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_tilize.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(
            embedding_compile_time_args,
             embedding_defines));

    if (num_blocks_per_core_group_1 > 0) {
        vector<uint32_t> compute_args_1 = {
            uint32_t(num_blocks_per_core_group_1),  // per_core_block_cnt
            uint32_t(num_tiles_per_block)           // per_core_block_tile_cnt
        };
        auto tilize_kernel_id_1 = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
            core_group_1,
            tt_metal::ComputeConfig{.compile_args = compute_args_1});
    }

    if (num_blocks_per_core_group_2 > 0) {
        vector<uint32_t> compute_args_2 = {
            uint32_t(num_blocks_per_core_group_2),  // per_core_block_cnt
            uint32_t(num_tiles_per_block)           // per_core_block_tile_cnt
        };
        auto tilize_kernel_id_2 = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.compile_args = compute_args_2});
    }

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)out_is_dram};

    // Tilized writer
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(
            writer_compile_time_args));

    uint32_t input_offset = 0;
    uint32_t weight_offset = 0;
    uint32_t tile_offset = 0;

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    std::vector<uint32_t> reader_runtime_args = {
        (std::uint32_t)a.buffer()->address(),
        (std::uint32_t)weights.buffer()->address(),
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
    };
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_runtime_args.push_back(pad_token.value());
    }

    std::vector<uint32_t> writer_runtime_args = {
        (std::uint32_t)output.buffer()->address(), (std::uint32_t)0, (std::uint32_t)0};

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord &core = cores[i];

        uint32_t local_input_offset = input_offset;
        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        {
            reader_runtime_args[2] = input_offset / num_blocks_per_batch;
            reader_runtime_args[3] = input_offset % num_blocks_per_batch * TILE_HEIGHT * input_element_size_bytes;
            reader_runtime_args[4] = local_num_blocks;
            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        }

        // Writer
        {
            writer_runtime_args[1] = num_tiles_per_block * local_num_blocks;
            writer_runtime_args[2] = tile_offset;
            tile_offset += local_num_blocks * num_tiles_per_block;
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }

        input_offset += local_num_blocks;
    }

    auto override_runtime_args_callback = [num_cores_x, num_cores_y, reader_kernel_id, writer_kernel_id, cores, device](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto output_dram_buffer = output_buffers.at(0);
        auto input_dram_buffer = input_buffers.at(0);
        auto weights_dram_buffer = input_buffers.at(1);

        for (const auto &core : cores) {
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_dram_buffer->address();
                runtime_args[1] = weights_dram_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = output_dram_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks embeddings_rm(
    const Tensor &a,
    const Tensor &weights,
    Tensor &output,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token) {
    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer *a_buffer = a.buffer();
    tt_metal::Buffer *weights_buffer = weights.buffer();
    tt_metal::Buffer *out_buffer = output.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    Device *device = a.device();
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
    constexpr uint32_t alignment = 32;
    uint32_t block_height = (alignment / input_element_size_bytes);
    uint32_t num_blocks = num_output_rows;
    uint32_t num_blocks_per_batch = num_output_rows_per_batch;

    auto num_embedding_dims = weights.get_legacy_shape()[-1];

    // setup problem and grid size
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t problem_size = num_blocks;

    auto compute_with_storage_grid_size = DeviceComputeWithStorageGridSize(device);
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

    uint32_t src0_cb_index = 0;
    uint32_t rounded_weight_page_size = round_up_to_mul32(weight_page_size);
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * rounded_weight_page_size, {{src0_cb_index, weights_cb_data_format}})
            .set_page_size(src0_cb_index, rounded_weight_page_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = 1;
    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(block_height * index_page_size, {{src1_cb_index, input_cb_data_format}})
            .set_page_size(src1_cb_index, block_height * index_page_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    if (embeddings_type == EmbeddingsType::PADDED) {
        uint32_t src2_cb_index = 2;
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    } else if (embeddings_type == EmbeddingsType::BINARY) {
        uint32_t src2_cb_index = 2;
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(2 * cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    }

    uint32_t output_cb_index = src0_cb_index;

    bool input_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_page_size);
    uint32_t input_log2_stick_size = input_stick_size_is_power_of_two ? (std::uint32_t)log2(input_page_size) : 0;
    bool weight_stick_size_is_power_of_two = is_power_of_two_at_least_32(weight_page_size);
    uint32_t weight_log2_stick_size = weight_stick_size_is_power_of_two ? (std::uint32_t)log2(weight_page_size) : 0;

    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)input_stick_size_is_power_of_two,
        (std::uint32_t)input_page_size,
        (std::uint32_t)input_log2_stick_size,
        (std::uint32_t)weights_is_dram,
        (std::uint32_t)weight_stick_size_is_power_of_two,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)weight_log2_stick_size,
        (std::uint32_t)block_height,
        (std::uint32_t)block_height * input_element_size_bytes};

    EmbeddingsIndexType embeddings_index_type;
    if(a.get_dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    }
    else{
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    std::map<string, string> embedding_defines = {{magic_enum::enum_name(embeddings_type).data(), "1"}, {magic_enum::enum_name(embeddings_index_type).data(), "1"}};

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(
            embedding_compile_time_args,
            embedding_defines));

    bool output_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_page_size);
    uint32_t output_log2_stick_size = output_stick_size_is_power_of_two ? (std::uint32_t)log2(output_page_size) : 0;
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
        const CoreCoord &core = cores[i];

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

    auto override_runtime_args_callback = [num_cores_x, num_cores_y, reader_kernel_id, writer_kernel_id, cores, device](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto output_dram_buffer = output_buffers.at(0);
        auto input_dram_buffer = input_buffers.at(0);
        auto weights_dram_buffer = input_buffers.at(1);

        for (const auto &core : cores) {
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input_dram_buffer->address();
                runtime_args[1] = weights_dram_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = output_dram_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks embeddings_(
    const Tensor &a,
    const Tensor &weights,
    Tensor &output,
    bool tilized,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token) {
    if (tilized) {
        return embeddings_tilized(a, weights, output, embeddings_type, pad_token);
    } else {
        return embeddings_rm(a, weights, output, embeddings_type, pad_token);
    }
}
}  // namespace ttnn::operations::embedding::detail
