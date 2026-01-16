// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads_program_factory.hpp"

namespace ttnn::operations::experimental::transformer::split_query_key_value_and_split_heads::program {

using namespace tt::constants;
using namespace tt;
using namespace tt_metal;

SplitFusedQKVAndSplitHeadsProgramFactory::cached_program_t SplitFusedQKVAndSplitHeadsProgramFactory::create(
    const split_query_key_value_and_split_heads::SplitQueryKeyValueAndSplitHeadsParams& operation_attributes,
    const split_query_key_value_and_split_heads::SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    const auto& a = tensor_args.input_tensor;
    auto& output = output_tensors;
    auto compute_with_storage_grid_size = operation_attributes.compute_with_storage_grid_size;

    const auto& ashape = a.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_core_tiles = ashape[3] / TILE_WIDTH;  // 96
    uint32_t num_tensors = 3;
    uint32_t num_tiles_per_tensor = per_core_tiles / num_tensors;  // 32
    uint32_t block_size = 1;  // TODO: Play around with different reader and writer block_sizes?
    bool block_size_is_one = block_size == 1;
    uint32_t num_blocks_per_tensor = num_tiles_per_tensor / block_size;

    // Per output tensor args
    // Output shape is: [B, 16, 384, 64] (Q, V heads) or [B, 16, 64, 384] (K heads)
    // For K heads, we write "w_dim" to h_dim instead, but keep nomenclature the same
    uint32_t out_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t out_w = 64;
    uint32_t out_w_tiles = out_w / TILE_WIDTH;
    uint32_t out_c = num_tiles_per_tensor / out_w_tiles;
    uint32_t out_HtWt = out_h_tiles * out_w_tiles;
    uint32_t out_CHtWt = out_c * out_HtWt;
    // If block_size_is_one, writer kernel waits differently
    uint32_t writer_num_blocks_per_tensor = block_size_is_one ? 16 : num_blocks_per_tensor;
    uint32_t num_c_per_block = block_size_is_one ? 1 : block_size / out_w_tiles;

    // Parallelize ashape[2] (384 / 32 = 12 tiles) across columns
    // Parallelize ashape[0] (B) across rows
    uint32_t num_cores_x = ashape[2] / TILE_HEIGHT;
    uint32_t num_cores_y = ashape[0];
    TT_ASSERT(num_cores_x <= compute_with_storage_grid_size.x);
    TT_ASSERT(num_cores_y <= compute_with_storage_grid_size.y);
    CoreCoord core_range = {num_cores_x, num_cores_y};

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    TT_ASSERT((output.size() == 3), "Output vector must be size 3 for split fused qkv!");
    tt_metal::Tensor& q = output[0];
    tt_metal::Tensor& k = output[1];
    tt_metal::Tensor& v = output[2];

    tt_metal::Buffer* q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer* k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");
    tt_metal::Buffer* v_buffer = v.buffer();
    TT_ASSERT(v_buffer != nullptr, "Output v buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    std::vector<uint32_t> reader_compile_time_args = {
        // READER COMPILE TIME ARGS
        (std::uint32_t)block_size,             // block_size
        (std::uint32_t)num_blocks_per_tensor,  // out_num_blocks_per_tensor
    };
    tt::tt_metal::TensorAccessorArgs(in0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        // WRITER COMPILE TIME ARGS
        (std::uint32_t)block_size_is_one,
        (std::uint32_t)block_size,                    // block_size
        (std::uint32_t)writer_num_blocks_per_tensor,  // out_num_blocks_per_tensor
        (std::uint32_t)num_c_per_block,               // out_num_c_per_block
        (std::uint32_t)out_w_tiles,                   // out_w_tiles
        (std::uint32_t)out_h_tiles,                   // out_h_tiles
        (std::uint32_t)out_HtWt,                      // out_HtWt
    };
    tt::tt_metal::TensorAccessorArgs(q_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(k_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(v_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
        "dataflow/reader_tm_tile_layout_create_qkv_heads.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
        "dataflow/writer_tm_tile_layout_create_qkv_heads.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Dummy compute kernel
    std::vector<uint32_t> compute_args = {num_tiles_per_tensor};
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/transpose_wh.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = compute_args});

    // Create circular buffers
    // Use cb0 and cb16 for K heads, which uses compute for transpose_wh
    uint32_t src0_cb_index = 0;
    uint32_t cb0_tiles = num_tiles_per_tensor * 2;  // double buffer
    uint32_t out_cb_index = 16;
    uint32_t out_cb_tiles = num_tiles_per_tensor;
    // Use cb1 as input and output for Q and V heads
    uint32_t src1_cb_index = 1;
    uint32_t cb1_tiles = num_tiles_per_tensor * 4;  // 2 tensors + double buffer
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    tt_metal::CircularBufferConfig cb_out_config =
        tt_metal::CircularBufferConfig(out_cb_tiles * single_tile_size, {{out_cb_index, cb_data_format}})
            .set_page_size(out_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

            std::vector<uint32_t> reader_runtime_args = {
                (std::uint32_t)in0_buffer->address(),                      // in0_tensor_addr,
                (core_idx_x + core_idx_y * num_cores_c) * per_core_tiles,  // in0_tensor_tile_id
            };
            std::vector<uint32_t> writer_runtime_args = {
                (std::uint32_t)q_buffer->address(),                     // q_tensor_addr
                (std::uint32_t)k_buffer->address(),                     // k_tensor_addr
                (std::uint32_t)v_buffer->address(),                     // v_tensor_addr
                (core_idx_x * out_w_tiles) + (core_idx_y * out_CHtWt),  // out_tensor_tile_id
                core_idx_x + (core_idx_y * out_CHtWt),                  // out_tensor_tile_id_with_transpose
            };

            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }
    }

    return {
        std::move(program), {reader_kernel_id, writer_kernel_id, num_cores_r, num_cores_c, start_core_x, start_core_y}};
}

void SplitFusedQKVAndSplitHeadsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const split_query_key_value_and_split_heads::SplitQueryKeyValueAndSplitHeadsParams& /*operation_attributes*/,
    const split_query_key_value_and_split_heads::SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    auto* src_dram_buffer = tensor_args.input_tensor.buffer();

    auto* dst_dram_buffer_query = output_tensors.at(0).buffer();
    auto* dst_dram_buffer_key = output_tensors.at(1).buffer();
    auto* dst_dram_buffer_value = output_tensors.at(2).buffer();

    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    auto reader_kernel_id = shared.reader_kernel_id;
    auto writer_kernel_id = shared.writer_kernel_id;
    auto num_cores_r = shared.num_cores_r;
    auto num_cores_c = shared.num_cores_c;
    auto start_core_x = shared.start_core_x;
    auto start_core_y = shared.start_core_y;

    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

            {
                auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_dram_buffer->address();
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer_query->address();
                runtime_args[1] = dst_dram_buffer_key->address();
                runtime_args[2] = dst_dram_buffer_value->address();
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::transformer::split_query_key_value_and_split_heads::program
