// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "nlp_create_qkv_heads_segformer_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;

tt::tt_metal::operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_segformer(
    const Tensor& a, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size) {
    const auto& ashape = a.get_padded_shape();

    tt_metal::IDevice* device = a.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);
    // Dummy
    tt_metal::Buffer* in1_buffer;
    uint32_t in1_buffer_addr = 0;

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t head_dim = 32;
    uint32_t per_tensor_tiles = ashape[3] / TILE_WIDTH;
    const uint32_t q_num_tiles_per_tensor = per_tensor_tiles;
    const uint32_t num_q_heads = q_num_tiles_per_tensor;  // hard-coding the head_dim = 32

    // Per output tensor args
    // Output shape for Q/K/V is: [B, head_num, s, 32] # Needs shuffling from [B, 1, s, hidden_dim]
    uint32_t q_out_h_tiles = ashape[2] / TILE_WIDTH;
    uint32_t q_out_w_tiles = 1;                                 // hard-coding the head_dim = 32
    uint32_t q_out_c = q_num_tiles_per_tensor / q_out_w_tiles;  // num_heads
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = q_out_c * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[1] * ashape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
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

    bool tile_dtype_is_bfloat16 = a.get_dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool out_is_dram = q_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool in1_is_dram = false;

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)in1_is_dram,
        (std::uint32_t)q_num_tiles,
    };
    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)out_is_dram,
        (std::uint32_t)q_out_h_tiles,
        (std::uint32_t)q_out_w_tiles,
        (std::uint32_t)q_out_HtWt,
        (std::uint32_t)num_q_heads,  // q_out_c
    };

    ///////////// K transpose ////////////////////
    const bool transpose_k_heads = false;
    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    //////////////////////////////////////////////
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_create_qkv_heads.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/device/kernels/dataflow/"
        "writer_tm_tile_layout_nlp_create_qkv_heads.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    // Create circular buffers
    uint32_t src1_cb_index = 1;
    uint32_t cb0_num_tiles = per_tensor_tiles * 2;  // double buffer
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(cb0_num_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_runtime_args = {
            (std::uint32_t)in0_buffer->address(),
            (std::uint32_t)in1_buffer_addr,
            num_blocks_per_core,
            num_blocks_written * per_tensor_tiles,
            0,
        };

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id = num_blocks_written / q_out_h_tiles * q_out_CHtWt + q_out_h_dim * q_out_w_tiles;

        std::vector<uint32_t> writer_runtime_args = {
            (std::uint32_t)q_buffer->address(),  // q_tensor_addr
            num_blocks_per_core,                 // num_blocks
            q_out_h_dim,
            q_out_tensor_tile_id,
        };

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        num_blocks_written += num_blocks_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, num_cores, num_cores_y](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_dram_buffer = input_tensors.at(0).buffer();

        auto dst_dram_buffer_query = output_tensors.at(0).buffer();

        for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_dram_buffer->address();
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer_query->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::experimental::transformer
