// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;

NlpCreateQkvHeadsVitProgramFactory::cached_program_t NlpCreateQkvHeadsVitProgramFactory::create(
    const NlpCreateQkvHeadsVitParams& /*operation_attributes*/,
    const NlpCreateQkvHeadsVitInputs& tensor_args,
    NlpCreateQkvHeadsVitResult& output) {
    const auto& a = tensor_args.input_tensor;
    const auto& ashape = a.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);
    // Dummy
    uint32_t in1_buffer_addr = 0;

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[3] / TILE_WIDTH;  // 72
    const uint32_t q_num_tiles_per_tensor = 24;
    const uint32_t num_q_heads = 12;
    const uint32_t num_kv_heads = 12;

    // Per output tensor args
    // Output shape for Q,K,V is: [B, 12, s, 64] # Needs shuffling from [B, 1, s, 2304]
    uint32_t q_out_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = 2;                                 // head_dim
    uint32_t q_out_c = q_num_tiles_per_tensor / q_out_w_tiles;  // num_heads
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = q_out_c * q_out_HtWt;
    uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;
    uint32_t kv_num_tiles = num_kv_heads * q_out_w_tiles;

    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[1] * ashape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
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

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)q_num_tiles,
        (std::uint32_t)kv_num_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(in0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)q_out_h_tiles,
        (std::uint32_t)q_out_w_tiles,
        (std::uint32_t)q_out_HtWt,
        (std::uint32_t)num_q_heads,   // q_out_c
        (std::uint32_t)num_kv_heads,  // kv_out_c
    };
    tt::tt_metal::TensorAccessorArgs(q_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(k_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(v_buffer).append_to(writer_compile_time_args);

    ///////////// K transpose ////////////////////
    const bool transpose_k_heads = false;
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    if (transpose_k_heads) {
        std::vector<uint32_t> compute_args_core_group_1 = {num_blocks_per_core_group_1 * kv_num_tiles};
        tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/transpose_wh.cpp",
            core_group_1,
            tt_metal::ComputeConfig{.compile_args = compute_args_core_group_1});

        if (core_group_2.num_cores() > 0) {
            std::vector<uint32_t> compute_args_core_group_2 = {num_blocks_per_core_group_2 * kv_num_tiles};
            tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/transpose_wh.cpp",
                core_group_2,
                tt_metal::ComputeConfig{.compile_args = compute_args_core_group_2});
        }
        reader_defines["TRANSPOSE_K_HEADS"] = "1";
        writer_defines["TRANSPOSE_K_HEADS"] = "1";
    }
    //////////////////////////////////////////////

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_create_qkv_heads.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/kernels/dataflow/"
        "writer_tm_tile_layout_nlp_create_qkv_heads.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    // Create circular buffers
    uint32_t src1_cb_index = 1;
    uint32_t cb0_num_tiles = per_tensor_tiles * 2;  // double buffer
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(cb0_num_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    // If we transpose_k_heads:
    // - reader will write to cb0, instead of cb1
    // - compute will wait on cb0 and write to cb16
    // - writer will wait on cb 16, instead of cb1
    if (transpose_k_heads) {
        uint32_t src0_cb_index = 0;
        uint32_t cb0_num_tiles = per_tensor_tiles * 2;  // double buffer
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(cb0_num_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
                .set_page_size(src0_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

        uint32_t out_cb_index = 16;
        uint32_t out_cb_num_tiles = per_tensor_tiles * 2;  // double buffer
        tt_metal::CircularBufferConfig cb_out_config =
            tt_metal::CircularBufferConfig(out_cb_num_tiles * single_tile_size, {{out_cb_index, cb_data_format}})
                .set_page_size(out_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);
    }

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
        uint32_t q_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * q_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t v_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t k_out_tensor_tile_id = transpose_k_heads
                                            ? (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + q_out_h_dim
                                            : v_out_tensor_tile_id;

        std::vector<uint32_t> writer_runtime_args = {
            (std::uint32_t)q_buffer->address(),  // q_tensor_addr
            (std::uint32_t)k_buffer->address(),  // k_tensor_addr
            (std::uint32_t)v_buffer->address(),  // v_tensor_addr
            num_blocks_per_core,                 // num_blocks
            q_out_h_dim,
            q_out_tensor_tile_id,
            k_out_tensor_tile_id,
            v_out_tensor_tile_id,
        };

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        num_blocks_written += num_blocks_per_core;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void NlpCreateQkvHeadsVitProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const NlpCreateQkvHeadsVitParams& /*operation_attributes*/,
    const NlpCreateQkvHeadsVitInputs& tensor_args,
    NlpCreateQkvHeadsVitResult& output) {
    auto& program = cached_program.program;
    const auto& shared_variables = cached_program.shared_variables;

    auto* src_dram_buffer = tensor_args.input_tensor.buffer();
    auto* dst_dram_buffer_query = output.at(0).buffer();
    auto* dst_dram_buffer_key = output.at(1).buffer();
    auto* dst_dram_buffer_value = output.at(2).buffer();

    for (uint32_t i = 0; i < shared_variables.num_cores; i++) {
        CoreCoord core = {i / shared_variables.num_cores_y, i % shared_variables.num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, shared_variables.reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, shared_variables.writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer_query->address();
            runtime_args[1] = dst_dram_buffer_key->address();
            runtime_args[2] = dst_dram_buffer_value->address();
        }
    }
}

}  // namespace ttnn::experimental::prim
