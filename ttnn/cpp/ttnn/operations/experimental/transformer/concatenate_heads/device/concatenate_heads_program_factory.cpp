// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads_program_factory.hpp"

#include "concatenate_heads_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::transformer::program {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt;

ConcatenateHeadsProgramFactory::cached_program_t ConcatenateHeadsProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& ashape = a.padded_shape();
    const auto& compute_with_storage_grid_size = operation_attributes.compute_with_storage_grid_size;

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // Output shape is: [B, 1, 384, 1024]
    uint32_t per_core_tiles = (ashape[1] * ashape[3]) / TILE_WIDTH;
    uint32_t in0_h_tiles = ashape[2] / TILE_HEIGHT;

    // These parameters are identical to out_* in multi_core_create_qkv_heads
    uint32_t in0_w = 64;
    uint32_t in0_w_tiles = in0_w / TILE_WIDTH;
    uint32_t in0_c = per_core_tiles / in0_w_tiles;
    uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    uint32_t in0_CHtWt = in0_c * in0_HtWt;

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
    tt_metal::Buffer* out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

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
        (std::uint32_t)in0_w_tiles,  // in0_w_tiles
        (std::uint32_t)in0_c,        // in0_c
        (std::uint32_t)in0_HtWt,     // in0_HtWt
    };
    tt::tt_metal::TensorAccessorArgs(in0_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {
        // WRITER COMPILE TIME ARGS
        (std::uint32_t)in0_w_tiles,  // in0_w_tiles
        (std::uint32_t)in0_c,        // in0_c
    };
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_concat_heads.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads/device/kernels/dataflow/"
        "writer_tm_tile_layout_concat_heads.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    uint32_t cb0_tiles = per_core_tiles * 2;  // double buffer
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            uint32_t in0_tensor_tile_id = (core_idx_x * in0_w_tiles) + (core_idx_y * in0_CHtWt);

            std::vector<uint32_t> reader_runtime_args = {
                (std::uint32_t)in0_buffer->address(),  // in0_tensor_addr
                in0_tensor_tile_id,                    // in0_tensor_tile_id
            };
            std::vector<uint32_t> writer_runtime_args = {
                (std::uint32_t)out_buffer->address(),                      // out_tensor_addr
                (core_idx_x + core_idx_y * num_cores_c) * per_core_tiles,  // out_tensor_tile_id
            };

            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }
    }

    return cached_program_t{
        std::move(program),
        {/* reader_kernel_id = */ reader_kernel_id,
         /* writer_kernel_id = */ writer_kernel_id,
         /* num_cores_r      = */ num_cores_r,
         /* num_cores_c      = */ num_cores_c}};
}

void ConcatenateHeadsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& output) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* src_dram_buffer = tensor_args.input.buffer();
    auto* dst_dram_buffer = output.buffer();

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    for (int core_idx_y = 0; core_idx_y < shared_vars.num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < shared_vars.num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

            {
                auto& runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_id, core);
                runtime_args[0] = src_dram_buffer->address();
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, shared_vars.writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer->address();
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::transformer::program
