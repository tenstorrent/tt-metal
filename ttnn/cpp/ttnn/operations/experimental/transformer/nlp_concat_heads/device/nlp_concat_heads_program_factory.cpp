// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "nlp_concat_heads_program_factory.hpp"
#include "nlp_concat_heads_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::nlp_concat_heads::program {

using namespace tt::constants;
using namespace tt;

NLPConcatHeadsProgramFactory::cached_program_t NLPConcatHeadsProgramFactory::create(
    const NlpConcatHeadsParams& /*operation_attributes*/,
    const NlpConcatHeadsInputs& tensor_args,
    tensor_return_value_t& output) {
    const auto& a = tensor_args.input;
    const auto& ashape = a.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    bool in_sharded = a.is_sharded();
    bool out_sharded = output.is_sharded();

    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[1] * ashape[3] / TILE_WIDTH;  // 142

    // Per output tensor args
    // Output shape is: [B, 1, s, 4544]
    uint32_t in0_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t in0_w_tiles = ashape[3] / TILE_WIDTH;    // head_dim
    uint32_t in0_c = per_tensor_tiles / in0_w_tiles;  // num_heads
    uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    uint32_t in0_CHtWt = in0_c * in0_HtWt;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[2] / TILE_HEIGHT;
    uint32_t num_cores = 0, num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores = CoreRangeSet(), core_group_1 = CoreRangeSet(), core_group_2 = CoreRangeSet();
    bool row_major = false;
    if (in_sharded) {
        all_cores = a.shard_spec().value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = a.shard_spec().value().shape[0] / a.padded_shape()[-2];
        per_tensor_tiles = a.shard_spec().value().shape[0] * a.shard_spec().value().shape[1] / TILE_HW;
        row_major = a.shard_spec().value().orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
    }
    uint32_t g1_numcores = core_group_1.num_cores();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer* out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();
    uint32_t src0_cb_index = 0, out_cb_index = 16;

    tt::tt_metal::KernelHandle reader_kernel_id = 0, writer_kernel_id = 0;
    if (in_sharded) {
        std::vector<uint32_t> compile_time_args = {
            (std::uint32_t)src0_cb_index,
            (std::uint32_t)out_cb_index,
            (std::uint32_t)in0_h_tiles,
            (std::uint32_t)in0_w_tiles * single_tile_size,
            (std::uint32_t)num_blocks_per_core_group_1 * in0_w_tiles * single_tile_size,
            (std::uint32_t)num_blocks_per_core_group_1 * in0_HtWt,
        };
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads_sharded.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(compile_time_args));
        writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads_sharded.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(compile_time_args));
    } else {
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)in0_h_tiles,
            (std::uint32_t)in0_w_tiles,
            (std::uint32_t)in0_c,
            (std::uint32_t)in0_HtWt,
        };
        tt_metal::TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)src0_cb_index};
        tt_metal::TensorAccessorArgs(*out_buffer).append_to(writer_compile_time_args);
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

        writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }

    // Create circular buffers
    tt::tt_metal::CBHandle cb_src0 = 0, cb_out = 0;
    uint32_t cb_src0_num_tiles = per_tensor_tiles;
    if (!in_sharded) {
        cb_src0_num_tiles *= 2;  // double buffer
    }
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(cb_src0_num_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    if (in_sharded) {
        cb_src0_config = cb_src0_config.set_globally_allocated_address(*in0_buffer);
    }
    cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    if (out_sharded) {
        uint32_t cb_out_num_tiles = per_tensor_tiles;
        tt_metal::CircularBufferConfig cb_out_config =
            tt_metal::CircularBufferConfig(cb_out_num_tiles * single_tile_size, {{out_cb_index, cb_data_format}})
                .set_page_size(out_cb_index, single_tile_size);
        cb_out_config = cb_out_config.set_globally_allocated_address(*out_buffer);
        cb_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);
    }

    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
    if (in_sharded) {
        uint32_t nheads_first_risc = div_up(num_blocks_per_core_group_1, 2);
        uint32_t nheads_second_risc = num_blocks_per_core_group_1 - nheads_first_risc;
        std::vector<uint32_t> reader_runtime_args = {
            (std::uint32_t)nheads_first_risc,
            0,
            0,
        };
        tt_metal::SetRuntimeArgs(program, reader_kernel_id, all_cores, reader_runtime_args);
        std::vector<uint32_t> writer_runtime_args = {
            (std::uint32_t)nheads_second_risc,
            (std::uint32_t)nheads_first_risc * in0_HtWt * single_tile_size,
            (std::uint32_t)nheads_first_risc * in0_w_tiles * single_tile_size,
        };
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, all_cores, writer_runtime_args);

    } else {
        for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_blocks_per_core = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

            uint32_t in0_h_dim = num_blocks_written % in0_h_tiles;
            uint32_t in0_tensor_tile_id = (num_blocks_written / in0_h_tiles * in0_CHtWt) + (in0_h_dim * in0_w_tiles);

            std::vector<uint32_t> reader_runtime_args = {
                (std::uint32_t)in0_buffer->address(),
                num_blocks_per_core,  // num_blocks
                in0_h_dim,            // in0_h_dim
                in0_tensor_tile_id,   // in0_tensor_tile_id
            };

            std::vector<uint32_t> writer_runtime_args = {
                (std::uint32_t)out_buffer->address(),  // out_tensor_addr
                num_blocks_per_core * per_tensor_tiles,
                num_blocks_written * per_tensor_tiles,
            };

            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
            num_blocks_written += num_blocks_per_core;
        }
    }

    return cached_program_t{
        std::move(program),
        {/* reader_kernel_id = */ reader_kernel_id,
         /* writer_kernel_id = */ writer_kernel_id,
         /* cb_src0 = */ cb_src0,
         /* cb_out = */ cb_out,
         /* cores = */ cores,
         /* in_sharded = */ in_sharded,
         /* out_sharded = */ out_sharded}};
}

void NLPConcatHeadsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const NlpConcatHeadsParams& /*operation_attributes*/,
    const NlpConcatHeadsInputs& tensor_args,
    tensor_return_value_t& output) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = output.buffer();

    if (shared_vars.in_sharded) {
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_src0, *src_buffer);
    } else {
        for (const auto& core : shared_vars.cores) {
            auto& runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }
    }

    if (shared_vars.out_sharded) {
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_out, *dst_buffer);
    } else {
        for (const auto& core : shared_vars.cores) {
            auto& runtime_args = GetRuntimeArgs(program, shared_vars.writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads::program
