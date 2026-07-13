// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "nlp_concat_heads_program_factory.hpp"
#include "nlp_concat_heads_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor NLPConcatHeadsProgramFactory::create_descriptor(
    const NlpConcatHeadsParams& /*operation_attributes*/, const Tensor& input, Tensor& output) {
    const auto& a = input;
    const auto& ashape = a.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    const auto& input_tile = a.tensor_spec().tile();
    const uint32_t input_tile_height = input_tile.get_height();
    const uint32_t input_tile_width = input_tile.get_width();
    const uint32_t input_tile_hw = input_tile.get_tile_hw();
    uint32_t single_tile_size = input_tile.get_tile_size(cb_data_format);

    tt_metal::Buffer* in0_buffer = a.buffer();
    bool in_sharded = a.is_sharded();
    bool out_sharded = output.is_sharded();

    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[1] * ashape[3] / input_tile_width;  // 142

    // Per output tensor args
    // Output shape is: [B, 1, s, 4544]
    uint32_t in0_h_tiles = ashape[2] / input_tile_height;
    uint32_t in0_w_tiles = ashape[3] / input_tile_width;  // head_dim
    uint32_t in0_c = per_tensor_tiles / in0_w_tiles;  // num_heads
    uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    uint32_t in0_CHtWt = in0_c * in0_HtWt;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[2] / input_tile_height;
    uint32_t num_cores = 0, num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores = CoreRangeSet(), core_group_1 = CoreRangeSet(), core_group_2 = CoreRangeSet();
    bool row_major = false;
    if (in_sharded) {
        all_cores = a.shard_spec().value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = a.shard_spec().value().shape[0] / a.padded_shape()[-2];
        per_tensor_tiles = a.shard_spec().value().shape[0] * a.shard_spec().value().shape[1] / input_tile_hw;
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
    ProgramDescriptor desc;
    uint32_t src0_cb_index = 0, out_cb_index = 16;

    KernelDescriptor reader_desc;
    KernelDescriptor writer_desc;
    if (in_sharded) {
        std::vector<uint32_t> compile_time_args = {
            (std::uint32_t)src0_cb_index,
            (std::uint32_t)out_cb_index,
            (std::uint32_t)in0_h_tiles,
            (std::uint32_t)in0_w_tiles * single_tile_size,
            (std::uint32_t)num_blocks_per_core_group_1 * in0_w_tiles * single_tile_size,
            (std::uint32_t)num_blocks_per_core_group_1 * in0_HtWt,
        };
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads_sharded.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = all_cores;
        reader_desc.compile_time_args = compile_time_args;
        reader_desc.config = ReaderConfigDescriptor{};

        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads_sharded.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = std::move(compile_time_args);
        writer_desc.config = WriterConfigDescriptor{};
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

        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_concat_heads.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = all_cores;
        reader_desc.compile_time_args = std::move(reader_compile_time_args);
        reader_desc.config = ReaderConfigDescriptor{};

        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = std::move(writer_compile_time_args);
        writer_desc.config = WriterConfigDescriptor{};
    }

    // Create circular buffers
    uint32_t cb_src0_num_tiles = per_tensor_tiles;
    if (!in_sharded) {
        cb_src0_num_tiles *= 2;  // double buffer
    }
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_src0_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
            .tile = input_tile}}},
        .buffer = in_sharded ? in0_buffer : nullptr,
    });

    if (out_sharded) {
        uint32_t cb_out_num_tiles = per_tensor_tiles;
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_out_num_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(out_cb_index),
                .data_format = cb_data_format,
                .page_size = single_tile_size,
                .tile = input_tile}}},
            .buffer = out_buffer,
        });
    }

    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
    if (in_sharded) {
        uint32_t nheads_first_risc = div_up(num_blocks_per_core_group_1, 2);
        uint32_t nheads_second_risc = num_blocks_per_core_group_1 - nheads_first_risc;
        // Mirror SetRuntimeArgs(program, kernel, all_cores, args) by emplacing the same
        // per-core args on every logical core in the sharded range set.
        for (const auto& core : corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)) {
            reader_desc.emplace_runtime_args(
                core,
                {
                    (std::uint32_t)nheads_first_risc,
                    uint32_t{0},
                    uint32_t{0},
                });
            writer_desc.emplace_runtime_args(
                core,
                {
                    (std::uint32_t)nheads_second_risc,
                    (std::uint32_t)nheads_first_risc * in0_HtWt * single_tile_size,
                    (std::uint32_t)nheads_first_risc * in0_w_tiles * single_tile_size,
                });
        }

    } else {
        for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
            const CoreCoord& core = cores[i];
            uint32_t num_blocks_per_core = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

            uint32_t in0_h_dim = num_blocks_written % in0_h_tiles;
            uint32_t in0_tensor_tile_id = (num_blocks_written / in0_h_tiles * in0_CHtWt) + (in0_h_dim * in0_w_tiles);

            reader_desc.emplace_runtime_args(
                core,
                {
                    in0_buffer,
                    num_blocks_per_core,  // num_blocks
                    in0_h_dim,            // in0_h_dim
                    in0_tensor_tile_id,   // in0_tensor_tile_id
                });

            writer_desc.emplace_runtime_args(
                core,
                {
                    out_buffer,  // out_tensor_addr
                    num_blocks_per_core * per_tensor_tiles,
                    num_blocks_written * per_tensor_tiles,
                });
            num_blocks_written += num_blocks_per_core;
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
