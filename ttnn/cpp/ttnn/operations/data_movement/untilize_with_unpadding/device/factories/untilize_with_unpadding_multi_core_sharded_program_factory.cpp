// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_sharded_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor UntilizeWithUnpaddingMultiCoreShardedProgramFactory::create_descriptor(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    ProgramDescriptor desc;

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();
    const auto& tile = a.tensor_spec().tile();
    const uint32_t tile_height = tile.get_height();
    const uint32_t tile_width = tile.get_width();
    // Special handling for tensors of W=16 and H%tile_height==0 on standard 32x32 tiles.
    // In this case skip untilizing on compute and in writer kernel just copy face0 and face2,
    // and skip face1 and face3.
    bool unpad_tensor_w_16 = output.padded_shape()[-1] == 16 && output.padded_shape()[-2] % tile_height == 0 &&
                             tile_height == TILE_HEIGHT && tile_width == TILE_WIDTH;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tile.get_tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tile.get_tile_size(output_cb_data_format);

    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0,
             num_output_rows_unpadded = 0;
    CoreCoord end_core;
    uint32_t last_idx = 0;
    auto shard_spec = a.shard_spec().value();

    // I am not sure it is correct to ever use the shard_spec here.
    auto out_shard_spec = output.shard_spec().has_value() ? output.shard_spec().value() : shard_spec;

    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto all_cores = shard_spec.grid;
    uint32_t ntiles_per_block = shard_spec.shape[1] / tile_width;
    uint32_t nblocks_per_core = shard_spec.shape[0] / tile_height;
    uint32_t global_batch = a.physical_volume() / (a.padded_shape()[-2] * a.padded_shape()[-1]);
    uint32_t batch =
        a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED
            ? std::max(1u, (shard_spec.shape[0] * shard_spec.shape[1]) / (a.padded_shape()[-2] * a.padded_shape()[-1]))
            : global_batch;
    uint32_t ntiles_per_batch = ntiles_per_block * nblocks_per_core / batch;

    num_rows_block = out_shard_spec.shape[0];
    block_row_size = out_shard_spec.shape[1] * output.element_size();     // in0_block_w * TILE_WIDTH * dtype_nbytes
    output_row_size = output.padded_shape()[-1] * output.element_size();  // output row size bytes
    last_block_row_size_unpadded = block_row_size - (tt::round_up(output.padded_shape()[-1], out_shard_spec.shape[1]) -
                                                     output.padded_shape()[-1]) *
                                                        output.element_size();
    uint32_t num_output_rows = output.physical_volume() / output.padded_shape()[-1];
    num_output_rows_unpadded =
        num_rows_block - (tt::round_up(num_output_rows, out_shard_spec.shape[0]) - num_output_rows);
    if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        last_idx = tt::div_up(output.padded_shape()[-1], out_shard_spec.shape[1]) - 1;
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        last_idx = tt::div_up(num_output_rows, out_shard_spec.shape[0]) - 1;
    } else {
        end_core = {
            tt::div_up(output.padded_shape()[-1], out_shard_spec.shape[1]) - 1,
            tt::div_up(num_output_rows, out_shard_spec.shape[0]) - 1};
    }
    if (!row_major) {
        std::swap(end_core.x, end_core.y);
    }

    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    // Input CB: sharded → bind .buffer to the input buffer; framework re-applies
    // UpdateDynamicCircularBufferAddress on cache hit.
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = TileDescriptor(tile),
        }}},
        .buffer = src_sharded ? a.buffer() : nullptr,
    });

    uint32_t num_output_tiles = out_sharded ? (unpad_tensor_w_16 ? 16 : ntiles_per_batch * 2) : ntiles_per_block * 2;
    uint32_t aligned_page_size = static_cast<uint32_t>(output.buffer()->aligned_page_size());
    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
            .tile = TileDescriptor(tile),
        }}},
    });

    constexpr uint8_t sharded_output_cb_index = tt::CBIndex::c_17;
    if (out_sharded) {
        // The kernel advances the write pointer by aligned_page_size (which may be
        // larger than block_row_size due to buffer alignment padding), so the CB
        // page size must match to avoid overflow.
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_output_rows_unpadded * aligned_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = sharded_output_cb_index,
                .data_format = output_cb_data_format,
                .page_size = aligned_page_size,
            }}},
            .buffer = output.buffer(),
        });
    }

    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    std::vector<uint32_t> reader_ct_args;
    reader_ct_args.push_back(src0_cb_index);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    /** writer
     */
    KernelDescriptor writer_desc;
    if (out_sharded) {
        std::vector<uint32_t> writer_ct_args{output_cb_index, sharded_output_cb_index, aligned_page_size};
        writer_desc.kernel_source =
            unpad_tensor_w_16
                ? "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                  "writer_unary_unpad_width_16_sharded.cpp"
                : "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                  "writer_unary_unpad_batch_rows_sharded.cpp";
        writer_desc.compile_time_args = std::move(writer_ct_args);
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        // Height-sharded -> interleaved uses a dedicated writer that walks each core's absolute rows
        // and drops both interior (row) and column padding per matrix. It handles any alignment of
        // matrices to core boundaries (whole matrices per core, a single matrix split across cores,
        // or a batch whose matrices straddle cores), so there is no unbatched restriction.
        std::vector<uint32_t> writer_ct_args = {tile_height};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_unpad_sharded_to_interleaved.cpp";
        writer_desc.compile_time_args = std::move(writer_ct_args);
    } else {
        std::vector<uint32_t> writer_ct_args = {
            (input_cb_data_format == tt::DataFormat::Float32 or input_cb_data_format == tt::DataFormat::UInt32 or
             input_cb_data_format == tt::DataFormat::Int32),
            output_row_size};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
        writer_desc.kernel_source = "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp";
        writer_desc.compile_time_args = std::move(writer_ct_args);
    }
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.config = WriterConfigDescriptor{};

    /** compute
     */
    std::vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
        (uint32_t)src0_cb_index,
        (uint32_t)output_cb_index,
    };

    KernelDescriptor::Defines compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    std::string compute_kernel("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
    if (unpad_tensor_w_16) {
        // Use copy compute kernel just for a potential data type conversion.
        compute_kernel = "ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp";
        compute_args[0] = (uint32_t)num_input_tiles;  // per_core_tile_cnt
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_kernel;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_args);
    compute_desc.defines = std::move(compute_kernel_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    // Runtime args: legacy code uses SetRuntimeArgs(program, kernel, all_cores, args) which broadcasts
    // the same args to every core. The descriptor API needs per-core entries; enumerate cores and
    // emit one runtime_args entry per core.
    const std::vector<CoreCoord> all_core_coords = corerange_to_cores(all_cores, std::nullopt, row_major);

    const std::vector<uint32_t> reader_rt_args = {ntiles_per_block * nblocks_per_core};
    reader_desc.runtime_args.reserve(all_core_coords.size());
    for (const auto& core : all_core_coords) {
        reader_desc.runtime_args.emplace_back(core, reader_rt_args);
    }

    if (out_sharded) {
        std::vector<uint32_t> writer_rt_args;
        if (unpad_tensor_w_16) {
            writer_rt_args = {num_output_rows_unpadded, num_input_tiles};
        } else {
            writer_rt_args = {
                num_output_rows_unpadded,
                ntiles_per_batch,
                out_shard_spec.shape[0] / batch,
                shard_spec.shape[1] * output.element_size(),
                block_row_size,
                batch};
        }
        writer_desc.runtime_args.reserve(all_core_coords.size());
        for (const auto& core : all_core_coords) {
            writer_desc.runtime_args.emplace_back(core, writer_rt_args);
        }
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        // General height-sharded -> interleaved. Each core owns the absolute padded rows
        // [i * shard_h, (i + 1) * shard_h) of the flattened [global_batch * H_padded, W_padded] row
        // space (enumerated in shard/core order). The kernel maps every row to its (matrix,
        // row-in-matrix) and writes the real rows to their logical interleaved page, dropping both
        // interior (row) and column padding. This is correct for any alignment of matrices to core
        // boundaries, so there is no unbatched restriction.
        const uint32_t matrix_h_padded = a.padded_shape()[-2];
        const uint32_t matrix_h_logical = output.logical_shape()[-2];
        const uint32_t shard_height = shard_spec.shape[0];
        const uint32_t cb_row_size = shard_spec.shape[1] * output.element_size();  // CB row stride (padded width)
        const uint32_t row_size_unpadded =
            output.logical_shape()[-1] * output.element_size();  // bytes written per row (logical width)

        writer_desc.runtime_args.reserve(all_core_coords.size());
        for (uint32_t i = 0; i < all_core_coords.size(); ++i) {
            const CoreCoord& core = all_core_coords[i];
            const uint32_t start_padded_row = i * shard_height;
            writer_desc.emplace_runtime_args(
                core,
                {dst_buffer,  // dst_addr
                 start_padded_row,
                 shard_height,
                 matrix_h_padded,
                 matrix_h_logical,
                 cb_row_size,
                 row_size_unpadded,
                 ntiles_per_block});
        }
    } else {
        writer_desc.runtime_args.reserve(all_core_coords.size());
        for (uint32_t i = 0; i < all_core_coords.size(); ++i) {
            CoreCoord core = all_core_coords[i];

            // writer runtime args
            uint32_t block_start_row_offset;
            uint32_t block_start_row_id_offset;
            uint32_t row_size_unpadded = block_row_size;
            uint32_t num_rows_unpadded = num_rows_block;
            if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
                block_start_row_offset = i * block_row_size;
                block_start_row_id_offset = 0;
                if (i > last_idx) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                } else {
                    num_rows_unpadded = num_output_rows_unpadded;
                    if (i == last_idx) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                }
            } else {
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }
                if (core.x > end_core.x || core.y > end_core.y) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                }
            }

            writer_desc.emplace_runtime_args(
                core,
                {dst_buffer,  // dst_addr
                 num_rows_block,
                 block_row_size,
                 std::uint32_t{1},
                 std::uint32_t{1},
                 std::uint32_t{1},
                 row_size_unpadded,
                 num_rows_unpadded,
                 block_start_row_id_offset,
                 block_start_row_offset});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
