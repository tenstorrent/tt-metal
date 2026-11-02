// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_program_factory.hpp"

#include <algorithm>
#include <tuple>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

#include "tilize_codegen_device_operation.hpp"
#include "tilize_codegen_supported.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

constexpr const char* kChunkedReaderPath =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels/reader_tilize_rm_chunked.cpp";
constexpr const char* kCompactReaderPath =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels/reader_tilize_rm_specialized.cpp";
constexpr const char* kComputePath = "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels/compute_tilize.cpp";
constexpr const char* kTileWriterPath =
    "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/writer_interleaved.cpp";

constexpr const char* kReaderBlockPath =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels/reader_tilize_block.cpp";
constexpr const char* kTypecastComputePath =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels/compute_tilize_typecast.cpp";

constexpr uint32_t kCbIn = tt::CBIndex::c_0;
constexpr uint32_t kCbMid = tt::CBIndex::c_8;
constexpr uint32_t kCbOut = tt::CBIndex::c_16;

// Tiles the conversion kernel holds in DEST at once; fits the fp32 half-sync DEST.
constexpr uint32_t kTypecastBlockTiles = 4;

constexpr uint32_t k32BitElemSize = 4;
constexpr int kWidthAxis = -1;

CBDescriptor make_cb(
    uint32_t cb_id, tt::DataFormat format, uint32_t depth, uint32_t page_size, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = depth * page_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = format,
            .page_size = page_size,
        }}},
    };
}

}  // namespace

ProgramDescriptor TilizeCodegenProgramFactory::create_descriptor(
    const TilizeCodegenOperationAttributes& operation_attributes,
    const TilizeCodegenTensorArgs& tensor_args,
    const Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    const auto& plan = operation_attributes.plan;
    auto* device = input.device();
    Buffer* const in_buf = input.buffer();
    Buffer* const out_buf = tensor_return_value.buffer();

    const DataType dtype = input.dtype();
    const DataType out_dtype = tensor_return_value.dtype();
    const auto geometry = ttnn::operations::data_movement::tilize_codegen::tilize_geometry(input, out_dtype);
    const bool convert = geometry.convert;
    const bool compact_reader = ttnn::operations::data_movement::tilize_codegen::uses_compact_reader(geometry, plan);
    const uint32_t cb_mid_depth = plan.chunk_wt;
    const tt::DataFormat format = datatype_to_dataformat_converter(dtype);
    const tt::DataFormat out_format = datatype_to_dataformat_converter(out_dtype);
    const uint32_t elem_size = input.element_size();
    const uint32_t stick_bytes = input.padded_shape()[kWidthAxis] * elem_size;
    // Same tile sizes the gate budgets with; every admitted format's tile is already DRAM-aligned.
    const uint32_t in_cb_page_size = tt::tile_size(format);
    const uint32_t out_cb_page_size = tt::tile_size(out_format);
    const uint32_t in_page_size = in_buf->aligned_page_size();
    const uint32_t out_page_size = out_buf->aligned_page_size();
    const bool column_2d = geometry.route == ttnn::operations::data_movement::tilize_codegen::TilizeRoute::Column2D;

    const auto grid = device->compute_with_storage_grid_size();
    CoreRangeSet core_range;
    CoreRangeSet cg1;
    CoreRangeSet cg2;
    uint32_t rows_per_core_1 = 0;
    uint32_t rows_per_core_2 = 0;
    if (column_2d) {
        core_range = num_cores_to_corerangeset(geometry.tile_rows * geometry.num_col_blocks, grid, /*row_wise=*/true);
    } else {
        std::tie(std::ignore, core_range, cg1, cg2, rows_per_core_1, rows_per_core_2) =
            split_work_to_cores(grid, geometry.tile_rows, /*row_wise=*/true);
    }

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(kCbIn, format, plan.cb_in_depth, in_cb_page_size, core_range));
    if (convert) {
        desc.cbs.push_back(make_cb(kCbMid, format, cb_mid_depth, in_cb_page_size, core_range));
    }
    desc.cbs.push_back(make_cb(kCbOut, out_format, plan.cb_out_depth, out_cb_page_size, core_range));

    const bool chunked_reader = !column_2d && !compact_reader;
    KernelDescriptor chunked;
    KernelDescriptor reader;
    reader.core_ranges = core_range;
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.config = ReaderConfigDescriptor{};
    if (column_2d) {
        reader.kernel_source = kReaderBlockPath;
        reader.compile_time_args = {
            plan.num_col_chunks, TILE_HEIGHT, plan.chunk_wt, TILE_WIDTH * elem_size, in_page_size};
        TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
        reader.emplace_common_runtime_args({in_buf});
    } else if (compact_reader) {
        reader.kernel_source = kCompactReaderPath;
        reader.compile_time_args = {kCbIn, TILE_HEIGHT, stick_bytes, geometry.wt};
        TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
        reader.emplace_common_runtime_args({in_buf});
    } else {
        chunked.kernel_source = kChunkedReaderPath;
        chunked.source_type = KernelDescriptor::SourceType::FILE_PATH;
        chunked.core_ranges = core_range;
        chunked.config = ReaderConfigDescriptor{};
        chunked.compile_time_args = {kCbIn, in_page_size};
        TensorAccessorArgs(*in_buf).append_to(chunked.compile_time_args);
    }

    KernelDescriptor writer;
    writer.kernel_source = kTileWriterPath;
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = core_range;
    writer.compile_time_args = {kCbOut, out_page_size, plan.write_batch};
    TensorAccessorArgs(*out_buf).append_to(writer.compile_time_args);
    writer.config = WriterConfigDescriptor{};

    const bool wide_dtype = elem_size == k32BitElemSize || tensor_return_value.element_size() == k32BitElemSize;
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (dtype == DataType::FLOAT32) {
        unpack_to_dest_mode[kCbIn] = UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor compute;
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = core_range;
    if (convert) {
        compute.kernel_source = kTypecastComputePath;
        compute.compile_time_args = {
            kCbIn,
            kCbMid,
            kCbOut,
            plan.num_col_chunks,
            plan.chunk_wt,
            kTypecastBlockTiles,
            static_cast<uint32_t>(format),
            static_cast<uint32_t>(out_format)};
    } else {
        compute.kernel_source = kComputePath;
        compute.compile_time_args = {kCbIn, kCbOut, plan.num_col_chunks, plan.chunk_wt};
    }
    compute.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = wide_dtype,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    if (column_2d) {
        // Row-wise core order, matching num_cores_to_corerangeset above: core i owns tile row
        // i / num_col_blocks and column block i % num_col_blocks.
        constexpr uint32_t one_row = 1;
        const auto cores = corerange_to_cores(core_range, std::nullopt, true);
        for (uint32_t i = 0; i < cores.size(); ++i) {
            const uint32_t tile_row = i / geometry.num_col_blocks;
            const uint32_t col_block = i % geometry.num_col_blocks;
            const uint32_t first_stick = tile_row * TILE_HEIGHT;
            const uint32_t col_byte_offset = col_block * geometry.block_wt * TILE_WIDTH * elem_size;
            const uint32_t block_tiles = geometry.block_wt;
            const uint32_t first_tile = tile_row * geometry.wt + col_block * geometry.block_wt;
            reader.emplace_runtime_args(cores[i], {one_row, first_stick, col_byte_offset});
            writer.emplace_runtime_args(cores[i], {out_buf, block_tiles, first_tile});
            compute.emplace_runtime_args(cores[i], {one_row});
        }
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(compute));
        desc.kernels.push_back(std::move(writer));
        return desc;
    }

    uint32_t assigned = 0;
    auto emit_group = [&](const CoreRangeSet& group, uint32_t rows_per_core) {
        if (group.empty()) {
            return;
        }
        for (const auto& core : corerange_to_cores(group, std::nullopt, true)) {
            const uint32_t rows = std::min(rows_per_core, geometry.tile_rows - assigned);
            const uint32_t first_stick = assigned * TILE_HEIGHT;
            if (compact_reader) {
                reader.emplace_runtime_args(core, {rows, first_stick});
            } else {
                chunked.emplace_runtime_args(
                    core,
                    {in_buf,
                     rows,
                     first_stick,
                     TILE_HEIGHT,
                     plan.chunk_wt,
                     plan.num_col_chunks,
                     TILE_WIDTH * elem_size});
            }
            writer.emplace_runtime_args(core, {out_buf, rows * geometry.wt, assigned * geometry.wt});
            compute.emplace_runtime_args(core, {rows});
            assigned += rows;
        }
    };
    emit_group(cg1, rows_per_core_1);
    emit_group(cg2, rows_per_core_2);

    desc.kernels.push_back(std::move(chunked_reader ? chunked : reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

}  // namespace ttnn::prim
