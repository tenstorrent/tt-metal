// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_program_factory.hpp"

#include <algorithm>
#include <variant>
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

constexpr uint32_t kCbIn = tt::CBIndex::c_0;
constexpr uint32_t kCbOut = tt::CBIndex::c_16;

constexpr uint32_t k32BitElemSize = 4;

namespace codegen = ttnn::operations::data_movement::tilize_codegen;

using RtArgs = std::vector<std::variant<uint32_t, Buffer*>>;

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
    const auto geometry = codegen::tilize_geometry(input);
    const bool compact_reader = codegen::uses_compact_reader(geometry, plan);
    const tt::DataFormat format = datatype_to_dataformat_converter(dtype);
    const uint32_t elem_size = input.element_size();
    const uint32_t stick_bytes = input.padded_shape()[codegen::kWidthAxis] * elem_size;
    const uint32_t fragment_bytes = TILE_WIDTH * elem_size;
    // Same tile size the gate budgets with; every admitted format's tile is already DRAM-aligned.
    const uint32_t cb_page_size = tt::tile_size(format);
    const uint32_t in_page_size = in_buf->aligned_page_size();
    const uint32_t out_page_size = out_buf->aligned_page_size();
    const bool column_2d = geometry.route == codegen::TilizeRoute::Column2D;

    const auto grid = device->compute_with_storage_grid_size();
    const auto [num_row_cores, row_cores, cg1, cg2, rows_per_core_1, rows_per_core_2] =
        split_work_to_cores(grid, geometry.tile_rows, /*row_wise=*/true);
    const CoreRangeSet core_range =
        column_2d ? num_cores_to_corerangeset(geometry.tile_rows * geometry.num_col_blocks, grid, /*row_wise=*/true)
                  : row_cores;

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(kCbIn, format, plan.cb_in_depth, cb_page_size, core_range));
    desc.cbs.push_back(make_cb(kCbOut, format, plan.cb_out_depth, cb_page_size, core_range));

    const bool chunked_reader = !column_2d && !compact_reader;
    KernelDescriptor chunked;
    KernelDescriptor reader;
    reader.core_ranges = core_range;
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.config = ReaderConfigDescriptor{};
    if (column_2d) {
        reader.kernel_source = kReaderBlockPath;
        reader.named_compile_time_args = {
            {"cb_in", kCbIn},
            {"num_col_chunks", plan.num_col_chunks},
            {"tile_height", TILE_HEIGHT},
            {"chunk_wt", plan.chunk_wt},
            {"fragment_bytes", fragment_bytes},
            {"aligned_page_size", in_page_size},
        };
        TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
        reader.emplace_common_runtime_args({in_buf});
    } else if (compact_reader) {
        reader.kernel_source = kCompactReaderPath;
        reader.named_compile_time_args = {
            {"cb_in", kCbIn},
            {"tile_height", TILE_HEIGHT},
            {"stick_bytes", stick_bytes},
            {"tiles_per_row", geometry.wt},
        };
        TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
        reader.emplace_common_runtime_args({in_buf});
    } else {
        chunked.kernel_source = kChunkedReaderPath;
        chunked.source_type = KernelDescriptor::SourceType::FILE_PATH;
        chunked.core_ranges = core_range;
        chunked.config = ReaderConfigDescriptor{};
        chunked.named_compile_time_args = {
            {"cb_in", kCbIn},
            {"tile_height", TILE_HEIGHT},
            {"chunk_wt", plan.chunk_wt},
            {"num_col_chunks", plan.num_col_chunks},
            {"fragment_bytes", fragment_bytes},
            {"aligned_page_size", in_page_size},
        };
        TensorAccessorArgs(*in_buf).append_to(chunked.compile_time_args);
        chunked.emplace_common_runtime_args({in_buf});
    }

    // The shared writer takes the output address as per-core runtime arg 0, bound to the Buffer so a program-cache
    // hit patches it on every core.
    KernelDescriptor writer;
    writer.kernel_source = kTileWriterPath;
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = core_range;
    writer.compile_time_args = {kCbOut, out_page_size, plan.write_batch};
    TensorAccessorArgs(*out_buf).append_to(writer.compile_time_args);
    writer.config = WriterConfigDescriptor{};

    // A 16-bit DEST would truncate 32-bit datums, float or integer, on their way to the packer.
    const bool wide_dtype = elem_size == k32BitElemSize;
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    // The tilize unpacker sends INT32/UINT32 straight to DEST on its own; FLOAT32 does so only when its unpack
    // destination format stays FLOAT32, which this mode requests.
    if (dtype == DataType::FLOAT32) {
        unpack_to_dest_mode[kCbIn] = UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor compute;
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = core_range;
    compute.kernel_source = kComputePath;
    compute.named_compile_time_args = {
        {"cb_in", kCbIn},
        {"cb_out", kCbOut},
        {"num_col_chunks", plan.num_col_chunks},
        {"chunk_wt", plan.chunk_wt},
    };
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
            const uint32_t col_byte_offset = col_block * geometry.block_wt * fragment_bytes;
            const uint32_t block_tiles = geometry.block_wt;
            const uint32_t first_tile = tile_row * geometry.wt + col_block * geometry.block_wt;
            reader.emplace_runtime_args(cores[i], {one_row, first_stick, col_byte_offset});
            writer.emplace_runtime_args(cores[i], RtArgs{out_buf, block_tiles, first_tile});
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
                chunked.emplace_runtime_args(core, {rows, first_stick});
            }
            writer.emplace_runtime_args(core, RtArgs{out_buf, rows * geometry.wt, assigned * geometry.wt});
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
