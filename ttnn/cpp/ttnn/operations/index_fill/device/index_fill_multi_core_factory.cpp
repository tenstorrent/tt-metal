// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "index_fill_device_operation.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::index_fill {

using namespace tt;
using namespace tt::tt_metal;

namespace {

// The input rows [start_row, end_row) one core reads, as pages row_id * row_page_stride + col_shard_id.
struct CoreWork {
    CoreCoord core;
    uint32_t start_row{};
    uint32_t end_row{};
    uint32_t col_shard_id{};     // column shard index (0 for HEIGHT/INTERLEAVED)
    uint32_t row_page_stride{};  // KW for WIDTH/BLOCK, 1 for HEIGHT/INTERLEAVED
};

struct WorkSplit {
    CoreRangeSet all_cores;
    std::vector<CoreWork> cores;
};

WorkSplit split_rows_across_cores(const ttnn::Tensor& input, uint32_t num_rows) {
    const auto clamp_row = [num_rows](uint32_t row) { return std::min(row, num_rows); };
    const auto input_mem_layout = input.memory_config().memory_layout();

    WorkSplit split;
    switch (input_mem_layout) {
        case TensorMemoryLayout::INTERLEAVED: {
            CoreRangeSet core_group_1;
            uint32_t num_rows_per_core_group_1{};
            uint32_t num_rows_per_core_group_2{};
            std::tie(
                std::ignore,
                split.all_cores,
                core_group_1,
                std::ignore,
                num_rows_per_core_group_1,
                num_rows_per_core_group_2) =
                split_work_to_cores(input.device()->compute_with_storage_grid_size(), num_rows);

            uint32_t start_row = 0;
            for (const auto& core : corerange_to_cores(split.all_cores, std::nullopt, false)) {
                const uint32_t rows =
                    core_group_1.contains(core) ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
                split.cores.push_back({core, start_row, start_row + rows, 0, 1});
                start_row += rows;
            }
            break;
        }
        case TensorMemoryLayout::HEIGHT_SHARDED: {
            const auto& shard_spec = input.shard_spec().value();
            split.all_cores = shard_spec.grid;
            const uint32_t shard_h = shard_spec.shape[0];

            uint32_t start_row = 0;
            for (const auto& core : corerange_to_cores(split.all_cores, std::nullopt, false)) {
                split.cores.push_back({core, clamp_row(start_row), clamp_row(start_row + shard_h), 0, 1});
                start_row += shard_h;
            }
            break;
        }
        case TensorMemoryLayout::WIDTH_SHARDED: {
            const auto& shard_spec = input.shard_spec().value();
            split.all_cores = shard_spec.grid;
            const auto cores = corerange_to_cores(
                split.all_cores, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);

            const uint32_t KW = cores.size();
            for (uint32_t k = 0; k < KW; ++k) {
                split.cores.push_back({cores[k], 0, num_rows, k, KW});
            }
            break;
        }
        case TensorMemoryLayout::BLOCK_SHARDED: {
            const auto& shard_spec = input.shard_spec().value();
            split.all_cores = shard_spec.grid;
            const bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
            const auto cores = corerange_to_cores(split.all_cores, std::nullopt, rm_orientation);

            const uint32_t shard_h = shard_spec.shape[0];
            const uint32_t shard_w = shard_spec.shape[1];
            const uint32_t KW = (input.padded_shape()[-1] + shard_w - 1) / shard_w;
            const uint32_t KH = (num_rows + shard_h - 1) / shard_h;
            for (uint32_t i = 0; i < cores.size(); ++i) {
                const uint32_t kh = rm_orientation ? (i / KW) : (i % KH);
                const uint32_t kw = rm_orientation ? (i % KW) : (i / KH);
                const uint32_t row_start = kh * shard_h;
                split.cores.push_back({cores[i], clamp_row(row_start), clamp_row(row_start + shard_h), kw, KW});
            }
            break;
        }
        default: TT_THROW("Index fill: Unsupported input memory layout: {}", input_mem_layout);
    }
    return split;
}

// How the writer places each input page into the output; fixed by the input and output memory layouts.
enum class OutputWriteMode {
    // WIDTH/BLOCK -> WIDTH/BLOCK (WIDTH<->WIDTH, BLOCK<->BLOCK, WIDTH<->BLOCK). Validation guarantees matching
    // column shard width, so input KW == output KW and the page formula row*KW+col is identical on both sides.
    // One write per row; the output TensorAccessor routes each page to the physically-owning core (which may
    // differ from the executing core when converting between WIDTH and BLOCK row layouts).
    ColToCol,
    // INTERLEAVED/HEIGHT -> WIDTH/BLOCK: one core splits each full input row into out_KW fragments and writes
    // each to its respective output shard.
    RowsToCols,
    // Any -> INTERLEAVED/HEIGHT: write the input fragment at a byte offset into the full output row.
    ToRows,
};

OutputWriteMode output_write_mode(TensorMemoryLayout input_layout, TensorMemoryLayout output_layout) {
    if (!is_col_sharded(output_layout)) {
        return OutputWriteMode::ToRows;
    }
    return is_col_sharded(input_layout) ? OutputWriteMode::ColToCol : OutputWriteMode::RowsToCols;
}

// The writer addresses output pages as page_id = row_id * row_page_stride + col_shard_id + kout,
// for kout in [0, num_col_shards), writing at col_byte_offset within each page.
struct OutputWriteArgs {
    uint32_t col_shard_id{};
    uint32_t row_page_stride{};
    uint32_t col_byte_offset{};
    uint32_t num_col_shards{};
};

OutputWriteArgs output_write_args(
    OutputWriteMode mode, const CoreWork& work, uint32_t out_KW, uint32_t input_page_size) {
    switch (mode) {
        case OutputWriteMode::ColToCol: return {work.col_shard_id, work.row_page_stride, 0, 1};
        case OutputWriteMode::RowsToCols: return {0, out_KW, 0, out_KW};
        case OutputWriteMode::ToRows: return {0, 1, work.col_shard_id * input_page_size, 1};
    }
    TT_THROW("Index fill: Unknown output write mode");
}

CBDescriptor make_cb(
    CBIndex cb_id, uint32_t total_size, DataFormat data_format, uint32_t page_size, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = total_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_id,
            .data_format = data_format,
            .page_size = page_size,
        }}},
    };
}

}  // namespace

tt::tt_metal::ProgramDescriptor IndexFillOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const ttnn::Tensor& index = tensor_args.index;
    const ttnn::Tensor& input = tensor_args.input;
    uint32_t dim = operation_attributes.dim;

    const auto input_shape = input.padded_shape();
    const auto n = input_shape.rank();
    uint32_t num_rows_in_dim = 1;
    for (int i = n - 2; i > static_cast<int>(dim); --i) {
        num_rows_in_dim *= input_shape[i];
    }
    const uint32_t dim_size = input_shape[dim];

    // fill_value goes to the kernel as a uint32_t
    const auto& value = operation_attributes.value;
    uint32_t fill_value{};
    switch (input.dtype()) {
        case DataType::BFLOAT16:
            fill_value = pack_two_bfloat16_into_uint32({bfloat16(std::get<float>(value)), bfloat16(0.0f)});
            break;
        case DataType::FLOAT32: fill_value = std::bit_cast<uint32_t>(std::get<float>(value)); break;
        case DataType::INT32: fill_value = static_cast<uint32_t>(std::get<int>(value)); break;
        default: TT_FATAL(false, "Unsupported datatype"); break;
    }

    // Distribute work across core grid
    const uint32_t num_rows = static_cast<uint32_t>(input.physical_volume() / input.padded_shape()[-1]);
    const WorkSplit split = split_rows_across_cores(input, num_rows);
    const CoreRangeSet& all_cores = split.all_cores;

    ProgramDescriptor desc;

    // Create circular buffers
    auto input_dataformat = datatype_to_dataformat_converter(input.dtype());
    auto index_dataformat = datatype_to_dataformat_converter(index.dtype());

    uint32_t input_page_size = input.mesh_buffer().aligned_page_size();
    uint32_t index_total_size = index.mesh_buffer().aligned_size();

    uint32_t input_cb_depth = 2;

    // Pages from the input tensor
    desc.cbs.push_back(
        make_cb(CBIndex::c_0, input_cb_depth * input_page_size, input_dataformat, input_page_size, all_cores));
    // The entire index tensor
    desc.cbs.push_back(make_cb(CBIndex::c_1, index_total_size, index_dataformat, index_total_size, all_cores));
    // An input page filled with fill_value
    desc.cbs.push_back(make_cb(CBIndex::c_2, input_page_size, input_dataformat, input_page_size, all_cores));

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/index_fill/device/kernels/";

    // Create reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_page_size,          // input tensor page size
        (std::uint32_t)index_total_size,         // index tensor total size
        (std::uint32_t)index.physical_volume(),  // num elements in index array
        (std::uint32_t)(dim == n - 1)            // is last dim
    };
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(index.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc{
        .kernel_source = kernels_dir_path + "reader_index_fill.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = std::move(reader_compile_time_args),
        .config = ReaderConfigDescriptor{},
    };

    const auto out_mem_layout = output.memory_config().memory_layout();
    const OutputWriteMode write_mode = output_write_mode(input.memory_config().memory_layout(), out_mem_layout);

    uint32_t out_write_size{};  // bytes written per NOC write to output
    uint32_t out_KW{};          // number of column writes per input row (1 unless splitting into WIDTH/BLOCK)

    if (is_col_sharded(out_mem_layout)) {
        // NOTE: assumes the input shard page is tightly packed (page_size == shard_width *
        // element_size, no extra NOC-alignment padding), so that the WIDTH/BLOCK→INTERLEAVED/HEIGHT
        // byte offset (col_shard_id * input_page_size) and the row-splitting write size line up.
        // This holds for row-major L1 sharded buffers used here; revisit if padded page sizes are introduced.
        out_write_size = output.mesh_buffer().aligned_page_size();
        out_KW = (input.padded_shape()[-1] * input.element_size()) / out_write_size;
    } else {
        // INTERLEAVED or HEIGHT_SHARDED output: write size = full input page
        out_write_size = input_page_size;
        out_KW = 1;
    }

    // Create writer kernel
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)input_page_size,          // CB page size (= input shard row size)
        (std::uint32_t)index.physical_volume(),  // num elements in index array
        (std::uint32_t)input.element_size(),     // element size in bytes
        (std::uint32_t)(dim == n - 1)            // is last dim
    };
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc{
        .kernel_source = kernels_dir_path + "writer_index_fill.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .compile_time_args = std::move(writer_compile_time_args),
        .config = WriterConfigDescriptor{},
    };

    for (const CoreWork& work : split.cores) {
        reader_desc.emplace_runtime_args(
            work.core,
            {
                input.buffer(),        // input tensor address (binding)
                index.buffer(),        // index tensor address (binding)
                work.start_row,        // start row
                work.end_row,          // end row
                num_rows_in_dim,       // num rows in dim
                dim_size,              // dim size
                work.col_shard_id,     // column shard index
                work.row_page_stride,  // page stride across rows
            });

        const OutputWriteArgs out = output_write_args(write_mode, work, out_KW, input_page_size);
        writer_desc.emplace_runtime_args(
            work.core,
            {
                output.buffer(),    // output tensor address (binding)
                work.start_row,     // start row
                work.end_row,       // end row
                num_rows_in_dim,    // num rows in dim
                dim_size,           // dim size
                fill_value,         // fill value
                work.col_shard_id,  // input col shard (for last-dim bounds check)
                // --- output write params ---
                out.col_shard_id,     // output col shard base index
                out.row_page_stride,  // output page stride
                out.col_byte_offset,  // byte offset within output page
                out.num_col_shards,   // number of output writes per row
                out_write_size,       // bytes per output write
            });
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::index_fill
