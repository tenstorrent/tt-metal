// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp"

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
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

inline void push_cb_pair(
    ProgramDescriptor& desc,
    const CoreRangeSet& core_ranges,
    uint32_t num_tiles,
    uint32_t input_single_tile_size,
    uint32_t output_single_tile_size,
    tt::DataFormat input_cb_data_format,
    tt::DataFormat output_cb_data_format,
    const Tile& tile) {
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * input_single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = TileDescriptor(tile),
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * output_single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
            .tile = TileDescriptor(tile),
        }}},
    });
}

}  // namespace

tt::tt_metal::ProgramDescriptor UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory::create_descriptor(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    ProgramDescriptor desc;

    const auto& tile = a.tensor_spec().tile();
    const uint32_t tile_height = tile.get_height();
    const uint32_t tile_width = tile.get_width();
    const uint32_t tile_hw = tile.get_tile_hw();

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tile.get_tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tile.get_tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t max_l1_size = operations::data_movement::get_max_l1_space(a);
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / tile_width;
    uint32_t num_tiles_per_col = a.padded_shape()[-2] / tile_height;

    uint32_t num_blocks = (a.padded_shape()[-1] * a.padded_shape()[-2]) / (tile_height * tile_width);
    uint32_t cb_block_size_limit = max_l1_size / (input_single_tile_size + output_single_tile_size);

    auto
        [ncores,
         all_cores,
         core_range,
         cliff_row_core_range,
         cliff_col_core_range,
         cliff_col_row_core_range,
         nblocks_per_core,
         single_block_size,
         single_block_size_cliff_row,
         single_block_size_cliff_col,
         has_cliff_row,
         has_cliff_col,
         full_cores_per_row,
         full_cores_per_col,
         single_sub_block_size] =
            ttnn::split_blocks_for_tilize_wh(
                available_grid, num_blocks, num_tiles_per_row, num_tiles_per_col, cb_block_size_limit);

    if (single_sub_block_size > 0 && single_block_size % single_sub_block_size) {
        TT_FATAL(false, "single_block_size is not divided by single_sub_block_size");
    }

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);
    uint32_t padded_row_size_bytes;
    uint32_t unpadded_row_size_bytes;

    uint32_t el_size;
    if (a.dtype() == DataType::BFLOAT8_B) {
        padded_row_size_bytes = input_shape[-1] * output.element_size();
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        padded_row_size_bytes = input_shape[-1] * a.element_size();
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
        el_size = a.element_size();
    }

    // CBs: emit a pair (input + output) per non-empty core sub-region. The legacy code
    // created two CreateCircularBuffer calls on different core ranges — descriptors mirror
    // this layout exactly.
    if (!core_range.empty()) {
        push_cb_pair(
            desc,
            core_range,
            single_sub_block_size,
            input_single_tile_size,
            output_single_tile_size,
            input_cb_data_format,
            output_cb_data_format,
            tile);
    }
    if (has_cliff_col && has_cliff_row) {
        push_cb_pair(
            desc,
            cliff_col_row_core_range,
            single_block_size_cliff_row,
            input_single_tile_size,
            output_single_tile_size,
            input_cb_data_format,
            output_cb_data_format,
            tile);
    }
    if (has_cliff_row) {
        push_cb_pair(
            desc,
            cliff_row_core_range,
            single_block_size_cliff_row,
            input_single_tile_size,
            output_single_tile_size,
            input_cb_data_format,
            output_cb_data_format,
            tile);
    }
    if (has_cliff_col) {
        push_cb_pair(
            desc,
            cliff_col_core_range,
            single_sub_block_size,
            input_single_tile_size,
            output_single_tile_size,
            input_cb_data_format,
            output_cb_data_format,
            tile);
    }

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // reader

    uint32_t num_tiles_2d = a.padded_shape()[-1] * a.padded_shape()[-2] / tile_hw;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    std::vector<uint32_t> reader_compile_time_args = {num_tiles_2d, third_dim, total_tiles_per_row};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // writer
    uint32_t total_num_rows = output.logical_shape()[-2];
    std::vector<uint32_t> writer_ct_args = {total_num_rows, third_dim, tile_height, unpadded_row_size_bytes};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_stick_layout_wh_multicore.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // compute
    uint32_t single_sub_block_size_wh = single_block_size * single_block_size / single_sub_block_size;
    uint32_t single_sub_block_size_cliff_col_wh =
        single_block_size_cliff_col * single_block_size / single_sub_block_size;
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

    const std::string compute_kernel_path(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp");

    auto push_compute = [&](const CoreRangeSet& cr, std::initializer_list<uint32_t> compile_args) {
        KernelDescriptor compute_desc;
        compute_desc.kernel_source = compute_kernel_path;
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = cr;
        compute_desc.compile_time_args = std::vector<uint32_t>(compile_args.begin(), compile_args.end());
        compute_desc.defines = compute_kernel_defines;
        compute_desc.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
        };
        desc.kernels.push_back(std::move(compute_desc));
    };

    if (!core_range.empty()) {
        push_compute(core_range, {single_sub_block_size_wh, single_sub_block_size, third_dim});
    }
    if (has_cliff_col && has_cliff_row) {
        push_compute(cliff_col_row_core_range, {single_block_size_cliff_col, single_block_size_cliff_row, third_dim});
    }
    if (has_cliff_row) {
        push_compute(cliff_row_core_range, {single_block_size, single_block_size_cliff_row, third_dim});
    }
    if (has_cliff_col) {
        push_compute(cliff_col_core_range, {single_sub_block_size_cliff_col_wh, single_sub_block_size, third_dim});
    }

    // RUNTIME ARGS
    const auto& cores = corerange_to_cores(available_grid);
    uint32_t start_row_id = 0;
    uint32_t start_column_id = 0;
    uint32_t tile_start_id = 0;
    uint32_t single_block_size_row_arg;
    uint32_t single_block_size_col_arg;
    uint32_t single_sub_block_size_row_arg;

    uint32_t total_row_cores = full_cores_per_row;
    if (has_cliff_row) {
        total_row_cores++;
    }
    uint32_t cores_col_count = 1;

    reader_desc.runtime_args.reserve(ncores);
    writer_desc.runtime_args.reserve(ncores);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];

        if (has_cliff_col && has_cliff_row && i == ncores - 1) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size_cliff_col;
            single_sub_block_size_row_arg = single_block_size_cliff_row;

        } else if (has_cliff_row && i != 0 && ((i + 1) % (full_cores_per_row + 1)) == 0) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size;
            single_sub_block_size_row_arg = single_block_size_cliff_row;

        } else if (i < total_row_cores * full_cores_per_col) {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size;
            single_sub_block_size_row_arg = single_sub_block_size;

        } else {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size_cliff_col;
            single_sub_block_size_row_arg = single_sub_block_size;
        }

        // reader runtime args
        reader_desc.emplace_runtime_args(
            core, {src0_buffer, tile_start_id, single_block_size_row_arg, single_block_size_col_arg});

        //  writer runtime args
        writer_desc.emplace_runtime_args(
            core,
            {dst_buffer,
             tile_width * el_size * single_block_size_row_arg,
             start_row_id,
             start_column_id,
             single_block_size_row_arg,
             single_block_size_col_arg,
             tile_width * el_size * single_sub_block_size_row_arg,
             single_sub_block_size_row_arg});

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * tile_width * el_size);
        start_column_id = end_column_id % padded_row_size_bytes;
        if (end_column_id % padded_row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * tile_height;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    // Insert reader+writer at the beginning so they occupy descriptor positions 0 and 1
    // (compute kernels follow).
    desc.kernels.insert(desc.kernels.begin(), std::move(writer_desc));
    desc.kernels.insert(desc.kernels.begin(), std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::prim
