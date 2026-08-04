// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_block_program_factory.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Helper: append a paired (input, output) CBDescriptor for a given core range.
void push_cb_pair(
    ProgramDescriptor& desc,
    const CoreRangeSet& core_ranges,
    uint32_t input_single_tile_size,
    uint32_t output_single_tile_size,
    uint32_t num_tiles,
    tt::DataFormat input_cb_data_format,
    tt::DataFormat output_cb_data_format) {
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * input_single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * output_single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });
}

}  // namespace

ProgramDescriptor UntilizeMultiCoreBlockProgramFactory::create_descriptor(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    uint32_t a_tile_width = a.tensor_spec().tile().get_width();
    uint32_t a_tile_height = a.tensor_spec().tile().get_height();

    uint32_t num_tiles_per_row = a.padded_shape()[-1] / a_tile_width;
    uint32_t num_tiles_per_col = a.padded_shape()[-2] / a_tile_height;

    uint32_t num_blocks = (a.padded_shape()[-1] * a.padded_shape()[-2]) / (a_tile_height * a_tile_width);

    uint32_t max_l1_size = operations::data_movement::get_max_l1_space(a);
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
                grid_size, num_blocks, num_tiles_per_row, num_tiles_per_col, cb_block_size_limit);

    if (single_sub_block_size > 0 && single_block_size % single_sub_block_size) {
        TT_FATAL(false, "single_block_size is not divided by single_sub_block_size");
    }

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);
    uint32_t row_size_bytes;

    uint32_t el_size = a.element_size();
    if (a.dtype() == DataType::BFLOAT8_B) {
        row_size_bytes = input_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        row_size_bytes = input_shape[-1] * a.element_size();
    }

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    ProgramDescriptor desc;

    if (!core_range.empty()) {
        push_cb_pair(
            desc,
            core_range,
            input_single_tile_size,
            output_single_tile_size,
            single_sub_block_size,
            input_cb_data_format,
            output_cb_data_format);
    }
    if (has_cliff_col && has_cliff_row) {
        push_cb_pair(
            desc,
            cliff_col_row_core_range,
            input_single_tile_size,
            output_single_tile_size,
            single_block_size_cliff_row,
            input_cb_data_format,
            output_cb_data_format);
    }
    if (has_cliff_row) {
        push_cb_pair(
            desc,
            cliff_row_core_range,
            input_single_tile_size,
            output_single_tile_size,
            single_block_size_cliff_row,
            input_cb_data_format,
            output_cb_data_format);
    }
    if (has_cliff_col) {
        push_cb_pair(
            desc,
            cliff_col_core_range,
            input_single_tile_size,
            output_single_tile_size,
            single_sub_block_size,
            input_cb_data_format,
            output_cb_data_format);
    }

    // reader
    uint32_t num_tiles_2d = a.padded_shape()[-1] * a.padded_shape()[-2] / TILE_HW;

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
    std::vector<uint32_t> writer_ct_args = {total_num_rows, third_dim, TILE_HEIGHT, row_size_bytes};
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

    std::vector<std::pair<std::string, std::string>> compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }

    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp";

    auto make_compute_kernel = [&](const CoreRangeSet& cores, std::vector<uint32_t> compile_args) {
        KernelDescriptor cd;
        cd.kernel_source = compute_kernel_path;
        cd.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cd.core_ranges = cores;
        cd.compile_time_args = std::move(compile_args);
        cd.defines = compute_kernel_defines;
        cd.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
        };
        return cd;
    };

    std::vector<KernelDescriptor> compute_kernels;
    if (!core_range.empty()) {
        compute_kernels.push_back(
            make_compute_kernel(core_range, {single_sub_block_size_wh, single_sub_block_size, third_dim}));
    }
    if (has_cliff_col && has_cliff_row) {
        compute_kernels.push_back(make_compute_kernel(
            cliff_col_row_core_range, {single_block_size_cliff_col, single_block_size_cliff_row, third_dim}));
    }
    if (has_cliff_row) {
        compute_kernels.push_back(
            make_compute_kernel(cliff_row_core_range, {single_block_size, single_block_size_cliff_row, third_dim}));
    }
    if (has_cliff_col) {
        compute_kernels.push_back(make_compute_kernel(
            cliff_col_core_range, {single_sub_block_size_cliff_col_wh, single_sub_block_size, third_dim}));
    }

    // RUNTIME ARGS
    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
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

        // reader runtime args — Buffer* slot auto-registers as a BufferBinding so the
        // framework patches addresses on cache hits.
        reader_desc.emplace_runtime_args(
            core, {src0_buffer, tile_start_id, single_block_size_row_arg, single_block_size_col_arg});

        // writer runtime args
        writer_desc.emplace_runtime_args(
            core,
            {dst_buffer,
             TILE_WIDTH * el_size * single_block_size_row_arg,
             start_row_id,
             start_column_id,
             single_block_size_row_arg,
             single_block_size_col_arg,
             TILE_WIDTH * el_size * single_sub_block_size_row_arg,
             single_sub_block_size_row_arg});

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * TILE_WIDTH * el_size);
        start_column_id = end_column_id % row_size_bytes;
        if (end_column_id % row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * TILE_HEIGHT;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    for (auto& cd : compute_kernels) {
        desc.kernels.push_back(std::move(cd));
    }

    return desc;
}

}  // namespace ttnn::prim
