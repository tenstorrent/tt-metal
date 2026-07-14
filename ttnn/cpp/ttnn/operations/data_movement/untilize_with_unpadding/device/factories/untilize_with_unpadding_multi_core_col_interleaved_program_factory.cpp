// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_col_interleaved_program_factory.hpp"

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

tt::tt_metal::ProgramDescriptor UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory::create_descriptor(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    ProgramDescriptor desc;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t num_blocks = input_shape[-1] / TILE_WIDTH;
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / TILE_WIDTH;
    uint32_t num_tiles_per_col = a.padded_shape()[-2] / TILE_HEIGHT;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.empty();

    uint32_t unpadded_row_size_bytes;

    uint32_t el_size;
    if (a.dtype() == DataType::BFLOAT8_B) {
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
        el_size = a.element_size();
    }

    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_per_col * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });
    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_per_col * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // reader
    uint32_t num_tiles_2d = a.padded_shape()[-1] * a.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    std::vector<uint32_t> reader_compile_time_args = {num_tiles_2d, third_dim, nblocks_per_core};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_col_multicore.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // writer
    uint32_t total_num_rows = output.logical_shape()[-2];

    std::vector<uint32_t> writer_ct_args = {total_num_rows, ncores, third_dim, TILE_WIDTH, unpadded_row_size_bytes};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_stick_layout_col_multicore.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // compute
    const std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_w.cpp");

    if (!core_range.empty()) {
        KernelDescriptor compute_desc;
        compute_desc.kernel_source = compute_kernel;
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = core_range;
        compute_desc.compile_time_args = {nblocks_per_core, num_tiles_per_col, third_dim};
        compute_desc.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en};
        desc.kernels.push_back(std::move(compute_desc));
    }
    if (has_cliff) {
        KernelDescriptor cliff_desc;
        cliff_desc.kernel_source = compute_kernel;
        cliff_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cliff_desc.core_ranges = core_range_cliff;
        cliff_desc.compile_time_args = {nblocks_per_core_cliff, num_tiles_per_col, third_dim};
        cliff_desc.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en};
        desc.kernels.push_back(std::move(cliff_desc));
    }

    // RUNTIME ARGS
    const auto& cores = corerange_to_cores(available_grid);
    reader_desc.runtime_args.reserve(ncores);
    writer_desc.runtime_args.reserve(ncores);
    uint32_t number_blocks_per_core;
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];

        if (has_cliff && i == ncores - 1) {
            number_blocks_per_core = nblocks_per_core_cliff;
        } else {
            number_blocks_per_core = nblocks_per_core;
        }
        uint32_t size_per_row_per_block = nblocks_per_core * TILE_WIDTH * el_size;

        //  writer runtime args
        writer_desc.emplace_runtime_args(
            core, {dst_buffer, i, size_per_row_per_block, number_blocks_per_core, TILE_WIDTH * el_size});

        // reader runtime args
        reader_desc.emplace_runtime_args(core, {src0_buffer, i, num_tiles_per_row, number_blocks_per_core});
    }

    // Insert reader+writer at the start so kernel ordering matches the legacy program: reader is
    // descriptor 0, writer is descriptor 1, compute kernels follow.
    desc.kernels.insert(desc.kernels.begin(), std::move(writer_desc));
    desc.kernels.insert(desc.kernels.begin(), std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::prim
