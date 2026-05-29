// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_parallelize_column_program_factory.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor UntilizeMultiCoreParallelizeColumnProgramFactory::create_descriptor(
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

    IDevice* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t ntiles = a.physical_volume() / TILE_HW;
    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = grid_size.y;

    ncores_x = untilize_helper::get_largest_divisor(ntiles, ncores_x);
    ncores_y = untilize_helper::get_largest_divisor(ntiles, ncores_y, ncores_x);

    TT_ASSERT(ntiles % (ncores_x * ncores_y) == 0);
    uint32_t ntiles_per_block = ntiles / (ncores_x * ncores_y);

    // TODO increase block size to increase untilize performance, currently each untilize block is a single tile
    uint32_t max_tiles = 1;

    uint32_t stick_s = a.padded_shape()[-1];
    uint32_t ntiles_per_row = stick_s / TILE_WIDTH;
    uint32_t stick_size = stick_s * output.element_size();
    uint32_t ntiles_per_column = ntiles / ntiles_per_row;
    uint32_t starting_tile = ntiles_per_block;
    if (ntiles_per_row > max_tiles) {
        starting_tile = max_tiles;
    }
    ntiles_per_block = untilize_helper::get_largest_divisor(ntiles_per_row, starting_tile);
    TT_ASSERT(
        ntiles_per_row % ntiles_per_block == 0 and ntiles_per_block >= 1 and ntiles_per_block <= ntiles_per_row and
        ntiles % ntiles_per_block == 0);

    uint32_t nblocks = (ntiles / ntiles_per_block);

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(CoreCoord(ncores_x, ncores_y), nblocks);

    bool row_major = true;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t output_cb_index = tt::CBIndex::c_16;

    ProgramDescriptor desc;

    uint32_t num_input_tiles = ntiles_per_block * 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    uint32_t num_output_tiles = ntiles_per_block * 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct_args = {stick_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    /** compute
     */
    std::vector<uint32_t> compute_args = {
        nblocks_per_core,  // per_core_block_cnt
        ntiles_per_block,  // per_block_ntiles
        src0_cb_index,
        output_cb_index};
    std::vector<uint32_t> compute_args_cliff = {
        nblocks_per_core_cliff,
        ntiles_per_block,  // per_block_ntiles
        src0_cb_index,
        output_cb_index};

    std::vector<std::pair<std::string, std::string>> compute_kernel_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";

    std::optional<KernelDescriptor> compute_desc;
    if (!core_range.ranges().empty()) {
        KernelDescriptor cd;
        cd.kernel_source = compute_kernel_path;
        cd.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cd.core_ranges = core_range;
        cd.compile_time_args = std::move(compute_args);
        cd.defines = compute_kernel_defines;
        cd.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
        };
        compute_desc = std::move(cd);
    }
    std::optional<KernelDescriptor> compute_cliff_desc;
    if (!core_range_cliff.ranges().empty()) {
        KernelDescriptor cd;
        cd.kernel_source = compute_kernel_path;
        cd.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cd.core_ranges = core_range_cliff;
        cd.compile_time_args = std::move(compute_args_cliff);
        cd.defines = std::move(compute_kernel_defines);
        cd.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        };
        compute_cliff_desc = std::move(cd);
    }

    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        // unequal case with cliff
        ncores_full -= 1;
        full_cores = core_range;
    }
    uint32_t tile_start_id = 0;
    uint32_t offset_within_stick = 0;
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);

    auto nsticks_per_core = ntiles_per_column * TILE_HEIGHT;

    for (const auto& core : cores) {
        if (!full_cores.contains(core)) {
            continue;
        }
        // reader runtime args — Buffer* slot auto-registers as a BufferBinding so the
        // framework patches addresses on cache hits.
        uint32_t ntiles_per_core = ntiles_per_block * nblocks_per_core;
        reader_desc.emplace_runtime_args(
            core,
            {src0_buffer,      // src_addr
             ntiles_per_core,  // ntiles
             tile_start_id});  // start_id

        writer_desc.emplace_runtime_args(
            core,
            {dst_buffer,                                                 // dst_addr
             nsticks_per_core,                                           // nsticks
             ntiles_per_core,                                            // ntiles_per_core
             static_cast<uint32_t>(TILE_WIDTH * output.element_size()),  // tile_width_size
             std::uint32_t{0},                                           // start stick id = 0
             offset_within_stick});

        tile_start_id += ntiles_per_core;
        offset_within_stick += ntiles_per_core * TILE_WIDTH * output.element_size();
    }
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        CoreCoord core = row_major ? CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x}
                                   : CoreCoord{ncores_full / ncores_y, ncores_full % ncores_y};
        uint32_t ntiles_per_core_cliff = ntiles_per_block * nblocks_per_core_cliff;
        reader_desc.emplace_runtime_args(
            core,
            {src0_buffer,            // src_addr
             ntiles_per_core_cliff,  // ntiles
             tile_start_id});        // start_id

        writer_desc.emplace_runtime_args(
            core,
            {dst_buffer,                                                 // dst_addr
             nsticks_per_core,                                           // nsticks
             stick_size,                                                 // block_size_nbytes
             ntiles_per_core_cliff,                                      // ntiles_per_core
             static_cast<uint32_t>(TILE_WIDTH * output.element_size()),  // tile_width_size
             std::uint32_t{0},                                           // start stick id = 0
             offset_within_stick});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    if (compute_desc.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc));
    }
    if (compute_cliff_desc.has_value()) {
        desc.kernels.push_back(std::move(*compute_cliff_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
