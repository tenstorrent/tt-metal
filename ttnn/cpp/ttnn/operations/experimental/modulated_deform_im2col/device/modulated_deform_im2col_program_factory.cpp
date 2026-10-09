// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "hostdevcommon/kernel_structs.h"

#include "modulated_deform_im2col_device_operation.hpp"

namespace ttnn::operations::experimental::modulated_deform_im2col {

using namespace tt::tt_metal;

namespace {

constexpr std::string_view kKernelDir = "ttnn/cpp/ttnn/operations/experimental/modulated_deform_im2col/device/kernels/";

}  // namespace

// Work unit: a block of 32 consecutive output rows (pixels), all K*K kernel
// positions of it on one core, so its offsets and mask are staged once. Blocks
// are independent, so there is no cross-core traffic.
ProgramDescriptor ModulatedDeformIm2colOperation::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, Tensor& output) {
    ProgramDescriptor descriptor{};
    const auto s = derive_shapes(attrs, tensor_args);

    const auto& input = tensor_args.input;
    const auto& offset = tensor_args.offset;
    const auto& mask = tensor_args.mask;

    constexpr uint32_t TILE_ROWS = tt::constants::TILE_HEIGHT;
    const uint32_t channel_tiles = s.channels / tt::constants::TILE_WIDTH;
    const uint32_t num_blocks = tt::div_up(s.rows, TILE_ROWS);

    const auto tile_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const uint32_t tile_nbytes = tt::tile_size(tile_format);

    // Page strides as the buffers lay the pages out; validation guarantees no row padding.
    const uint32_t input_page_nbytes = input.buffer()->aligned_page_size();
    const uint32_t offset_page_nbytes = offset.buffer()->aligned_page_size();
    const uint32_t mask_page_nbytes = mask.buffer()->aligned_page_size();

    const auto compute_grid = input.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, blocks_per_core_1, blocks_per_core_2] =
        split_work_to_cores(compute_grid, num_blocks);

    constexpr uint8_t offset_arena_cb = tt::CBIndex::c_0;
    constexpr uint8_t mask_arena_cb = tt::CBIndex::c_1;
    constexpr uint8_t corner_block_cb = tt::CBIndex::c_2;
    constexpr uint8_t scalar_tile_cb = tt::CBIndex::c_3;
    constexpr uint8_t tiled_block_cb = tt::CBIndex::c_4;
    constexpr uint8_t geom_in_cb = tt::CBIndex::c_5;
    constexpr uint8_t geom_index_cb = tt::CBIndex::c_6;
    constexpr uint8_t geom_weight_cb = tt::CBIndex::c_7;
    constexpr uint8_t output_tile_cb = tt::CBIndex::c_16;
    constexpr uint32_t CORNERS = 4;

    auto push_cb =
        [&](uint8_t idx, uint32_t pages, uint32_t page_size, tt::DataFormat format = tt::DataFormat::Invalid) {
            descriptor.cbs.push_back(CBDescriptor{
                .total_size = pages * page_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = idx,
                    .data_format = format == tt::DataFormat::Invalid ? tile_format : format,
                    .page_size = page_size,
                }}},
            });
        };
    // Reader-only arenas, one stick per row of a block.
    push_cb(offset_arena_cb, TILE_ROWS, offset_page_nbytes);
    push_cb(mask_arena_cb, TILE_ROWS, mask_page_nbytes);
    // The four corners of a kernel position, row-major in tile-sized pages, double-buffered so
    // the reader lands the next position while compute tilizes this one.
    push_cb(corner_block_cb, 2 * CORNERS * channel_tiles, tile_nbytes);
    push_cb(tiled_block_cb, CORNERS * channel_tiles, tile_nbytes);
    // One scalar tile per corner, two positions deep.
    push_cb(scalar_tile_cb, 2 * CORNERS, tile_nbytes);
    push_cb(output_tile_cb, 2 * channel_tiles, tile_nbytes);
    // A block's geometry, [row, kernel position] tiles: the reader's offsets, mask and base
    // coordinates in fp32, and compute's corner indices (int32) and weights, two blocks deep so
    // compute hands over the next block's while the reader works on this one.
    constexpr uint32_t GEOM_IN_TILES = 5;
    push_cb(geom_in_cb, GEOM_IN_TILES, tt::tile_size(tt::DataFormat::Float32), tt::DataFormat::Float32);
    push_cb(geom_index_cb, 2 * 2, tt::tile_size(tt::DataFormat::Int32), tt::DataFormat::Int32);
    push_cb(geom_weight_cb, 2 * CORNERS, tile_nbytes);

    KernelDescriptor::CompileTimeArgs reader_ct{
        offset_arena_cb,
        mask_arena_cb,
        corner_block_cb,
        scalar_tile_cb,
        s.channels,
        s.in_h,
        s.in_w,
        s.out_h,
        s.out_w,
        attrs.kernel_size[1],
        s.kernel_positions,
        attrs.stride[0],
        attrs.stride[1],
        attrs.padding[0],
        attrs.padding[1],
        attrs.dilation[0],
        attrs.dilation[1],
        input_page_nbytes,
        offset_page_nbytes,
        mask_page_nbytes,
        kOffsetClampPixels,
        geom_in_cb,
        geom_index_cb,
        geom_weight_cb,
    };
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*offset.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*mask.buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kKernelDir) + "dataflow/reader_deform_im2col.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(kKernelDir) + "compute/compute_deform_im2col.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {
        corner_block_cb,
        scalar_tile_cb,
        output_tile_cb,
        channel_tiles,
        s.kernel_positions,
        tiled_block_cb,
        geom_in_cb,
        geom_index_cb,
        geom_weight_cb};
    // The geometry runs in fp32 and keeps eight tiles in dest at once, which takes full-sync dest;
    // its fp32 inputs unpack straight to dest, as srcA would round them to TF32.
    ComputeConfigDescriptor compute_config{};
    compute_config.fp32_dest_acc_en = true;
    compute_config.dst_full_sync_en = true;
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    compute_config.unpack_to_dest_mode[geom_in_cb] = UnpackToDestMode::UnpackToDestFp32;
    compute_desc.config = compute_config;

    KernelDescriptor::CompileTimeArgs writer_ct{output_tile_cb, channel_tiles, s.kernel_positions, tile_nbytes};
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(kKernelDir) + "dataflow/writer_deform_im2col.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    // Buffer* entries register as buffer bindings, so cache hits patch the addresses.
    const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    uint32_t block_cursor = 0;
    for (const auto& core : cores) {
        uint32_t blocks_here = 0;
        if (core_group_1.contains(core)) {
            blocks_here = blocks_per_core_1;
        } else if (core_group_2.contains(core)) {
            blocks_here = blocks_per_core_2;
        } else {
            continue;
        }
        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(input.buffer());
        reader_args.push_back(offset.buffer());
        reader_args.push_back(mask.buffer());
        reader_args.push_back(block_cursor);
        reader_args.push_back(blocks_here);
        reader_args.push_back(s.rows);
        reader_desc.emplace_runtime_args(core, reader_args);

        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{blocks_here});

        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(output.buffer());
        writer_args.push_back(block_cursor);
        writer_args.push_back(blocks_here);
        writer_desc.emplace_runtime_args(core, writer_args);

        block_cursor += blocks_here;
    }

    descriptor.kernels.push_back(std::move(reader_desc));
    descriptor.kernels.push_back(std::move(compute_desc));
    descriptor.kernels.push_back(std::move(writer_desc));
    return descriptor;
}

}  // namespace ttnn::operations::experimental::modulated_deform_im2col
