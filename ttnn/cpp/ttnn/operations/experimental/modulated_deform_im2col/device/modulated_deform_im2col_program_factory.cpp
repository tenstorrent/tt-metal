// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

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
// The widest margin a core's input band takes beyond the kernel's span, and the L1 the band leaves
// free for what the op's program does not account for.
constexpr uint32_t kMaxBandMargin = 4;
constexpr uint32_t kBandL1Reserve = 32 * 1024;

}  // namespace

// Work unit: a block of 32 consecutive output rows (pixels), all K*K kernel
// positions of it on one core, so its offsets and mask logits are staged once. Blocks
// are independent, so there is no cross-core traffic.
ProgramDescriptor ModulatedDeformIm2colOperation::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, Tensor& output) {
    ProgramDescriptor descriptor{};
    const auto s = derive_shapes(attrs, tensor_args);

    const auto& input = tensor_args.input;
    const auto& offset_mask = tensor_args.offset_mask;

    constexpr uint32_t TILE_ROWS = tt::constants::TILE_HEIGHT;
    const uint32_t channel_tiles = s.channels / tt::constants::TILE_WIDTH;
    const uint32_t num_blocks = tt::div_up(s.rows, TILE_ROWS);

    const auto tile_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const uint32_t tile_nbytes = tt::tile_size(tile_format);

    // Page strides as the buffers lay the pages out; validation guarantees no row padding.
    const uint32_t input_page_nbytes = input.buffer()->aligned_page_size();
    const uint32_t offset_mask_page_nbytes = offset_mask.buffer()->aligned_page_size();

    const auto compute_grid = input.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, blocks_per_core_1, blocks_per_core_2] =
        split_work_to_cores(compute_grid, num_blocks);

    constexpr uint8_t offset_mask_arena_cb = tt::CBIndex::c_0;
    constexpr uint8_t corner_block_cb = tt::CBIndex::c_2;
    constexpr uint8_t scalar_tile_cb = tt::CBIndex::c_3;
    constexpr uint8_t tiled_block_cb = tt::CBIndex::c_4;
    constexpr uint8_t geom_in_cb = tt::CBIndex::c_5;
    constexpr uint8_t geom_index_cb = tt::CBIndex::c_6;
    constexpr uint8_t geom_weight_cb = tt::CBIndex::c_7;
    constexpr uint8_t geom_kernel_cb = tt::CBIndex::c_8;
    // The second reader's copies of the per-position buffers, for the odd kernel positions.
    constexpr uint8_t odd_corner_block_cb = tt::CBIndex::c_9;
    constexpr uint8_t odd_scalar_tile_cb = tt::CBIndex::c_10;
    constexpr uint8_t odd_geom_index_cb = tt::CBIndex::c_11;
    constexpr uint8_t odd_geom_weight_cb = tt::CBIndex::c_12;
    // The two halves of a core's input band, each copied in by one reader.
    constexpr uint8_t band_a_cb = tt::CBIndex::c_13;
    constexpr uint8_t band_b_cb = tt::CBIndex::c_14;
    constexpr uint8_t output_tile_cb = tt::CBIndex::c_16;
    constexpr uint8_t odd_output_tile_cb = tt::CBIndex::c_17;
    constexpr uint32_t CORNERS = 4;

    // Blocks per core, in the order the cores take consecutive blocks.
    const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    std::vector<uint32_t> core_blocks;
    for (const auto& core : cores) {
        core_blocks.push_back(
            core_group_1.contains(core) ? blocks_per_core_1 : (core_group_2.contains(core) ? blocks_per_core_2 : 0));
    }

    // A core's blocks are consecutive output rows, which sample a band of input rows: the dilated
    // kernel's rows plus the bottom corners' row, widened by a margin on each side for the offsets.
    // The readers copy the band into L1 once and read the corners inside it from there: corner
    // reads spread over every core's L1 saturate the NoC around the first rows of cores, and those
    // cores then set the op's time. Corners outside the band are read from the input as before.
    // The margin is the widest whose band fits the L1 the op's other buffers leave at the first call;
    // a cached program run later with less L1 free fails its circular-buffer check.
    const uint32_t image_rows = s.out_h * s.out_w;
    const int64_t kernel_rows = static_cast<int64_t>(attrs.dilation[0]) * (attrs.kernel_size[0] - 1) + 1;
    auto band = [&](uint32_t first_block, uint32_t blocks, uint32_t margin) -> std::pair<uint32_t, uint32_t> {
        if (blocks == 0) {
            return {0, 0};
        }
        const uint32_t first_row = first_block * TILE_ROWS;
        const uint32_t last_row = std::min(s.rows, (first_block + blocks) * TILE_ROWS) - 1;
        auto clamp_row = [&](int64_t y) { return std::clamp<int64_t>(y, 0, static_cast<int64_t>(s.in_h) - 1); };
        const int64_t top =
            static_cast<int64_t>((first_row / s.out_w) % s.out_h) * attrs.stride[0] - attrs.padding[0] - margin;
        const int64_t bottom = static_cast<int64_t>((last_row / s.out_w) % s.out_h) * attrs.stride[0] -
                               attrs.padding[0] + kernel_rows + margin;
        const uint64_t image_pages = static_cast<uint64_t>(s.in_h) * s.in_w;
        const uint64_t lo = (first_row / image_rows) * image_pages + clamp_row(top) * s.in_w;
        const uint64_t hi = (last_row / image_rows) * image_pages + (clamp_row(bottom) + 1) * s.in_w;
        return {static_cast<uint32_t>(lo), static_cast<uint32_t>(hi - lo)};
    };

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
    push_cb(offset_mask_arena_cb, TILE_ROWS, offset_mask_page_nbytes);
    // The four corners of a kernel position, row-major in tile-sized pages, and one scalar tile
    // per corner, for each reader, two positions deep: a reader lands its next position while
    // compute still holds its last one.
    for (const uint8_t cb : {corner_block_cb, odd_corner_block_cb}) {
        push_cb(cb, 2 * CORNERS * channel_tiles, tile_nbytes);
    }
    for (const uint8_t cb : {scalar_tile_cb, odd_scalar_tile_cb}) {
        push_cb(cb, 2 * CORNERS, tile_nbytes);
    }
    push_cb(tiled_block_cb, CORNERS * channel_tiles, tile_nbytes);
    // Each reader writes the output of its own kernel positions.
    for (const uint8_t cb : {output_tile_cb, odd_output_tile_cb}) {
        push_cb(cb, 2 * channel_tiles, tile_nbytes);
    }
    // A block's geometry, [row, kernel position] tiles: the first reader's offsets, mask logits and window
    // origins in fp32, and compute's corner pages (int32) and weights, one copy per reader and two
    // blocks deep so compute hands over the next block's while the readers work on this one. The
    // kernel offsets are one pair of fp32 tiles for the whole run.
    constexpr uint32_t GEOM_IN_TILES = 5;
    const uint32_t fp32_tile_nbytes = tt::tile_size(tt::DataFormat::Float32);
    push_cb(geom_in_cb, GEOM_IN_TILES, fp32_tile_nbytes, tt::DataFormat::Float32);
    push_cb(geom_kernel_cb, 2, fp32_tile_nbytes, tt::DataFormat::Float32);
    for (const uint8_t cb : {geom_index_cb, odd_geom_index_cb}) {
        push_cb(cb, 2, tt::tile_size(tt::DataFormat::Int32), tt::DataFormat::Int32);
    }
    for (const uint8_t cb : {geom_weight_cb, odd_geom_weight_cb}) {
        push_cb(cb, 2 * CORNERS, tile_nbytes);
    }
    // L1 below the lowest buffer, less the other circular buffers and a reserve.
    const auto* device = input.device();
    const uint64_t l1_top = device->lowest_occupied_compute_l1_address().value_or(device->l1_size_per_core());
    const uint64_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint64_t other_cbs = 0;
    for (const auto& cb : descriptor.cbs) {
        other_cbs += cb.total_size;
    }
    const uint64_t band_budget =
        l1_top > l1_base + other_cbs + kBandL1Reserve ? l1_top - l1_base - other_cbs - kBandL1Reserve : 0;
    uint32_t band_margin = 0;
    uint32_t band_pages = 0;
    bool use_band = false;
    for (uint32_t margin = 0; margin <= kMaxBandMargin; ++margin) {
        uint32_t pages = 0;
        uint32_t cursor = 0;
        for (const uint32_t blocks : core_blocks) {
            pages = std::max(pages, band(cursor, blocks, margin).second);
            cursor += blocks;
        }
        if (static_cast<uint64_t>(pages) * input_page_nbytes > band_budget) {
            break;
        }
        use_band = true;
        band_margin = margin;
        band_pages = pages;
    }
    const uint32_t band_half_pages = std::max<uint32_t>(1, (band_pages + 1) / 2);
    for (const uint8_t cb : {band_a_cb, band_b_cb}) {
        push_cb(cb, 1, band_half_pages * input_page_nbytes);
    }

    // Both readers run the same kernel and write the output of their own kernel positions: the
    // first, on its usual RISC-V and NoC, takes the even ones; the second, on the other RISC-V and
    // NoC, the odd ones, one fewer, and stages the geometry too.
    auto reader_ct = [&](bool first) {
        KernelDescriptor::CompileTimeArgs ct{
            offset_mask_arena_cb,
            first ? corner_block_cb : odd_corner_block_cb,
            first ? scalar_tile_cb : odd_scalar_tile_cb,
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
            offset_mask_page_nbytes,
            kOffsetClampPixels,
            geom_in_cb,
            first ? geom_index_cb : odd_geom_index_cb,
            first ? geom_weight_cb : odd_geom_weight_cb,
            geom_kernel_cb,
            first ? 0u : 1u,  // stages the geometry
            first ? 0u : 1u,  // first kernel position
            first ? output_tile_cb : odd_output_tile_cb,
            tile_nbytes,
            band_a_cb,
            band_b_cb,
        };
        TensorAccessorArgs(*input.buffer()).append_to(ct);
        TensorAccessorArgs(*offset_mask.buffer()).append_to(ct);
        TensorAccessorArgs(*output.buffer()).append_to(ct);
        return ct;
    };

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kKernelDir) + "dataflow/reader_deform_im2col.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_ct(true);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor odd_reader_desc;
    odd_reader_desc.kernel_source = reader_desc.kernel_source;
    odd_reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    odd_reader_desc.core_ranges = all_cores;
    odd_reader_desc.compile_time_args = reader_ct(false);
    odd_reader_desc.config = WriterConfigDescriptor{};

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
        geom_weight_cb,
        geom_kernel_cb,
        std::bit_cast<uint32_t>(static_cast<float>(s.in_h)),
        std::bit_cast<uint32_t>(static_cast<float>(s.in_h) - 1.0f),
        std::bit_cast<uint32_t>(static_cast<float>(s.in_w)),
        std::bit_cast<uint32_t>(static_cast<float>(s.in_w) - 1.0f),
        odd_corner_block_cb,
        odd_scalar_tile_cb,
        odd_geom_index_cb,
        odd_geom_weight_cb,
        odd_output_tile_cb};
    // The geometry runs in fp32 and keeps eight tiles in dest at once, which takes full-sync dest;
    // its fp32 inputs unpack straight to dest, as srcA would round them to TF32.
    ComputeConfigDescriptor compute_config{};
    compute_config.fp32_dest_acc_en = true;
    compute_config.dst_full_sync_en = true;
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    compute_config.unpack_to_dest_mode[geom_in_cb] = UnpackToDestMode::UnpackToDestFp32;
    compute_config.unpack_to_dest_mode[geom_kernel_cb] = UnpackToDestMode::UnpackToDestFp32;
    compute_desc.config = compute_config;

    // Buffer* entries register as buffer bindings, so cache hits patch the addresses.
    uint32_t block_cursor = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        const uint32_t blocks_here = core_blocks[i];
        if (blocks_here == 0) {
            continue;
        }
        const auto [band_first_page, band_core_pages] =
            use_band ? band(block_cursor, blocks_here, band_margin) : std::pair<uint32_t, uint32_t>{0, 0};
        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(input.buffer());
        reader_args.push_back(offset_mask.buffer());
        reader_args.push_back(block_cursor);
        reader_args.push_back(blocks_here);
        reader_args.push_back(s.rows);
        reader_args.push_back(output.buffer());
        reader_args.push_back(band_first_page);
        reader_args.push_back(band_core_pages);
        reader_desc.emplace_runtime_args(core, reader_args);
        odd_reader_desc.emplace_runtime_args(core, reader_args);

        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{blocks_here});

        block_cursor += blocks_here;
    }

    descriptor.kernels.push_back(std::move(reader_desc));
    descriptor.kernels.push_back(std::move(compute_desc));
    descriptor.kernels.push_back(std::move(odd_reader_desc));
    return descriptor;
}

}  // namespace ttnn::operations::experimental::modulated_deform_im2col
