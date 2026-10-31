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

#include "modulated_deform_conv2d_device_operation.hpp"

namespace ttnn::operations::experimental::modulated_deform_conv2d {

using namespace tt::tt_metal;

// The whole conv in one program: each block's sampled columns of one chunk of channels at one kernel
// position are multiplied by that pass's chunk x C_out weight slice in the same core, so the columns
// never leave L1. Kernel positions run outermost so a core loads each weight slice once; an input
// wider than kConvMaxChunkTiles tiles takes several passes per position, one per chunk of channels.
// The work unit is a block of 32 consecutive output rows, which shares one set of offset and mask
// loads; blocks are split evenly over the cores, or by shard for a sharded input, and each core keeps
// its blocks' geometry and output accumulators for the whole run.
ProgramDescriptor ModulatedDeformConv2dOperation::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, Tensor& output) {
    ProgramDescriptor descriptor{};
    const auto s = derive_shapes(attrs, tensor_args);

    const auto& input = tensor_args.input;
    const auto& offset_mask = tensor_args.offset_mask;
    const auto& weight = tensor_args.weight;
    const bool has_bias = tensor_args.bias.has_value();

    constexpr uint32_t TILE_ROWS = tt::constants::TILE_HEIGHT;
    constexpr uint32_t CORNERS = 4;
    constexpr uint32_t GEOM_IN_TILES = 5;
    const uint32_t channel_tiles = s.channels / tt::constants::TILE_WIDTH;
    // The widest chunk up to kConvMaxChunkTiles that splits the channels into whole chunks.
    uint32_t chunk_tiles = std::min(channel_tiles, kConvMaxChunkTiles);
    while (channel_tiles % chunk_tiles != 0) {
        --chunk_tiles;
    }
    const uint32_t chunks = channel_tiles / chunk_tiles;
    const uint32_t out_tiles = s.out_channels / tt::constants::TILE_WIDTH;
    const uint32_t num_blocks = tt::div_up(s.rows, TILE_ROWS);

    const auto tile_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const uint32_t tile_nbytes = tt::tile_size(tile_format);
    const uint32_t fp32_tile_nbytes = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t int32_tile_nbytes = tt::tile_size(tt::DataFormat::Int32);

    const uint32_t input_page_nbytes = input.buffer()->aligned_page_size();
    const uint32_t offset_mask_page_nbytes = offset_mask.buffer()->aligned_page_size();

    // Each core's blocks, in order. A height-sharded input fixes them: a core takes the blocks of its
    // own shard, so most corners lie in its own L1 or a neighbour's, and a core whose shard is empty
    // still joins the weight multicast. Otherwise the blocks spread evenly over the grid.
    const bool sharded_input = input.memory_config().is_sharded();
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> core_blocks;
    if (sharded_input) {
        const auto& spec = *input.shard_spec();
        all_cores = spec.grid;
        cores = corerange_to_cores(
            all_cores, all_cores.num_cores(), spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        const uint32_t shard_blocks = spec.shape[0] / TILE_ROWS;
        for (uint32_t i = 0; i < cores.size(); ++i) {
            const uint32_t first = std::min(num_blocks, i * shard_blocks);
            core_blocks.push_back(std::min(num_blocks, first + shard_blocks) - first);
        }
    } else {
        const auto compute_grid = input.device()->compute_with_storage_grid_size();
        auto [num_cores, grid_cores, core_group_1, core_group_2, blocks_per_core_1, blocks_per_core_2] =
            split_work_to_cores(compute_grid, num_blocks);
        all_cores = grid_cores;
        cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
        for (const auto& core : cores) {
            core_blocks.push_back(
                core_group_1.contains(core) ? blocks_per_core_1
                                            : (core_group_2.contains(core) ? blocks_per_core_2 : 0));
        }
    }
    const uint32_t num_cores = cores.size();
    const uint32_t max_blocks = *std::max_element(core_blocks.begin(), core_blocks.end());

    constexpr uint8_t offset_mask_arena_cb = tt::CBIndex::c_0;
    constexpr uint8_t corner_block_cb = tt::CBIndex::c_2;
    constexpr uint8_t scalar_tile_cb = tt::CBIndex::c_3;
    constexpr uint8_t tiled_block_cb = tt::CBIndex::c_4;
    constexpr uint8_t geom_in_cb = tt::CBIndex::c_5;
    constexpr uint8_t geom_index_cb = tt::CBIndex::c_6;
    constexpr uint8_t geom_weight_cb = tt::CBIndex::c_7;
    constexpr uint8_t geom_kernel_cb = tt::CBIndex::c_8;
    // The second reader's buffers, for the odd blocks.
    constexpr uint8_t odd_corner_block_cb = tt::CBIndex::c_9;
    constexpr uint8_t odd_scalar_tile_cb = tt::CBIndex::c_10;
    constexpr uint8_t odd_geom_index_cb = tt::CBIndex::c_11;
    constexpr uint8_t odd_geom_weight_cb = tt::CBIndex::c_12;
    constexpr uint8_t band_a_cb = tt::CBIndex::c_13;
    constexpr uint8_t band_b_cb = tt::CBIndex::c_14;
    constexpr uint8_t conv_weight_cb = tt::CBIndex::c_15;
    constexpr uint8_t output_tile_cb = tt::CBIndex::c_16;
    constexpr uint8_t odd_output_tile_cb = tt::CBIndex::c_17;
    constexpr uint8_t column_cb = tt::CBIndex::c_18;
    constexpr uint8_t acc_cb = tt::CBIndex::c_19;
    constexpr uint8_t bias_cb = tt::CBIndex::c_20;
    // The second reader stages the odd blocks' geometry inputs.
    constexpr uint8_t odd_offset_mask_arena_cb = tt::CBIndex::c_1;
    constexpr uint8_t odd_geom_in_cb = tt::CBIndex::c_21;
    constexpr uint8_t geom_scratch_cb = tt::CBIndex::c_22;

    // A core's blocks are consecutive output rows, which sample a band of input rows: the dilated
    // kernel's rows plus the bottom corners' row, widened by a margin on each side for the offsets.
    // The readers copy the band into L1 once and read the corners inside it from there: corner
    // reads spread over every core's L1 saturate the NoC around the first rows of cores, and those
    // cores then set the op's time. Corners outside the band are read from the input.
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
    // Each reader's staging arena, half a block's offset-mask rows.
    for (const uint8_t cb : {offset_mask_arena_cb, odd_offset_mask_arena_cb}) {
        push_cb(cb, TILE_ROWS / 2, offset_mask_page_nbytes);
    }
    // The four corners of a (pass, block) and one scalar tile per corner, per reader, two deep.
    for (const uint8_t cb : {corner_block_cb, odd_corner_block_cb}) {
        push_cb(cb, 2 * CORNERS * chunk_tiles, tile_nbytes);
    }
    for (const uint8_t cb : {scalar_tile_cb, odd_scalar_tile_cb}) {
        push_cb(cb, 2 * CORNERS, tile_nbytes);
    }
    push_cb(tiled_block_cb, CORNERS * chunk_tiles, tile_nbytes);
    // Each reader's geometry inputs, one block deep, and its blocks' geometry for the whole run.
    for (const uint8_t cb : {geom_in_cb, odd_geom_in_cb}) {
        push_cb(cb, GEOM_IN_TILES, fp32_tile_nbytes, tt::DataFormat::Float32);
    }
    push_cb(geom_kernel_cb, 2, fp32_tile_nbytes, tt::DataFormat::Float32);
    // The geometry's fp32 intermediates between its rounds in dest: a block pushes five, so each
    // block starts at the front and its four last stay contiguous.
    push_cb(geom_scratch_cb, 5, fp32_tile_nbytes, tt::DataFormat::Float32);
    const uint32_t even_slots = std::max<uint32_t>(1, (max_blocks + 1) / 2);
    const uint32_t odd_slots = std::max<uint32_t>(1, max_blocks / 2);
    push_cb(geom_index_cb, even_slots, int32_tile_nbytes, tt::DataFormat::Int32);
    push_cb(odd_geom_index_cb, odd_slots, int32_tile_nbytes, tt::DataFormat::Int32);
    push_cb(geom_weight_cb, CORNERS * even_slots, tile_nbytes);
    push_cb(odd_geom_weight_cb, CORNERS * odd_slots, tile_nbytes);
    // A block's columns of one pass; two weight slices, the next loading while compute works
    // through the current one; every block's output, accumulated across passes in bfloat16 (fp32
    // within a pass, in dest); the bias.
    push_cb(column_cb, chunk_tiles, tile_nbytes);
    push_cb(conv_weight_cb, 2 * chunk_tiles * out_tiles, tile_nbytes);
    push_cb(acc_cb, max_blocks * out_tiles, tile_nbytes);
    push_cb(bias_cb, out_tiles, tile_nbytes);

    const auto* device = input.device();
    const uint64_t l1_budget = attrs.l1_budget;
    uint64_t other_cbs = 0;
    for (const auto& cb : descriptor.cbs) {
        other_cbs += cb.total_size;
    }
    // Each reader's output, two blocks deep where L1 allows; it only drains the epilogue.
    const uint32_t output_depth = other_cbs + 4 * out_tiles * tile_nbytes <= l1_budget ? 2 : 1;
    for (const uint8_t cb : {output_tile_cb, odd_output_tile_cb}) {
        push_cb(cb, output_depth * out_tiles, tile_nbytes);
        other_cbs += output_depth * out_tiles * tile_nbytes;
    }
    // The band is optional: without room for it the readers fetch the corners from the input where it
    // is. A sharded input needs none, its corners being in the core's own shard or a neighbour's.
    // Its two halves take one page each even when unused, and a band of pages takes both halves
    // rounded up to whole pages.
    auto band_cbs_nbytes = [&](uint32_t pages) -> uint64_t {
        return 2ull * std::max<uint32_t>(1, (pages + 1) / 2) * input_page_nbytes;
    };
    const uint64_t band_budget = !sharded_input && l1_budget > other_cbs ? l1_budget - other_cbs : 0;
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
        if (band_cbs_nbytes(pages) > band_budget) {
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
    other_cbs += band_cbs_nbytes(band_pages);
    TT_FATAL(
        other_cbs <= l1_budget,
        "modulated_deform_conv2d: needs {} KB of L1 per core, {} KB free: {} blocks of 32 rows per core, C {}, C_out "
        "{} (output accumulators {} KB, weight slices {} KB)",
        other_cbs / 1024,
        l1_budget / 1024,
        max_blocks,
        s.channels,
        s.out_channels,
        max_blocks * out_tiles * tile_nbytes / 1024,
        2 * chunk_tiles * out_tiles * tile_nbytes / 1024);

    // One core reads the weight slices and multicasts them when the cores form one rectangle; the
    // multicast fills a rectangle, and a ragged set would leave it writing into idle cores' L1.
    const bool mcast_weight = all_cores.ranges().size() == 1 && num_cores > 1;
    const CoreRange core_rect = *all_cores.ranges().begin();
    const CoreCoord mcast_start = device->worker_core_from_logical_core(core_rect.start_coord);
    const CoreCoord mcast_end = device->worker_core_from_logical_core(core_rect.end_coord);
    constexpr uint32_t weight_sender_sem_id = 0;
    constexpr uint32_t weight_receiver_sem_id = 1;
    for (const uint32_t id : {weight_sender_sem_id, weight_receiver_sem_id}) {
        descriptor.semaphores.push_back(SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }

    // The weight slices go to the reader with fewer blocks on the busiest cores: with an odd count
    // the even reader has one more block of corners to gather.
    const uint32_t weight_parity = max_blocks % 2;
    // The copy of a block parity: 0 takes the even blocks, 1 the odd ones.
    auto reader_named_ct = [&](uint32_t parity) {
        const bool even = parity == 0;
        return KernelDescriptor::NamedCompileTimeArgs{
            {"offset_mask_arena_cb", even ? offset_mask_arena_cb : odd_offset_mask_arena_cb},
            {"corner_block_cb", even ? corner_block_cb : odd_corner_block_cb},
            {"scalar_tile_cb", even ? scalar_tile_cb : odd_scalar_tile_cb},
            {"geom_in_cb", even ? geom_in_cb : odd_geom_in_cb},
            {"geom_index_cb", even ? geom_index_cb : odd_geom_index_cb},
            {"geom_weight_cb", even ? geom_weight_cb : odd_geom_weight_cb},
            {"output_cb", even ? output_tile_cb : odd_output_tile_cb},
            {"geom_kernel_cb", geom_kernel_cb},
            {"band_a_cb", band_a_cb},
            {"band_b_cb", band_b_cb},
            {"conv_weight_cb", conv_weight_cb},
            {"bias_cb", bias_cb},
            {"block_parity", parity},
            {"weight_parity", weight_parity},
            {"channels", s.channels},
            {"in_h", s.in_h},
            {"in_w", s.in_w},
            {"out_h", s.out_h},
            {"out_w", s.out_w},
            {"kernel_w", attrs.kernel_size[1]},
            {"kernel_positions", s.kernel_positions},
            {"stride_h", attrs.stride[0]},
            {"stride_w", attrs.stride[1]},
            {"pad_h", attrs.padding[0]},
            {"pad_w", attrs.padding[1]},
            {"dilation_h", attrs.dilation[0]},
            {"dilation_w", attrs.dilation[1]},
            {"input_page_nbytes", input_page_nbytes},
            {"offset_mask_page_nbytes", offset_mask_page_nbytes},
            {"offset_clamp", kOffsetClampPixels},
            {"tile_nbytes", tile_nbytes},
            {"out_tiles", out_tiles},
            {"chunk_tiles", chunk_tiles},
            {"has_bias", has_bias ? 1u : 0u},
            {"mcast_weight", mcast_weight ? 1u : 0u},
            {"mcast_num_dests", num_cores - 1},
            {"weight_sender_sem_id", weight_sender_sem_id},
            {"weight_receiver_sem_id", weight_receiver_sem_id},
            {"input_shard_pages", sharded_input ? input.shard_spec()->shape[0] : 0u},
        };
    };
    // Only the tensor accessors' args are positional, chained from index 0.
    auto reader_ct = [&]() {
        KernelDescriptor::CompileTimeArgs ct;
        TensorAccessorArgs(*input.buffer()).append_to(ct);
        TensorAccessorArgs(*offset_mask.buffer()).append_to(ct);
        TensorAccessorArgs(*output.buffer()).append_to(ct);
        TensorAccessorArgs(*weight.buffer()).append_to(ct);
        // Without a bias the weight stands in, so the accessor's args exist; the kernel never reads it.
        TensorAccessorArgs(has_bias ? *tensor_args.bias->buffer() : *weight.buffer()).append_to(ct);
        return ct;
    };

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kKernelDir) + "dataflow/reader_deform_conv.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_ct();
    reader_desc.named_compile_time_args = reader_named_ct(0);
    if (sharded_input) {
        // Both readers issue on either NoC, so each corner can take the NoC that suits its shard.
        reader_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0, .noc_mode = NOC_MODE::DM_DYNAMIC_NOC};
    } else {
        reader_desc.config = ReaderConfigDescriptor{};
    }

    KernelDescriptor odd_reader_desc;
    odd_reader_desc.kernel_source = reader_desc.kernel_source;
    odd_reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    odd_reader_desc.core_ranges = all_cores;
    odd_reader_desc.compile_time_args = reader_ct();
    odd_reader_desc.named_compile_time_args = reader_named_ct(1);
    // The second copy runs on the other data-movement RISC-V and NoC.
    if (sharded_input) {
        odd_reader_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1, .noc_mode = NOC_MODE::DM_DYNAMIC_NOC};
    } else {
        odd_reader_desc.config = WriterConfigDescriptor{};
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(kKernelDir) + "compute/compute_deform_conv.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.named_compile_time_args = {
        {"corner_block_cb", corner_block_cb},
        {"scalar_tile_cb", scalar_tile_cb},
        {"geom_in_cb", geom_in_cb},
        {"geom_index_cb", geom_index_cb},
        {"geom_weight_cb", geom_weight_cb},
        {"output_cb", output_tile_cb},
        {"odd_corner_block_cb", odd_corner_block_cb},
        {"odd_scalar_tile_cb", odd_scalar_tile_cb},
        {"odd_geom_in_cb", odd_geom_in_cb},
        {"odd_geom_index_cb", odd_geom_index_cb},
        {"odd_geom_weight_cb", odd_geom_weight_cb},
        {"odd_output_cb", odd_output_tile_cb},
        {"geom_kernel_cb", geom_kernel_cb},
        {"geom_scratch_cb", geom_scratch_cb},
        {"tiled_block_cb", tiled_block_cb},
        {"column_cb", column_cb},
        {"conv_weight_cb", conv_weight_cb},
        {"acc_cb", acc_cb},
        {"bias_cb", bias_cb},
        {"chunk_tiles", chunk_tiles},
        {"chunks", chunks},
        {"kernel_positions", s.kernel_positions},
        {"out_tiles", out_tiles},
        {"has_bias", has_bias ? 1u : 0u},
        {"relu", attrs.relu ? 1u : 0u},
        {"in_h_fp32", std::bit_cast<uint32_t>(static_cast<float>(s.in_h))},
        {"in_h_less_one_fp32", std::bit_cast<uint32_t>(static_cast<float>(s.in_h) - 1.0f)},
        {"in_w_fp32", std::bit_cast<uint32_t>(static_cast<float>(s.in_w))},
        {"in_w_less_one_fp32", std::bit_cast<uint32_t>(static_cast<float>(s.in_w) - 1.0f)},
    };
    // The geometry runs in fp32: its inputs and intermediates unpack straight to dest, as srcA would
    // round them to TF32. Each pass's matmul reduction accumulates in fp32 too. Half-sync dest, four
    // fp32 tiles a half, lets math fill one half while the packer drains the other.
    // HiFi2 for the corner products and the matmul: it drops the products' low bits, which costs
    // about as much accuracy as rounding the output to bfloat16, and the cycles it saves are on the
    // conv's critical path.
    ComputeConfigDescriptor compute_config{};
    compute_config.fp32_dest_acc_en = true;
    compute_config.math_fidelity = MathFidelity::HiFi2;
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    compute_config.unpack_to_dest_mode[geom_in_cb] = UnpackToDestMode::UnpackToDestFp32;
    compute_config.unpack_to_dest_mode[odd_geom_in_cb] = UnpackToDestMode::UnpackToDestFp32;
    compute_config.unpack_to_dest_mode[geom_kernel_cb] = UnpackToDestMode::UnpackToDestFp32;
    compute_config.unpack_to_dest_mode[geom_scratch_cb] = UnpackToDestMode::UnpackToDestFp32;
    compute_desc.config = compute_config;

    uint32_t block_cursor = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        const uint32_t blocks_here = core_blocks[i];
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
        reader_args.push_back(weight.buffer());
        reader_args.push_back(has_bias ? tensor_args.bias->buffer() : weight.buffer());
        reader_args.push_back(core == core_rect.start_coord ? 1u : 0u);
        reader_args.push_back(static_cast<uint32_t>(mcast_start.x));
        reader_args.push_back(static_cast<uint32_t>(mcast_start.y));
        reader_args.push_back(static_cast<uint32_t>(mcast_start.x));
        reader_args.push_back(static_cast<uint32_t>(mcast_start.y));
        reader_args.push_back(static_cast<uint32_t>(mcast_end.x));
        reader_args.push_back(static_cast<uint32_t>(mcast_end.y));
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

}  // namespace ttnn::operations::experimental::modulated_deform_conv2d
