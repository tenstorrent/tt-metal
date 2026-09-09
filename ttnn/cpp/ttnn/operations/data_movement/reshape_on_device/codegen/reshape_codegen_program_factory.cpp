// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_codegen_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// MODE_* values from reader_stick_interleaved_unified.cpp / sequencers.h.
constexpr uint32_t kModeSequential = 0;
constexpr uint32_t kModeNonaligned = 2;

// Writer PARTIAL modes from writer_reshape_rm.cpp; this port only emits the
// non-partial (whole-page) writer.
constexpr uint32_t kWriterPartialNone = 0;

struct CoreSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores_in_order;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

CoreSplit split_work(const Tensor& input, uint32_t total_work) {
    IDevice* device = input.device();
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid_size, total_work, /*row_wise=*/false);
    return CoreSplit{
        .all_cores = all_cores,
        .cores_in_order = corerange_to_cores(all_cores, num_cores, /*row_wise=*/false),
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .work_per_core_1 = work_per_core_1,
        .work_per_core_2 = work_per_core_2,
    };
}

uint32_t work_for_core(const CoreSplit& split, const CoreCoord& core) {
    if (split.core_group_1.contains(core)) {
        return split.work_per_core_1;
    }
    if (split.core_group_2.contains(core)) {
        return split.work_per_core_2;
    }
    return 0;
}

// Per-core stick-transport schedule, transliterated verbatim from spec.py's
// `_rm_per_core`. `n_split` is this core's share of the split axis (old sticks
// if split_by_old, else new sticks); `start` is its starting offset on that
// axis. Returns (num_old_reads, old_sticks_per_read, old_sticks_per_cb_push,
// num_new_reads, new_sticks_per_read, new_sticks_per_cb_push, curr_old, curr_new).
struct PerCoreSchedule {
    uint32_t num_old_reads = 0;
    uint32_t old_sticks_per_read = 0;
    uint32_t old_sticks_per_cb_push = 0;
    uint32_t num_new_reads = 0;
    uint32_t new_sticks_per_read = 0;
    uint32_t new_sticks_per_cb_push = 0;
    uint32_t curr_old = 0;
    uint32_t curr_new = 0;
};

PerCoreSchedule rm_per_core(
    uint32_t start,
    uint32_t n_split,
    bool split_by_old,
    uint32_t ratio,
    bool can_coalesce,
    uint32_t old_stick_size,
    uint32_t new_stick_size) {
    PerCoreSchedule sched;
    uint32_t n_old = 0;
    uint32_t n_new = 0;
    if (split_by_old) {
        n_old = n_split;
        n_new = n_old * ratio;
        sched.curr_old = start;
        sched.curr_new = start * ratio;
    } else {
        n_new = n_split;
        n_old = n_new * ratio;
        sched.curr_new = start;
        sched.curr_old = start * ratio;
    }

    if (!can_coalesce) {
        if (split_by_old) {
            sched.num_old_reads = n_old;
            sched.old_sticks_per_read = 1;
            sched.old_sticks_per_cb_push = ratio;
            sched.num_new_reads = n_new;
            sched.new_sticks_per_read = 1;
            sched.new_sticks_per_cb_push = 1;
        } else {
            sched.num_old_reads = n_new;
            sched.old_sticks_per_read = ratio;
            sched.old_sticks_per_cb_push = 1;
            sched.num_new_reads = n_new;
            sched.new_sticks_per_read = 1;
            sched.new_sticks_per_cb_push = 1;
        }
    } else if (old_stick_size > new_stick_size) {
        if (n_old != 0) {
            sched.num_old_reads = merge_num_sticks_to_read(n_old, old_stick_size, kReshapeMaxReadSize);
            sched.old_sticks_per_read = n_old / sched.num_old_reads;
            sched.old_sticks_per_cb_push = sched.old_sticks_per_read * ratio;
            sched.new_sticks_per_cb_push = sched.old_sticks_per_cb_push;
            sched.new_sticks_per_read = sched.old_sticks_per_cb_push;
            sched.num_new_reads = n_new / sched.new_sticks_per_read;
        }
    } else {
        if (n_new != 0) {
            sched.num_new_reads = merge_num_sticks_to_read(n_new, new_stick_size, kReshapeMaxReadSize);
            sched.new_sticks_per_read = n_new / sched.num_new_reads;
            sched.new_sticks_per_cb_push = sched.new_sticks_per_read;
            sched.old_sticks_per_cb_push = sched.new_sticks_per_cb_push;
            sched.old_sticks_per_read = sched.old_sticks_per_cb_push * ratio;
            sched.num_old_reads = n_old / sched.old_sticks_per_read;
        }
    }
    return sched;
}

// L1 alignment used by the RM CB-sizing rounding, matching the generator's `_L1_ALIGN`.
constexpr uint32_t kL1Align = 16;

// Alignment margin the MODE_NONALIGNED scratch CB reserves per batched read,
// matching the generator's `_NABATCH*(aligned+64)` sizing.
constexpr uint32_t kNonalignedScratchMargin = 64;

uint32_t round_up_u32(uint32_t v, uint32_t align) { return ((v + align - 1) / align) * align; }

// Physical page pitch for an interleaved RM buffer, matching the generator's
// `interleaved_accessor_page_size`: TensorAccessorArgs for an interleaved buffer
// yields exactly 2 compile-time args, [1] being the aligned page size.
uint32_t interleaved_aligned_page_size(Buffer* buffer) { return static_cast<uint32_t>(buffer->aligned_page_size()); }

}  // namespace

ProgramDescriptor ReshapeCodegenRmProgramFactory::create_descriptor(
    const ReshapeCodegenParams& /*operation_attributes*/,
    const ReshapeCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(src_buffer != nullptr, "ReshapeCodegen input must be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "ReshapeCodegen output must be allocated on device!");

    const auto& in_shape = input.padded_shape();
    const auto& out_shape = output.padded_shape();
    TT_FATAL(in_shape.rank() >= 1 && out_shape.rank() >= 1, "ReshapeCodegen RM requires rank >= 1 tensors");

    uint32_t num_old_sticks = 1;
    for (uint32_t i = 0; i + 1 < in_shape.rank(); ++i) {
        num_old_sticks *= in_shape[i];
    }
    uint32_t num_new_sticks = 1;
    for (uint32_t i = 0; i + 1 < out_shape.rank(); ++i) {
        num_new_sticks *= out_shape[i];
    }

    const uint32_t old_stick_size = in_shape[-1] * input.element_size();
    const uint32_t new_stick_size = out_shape[-1] * output.element_size();

    const uint32_t old_aligned = interleaved_aligned_page_size(src_buffer);
    const uint32_t new_aligned = interleaved_aligned_page_size(dst_buffer);

    const uint32_t mx = std::max(old_stick_size, new_stick_size);
    const uint32_t mn = std::min(old_stick_size, new_stick_size);
    TT_FATAL(
        mn > 0 && mx % mn == 0,
        "ReshapeCodegen RM requires one stick size to divide the other ({} vs {})",
        old_stick_size,
        new_stick_size);
    const uint32_t ratio = mx / mn;

    const bool old_aligned_match = (old_stick_size == old_aligned);
    const bool new_aligned_match = (new_stick_size == new_aligned);
    const bool can_coalesce = old_aligned_match && new_aligned_match;

    const bool split_by_old = old_stick_size > new_stick_size;
    const uint32_t split_sticks = split_by_old ? num_old_sticks : num_new_sticks;

    const CoreSplit split = split_work(input, split_sticks);
    const uint32_t max_sticks_per_core = std::max(split.work_per_core_1, split.work_per_core_2);
    const bool split_nonaligned = split_by_old && !can_coalesce;

    uint32_t cb_page_size = 0;
    uint32_t cb_total = 0;
    if (split_nonaligned) {
        cb_page_size = round_up_u32(new_stick_size, kL1Align);
        const uint32_t max_new_per_core = max_sticks_per_core * ratio;
        cb_total = max_new_per_core * cb_page_size * 2;
    } else if (!can_coalesce) {
        cb_page_size = round_up_u32(new_stick_size, kL1Align);
        const uint32_t max_new_per_core = max_sticks_per_core;
        cb_total = max_new_per_core * cb_page_size * 2;
    } else if (split_by_old) {
        cb_page_size = new_stick_size;
        cb_total = max_sticks_per_core * old_stick_size;
    } else {
        cb_page_size = new_stick_size;
        cb_total = max_sticks_per_core * new_stick_size;
    }
    TT_FATAL(cb_page_size > 0, "ReshapeCodegen RM computed a zero-size CB page");

    const uint32_t scratch_size = can_coalesce ? 0 : kReshapeNabatch * (old_aligned + kNonalignedScratchMargin);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    ProgramDescriptor desc;
    constexpr uint8_t kCbIn = 0;
    constexpr uint8_t kCbScratch = 1;

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_total,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbIn,
            .data_format = cb_data_format,
            .page_size = cb_page_size,
        }}},
    });
    if (!can_coalesce) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = scratch_size,
            .core_ranges = split.all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = kCbScratch,
                .data_format = cb_data_format,
                .page_size = scratch_size,
            }}},
        });
    }

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/codegen/kernels/"
        "reader_stick_interleaved_unified.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.named_compile_time_args = {
        {"mode", can_coalesce ? kModeSequential : kModeNonaligned},
        {"cb_id", kCbIn},
        {"stick_bytes", old_stick_size},
        {"aligned_page_size", old_aligned},
        {"seq_id", 0},
        {"batch", 1},
        {"nabatch", kReshapeNabatch},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct_args = {kCbIn, new_stick_size, new_aligned};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    writer_ct_args.push_back(kWriterPartialNone);
    writer_ct_args.push_back(0);  // partial_bytes, unused when PARTIAL==0
    writer_ct_args.push_back(static_cast<uint32_t>(tt::tt_metal::hal::get_noc_max_burst_size_bytes()));

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/codegen/kernels/writer_reshape_rm.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = work_for_core(split, core);
        const PerCoreSchedule sched =
            rm_per_core(start, n, split_by_old, ratio, can_coalesce, old_stick_size, new_stick_size);

        if (can_coalesce) {
            reader_desc.emplace_runtime_args(
                core,
                {src_buffer,
                 sched.num_old_reads,
                 sched.old_sticks_per_read,
                 sched.old_sticks_per_cb_push,
                 sched.curr_old});
        } else {
            reader_desc.emplace_runtime_args(
                core,
                {src_buffer,
                 sched.num_old_reads,
                 sched.old_sticks_per_read,
                 sched.curr_old,
                 old_stick_size,
                 new_stick_size,
                 ratio,
                 split_nonaligned ? 1u : 0u,
                 static_cast<uint32_t>(kCbScratch)});
        }
        writer_desc.emplace_runtime_args(
            core,
            {dst_buffer, sched.num_new_reads, sched.new_sticks_per_read, sched.new_sticks_per_cb_push, sched.curr_new});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

ProgramDescriptor ReshapeCodegenTileProgramFactory::create_descriptor(
    const ReshapeCodegenParams& /*operation_attributes*/,
    const ReshapeCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(src_buffer != nullptr, "ReshapeCodegen input must be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "ReshapeCodegen output must be allocated on device!");

    const auto& in_shape = input.padded_shape();
    const auto& out_shape = output.padded_shape();
    TT_FATAL(in_shape.rank() >= 1 && out_shape.rank() >= 1, "ReshapeCodegen TILE requires rank >= 1 tensors");

    const uint32_t W_in = in_shape[-1];
    const uint32_t W_out = out_shape[-1];
    TT_FATAL(
        W_in % tt::constants::TILE_WIDTH == 0 && W_out % tt::constants::TILE_WIDTH == 0,
        "ReshapeCodegen TILE requires tile-aligned widths (in={}, out={})",
        W_in,
        W_out);
    const uint32_t Wt_in = W_in / tt::constants::TILE_WIDTH;
    const uint32_t Wt_out = W_out / tt::constants::TILE_WIDTH;

    const uint32_t lcm_w = std::lcm(W_in, W_out);
    const uint32_t in_tile_rows_per_chunk = lcm_w / W_in;
    const uint32_t out_tile_rows_per_chunk = lcm_w / W_out;
    const uint32_t in_tiles_per_chunk = in_tile_rows_per_chunk * Wt_in;
    const uint32_t out_tiles_per_chunk = out_tile_rows_per_chunk * Wt_out;

    const uint64_t padded_volume = input.physical_volume();
    TT_FATAL(
        padded_volume % (static_cast<uint64_t>(tt::constants::TILE_HEIGHT) * W_in) == 0,
        "ReshapeCodegen TILE: padded volume {} not divisible by TILE_HEIGHT*W_in ({})",
        padded_volume,
        static_cast<uint64_t>(tt::constants::TILE_HEIGHT) * W_in);
    const uint32_t Ht_in = static_cast<uint32_t>(padded_volume / (tt::constants::TILE_HEIGHT * W_in));
    TT_FATAL(
        Ht_in % in_tile_rows_per_chunk == 0,
        "ReshapeCodegen TILE: Ht_in={} not divisible by in_tile_rows_per_chunk={}",
        Ht_in,
        in_tile_rows_per_chunk);
    const uint32_t total_chunks = Ht_in / in_tile_rows_per_chunk;

    const CoreSplit split = split_work(input, total_chunks);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_bytes = tt::tile_size(cb_data_format);

    // 4-byte datums need 32-bit DEST accumulation; also caps pack_untilize's
    // max block-ct-dim (4 vs 8) to avoid DEST overflow at multi-tile-column shapes.
    const bool fp32 = (input.element_size() == 4);
    const uint32_t max_bct = fp32 ? 4 : 8;

    constexpr uint8_t kCbIn = 0;
    constexpr uint8_t kCbMid = 1;
    constexpr uint8_t kCbOut = 16;
    const uint32_t mid_pages = in_tiles_per_chunk;
    const uint32_t cb_out_depth = std::max(out_tiles_per_chunk, kReshapeTileWriteBatch * 2);

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_tiles_per_chunk * tile_bytes,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbIn,
            .data_format = cb_data_format,
            .page_size = tile_bytes,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = mid_pages * tile_bytes,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbMid,
            .data_format = cb_data_format,
            .page_size = tile_bytes,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_out_depth * tile_bytes,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbOut,
            .data_format = cb_data_format,
            .page_size = tile_bytes,
        }}},
    });

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/codegen/kernels/"
        "reader_tile_interleaved_unified.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.named_compile_time_args = {
        {"seq_id", 0},  // SEQ_IDENTITY
        {"cb_id", kCbIn},
        {"batch", 1},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    const uint32_t out_page = round_up_u32(tile_bytes, tt::tt_metal::hal::get_dram_alignment());
    std::vector<uint32_t> writer_ct_args = {kCbOut, out_page};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    writer_ct_args.push_back(kReshapeTileWriteBatch);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/codegen/kernels/writer_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_ct_args = {
        kCbIn, kCbMid, kCbOut, Wt_in, Wt_out, in_tile_rows_per_chunk, out_tile_rows_per_chunk, max_bct};
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/codegen/kernels/compute_reshape_tile.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = split.all_cores;
    compute_desc.compile_time_args = std::move(compute_ct_args);
    compute_desc.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32};

    uint32_t in_tile_offset = 0;
    uint32_t out_tile_offset = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n_chunks = work_for_core(split, core);
        const uint32_t n_in_tiles = n_chunks * in_tiles_per_chunk;
        const uint32_t n_out_tiles = n_chunks * out_tiles_per_chunk;

        reader_desc.emplace_runtime_args(core, {src_buffer, n_in_tiles, in_tile_offset});
        writer_desc.emplace_runtime_args(core, {dst_buffer, n_out_tiles, out_tile_offset});
        compute_desc.emplace_runtime_args(core, {n_chunks});

        in_tile_offset += n_in_tiles;
        out_tile_offset += n_out_tiles;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}
