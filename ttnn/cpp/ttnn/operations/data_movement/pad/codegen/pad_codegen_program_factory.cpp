// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_program_factory.hpp"

#include <algorithm>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_supported.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

constexpr uint32_t kCbOut = 0;
constexpr uint32_t kCbPad = 1;

// SEQ_PAD, see common/kernels/codegen/sequencers.h.
constexpr uint32_t kSeqPad = 5;

constexpr const char* kSharedKernelDir = "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen";

using RtArgs = std::vector<std::variant<uint32_t, Buffer*>>;

struct CoreSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores_in_order;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

// row_wise=false for both the split and the core walk so the per-core ranges line up.
CoreSplit split_work(const Tensor& input, uint32_t total_work) {
    const auto grid_size = input.device()->compute_with_storage_grid_size();
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

CBDescriptor make_cb(
    const CoreRangeSet& cores, uint32_t index, tt::DataFormat format, uint32_t page_size, uint32_t num_pages) {
    return CBDescriptor{
        .total_size = num_pages * page_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = page_size,
        }}},
    };
}

// Elements of one dim that lie before `coord` inside the data window [front, front + extent).
uint32_t data_before(uint32_t coord, uint32_t front, uint32_t extent) {
    return std::clamp(coord, front, front + extent) - front;
}

bool in_window(uint32_t coord, uint32_t front, uint32_t extent) { return coord >= front && coord < front + extent; }

struct Coord4 {
    uint32_t n, c, h, w;
};

Coord4 unflatten(uint32_t index, uint32_t C, uint32_t H, uint32_t W) {
    Coord4 r;
    r.w = index % W;
    index /= W;
    r.h = index % H;
    index /= H;
    r.c = index % C;
    r.n = index / C;
    return r;
}

// Input tiles consumed before output tile `p`.
uint32_t source_index_before(const Coord4& p, const Coord4& front, const Coord4& in) {
    uint32_t count = data_before(p.n, front.n, in.n) * in.c * in.h * in.w;
    if (in_window(p.n, front.n, in.n)) {
        count += data_before(p.c, front.c, in.c) * in.h * in.w;
        if (in_window(p.c, front.c, in.c)) {
            count += data_before(p.h, front.h, in.h) * in.w;
            if (in_window(p.h, front.h, in.h)) {
                count += data_before(p.w, front.w, in.w);
            }
        }
    }
    return count;
}

ProgramDescriptor build_pad_tiled(
    const PadCodegenParams& attrs,
    const operations::data_movement::pad_codegen::BatchPlan& batches,
    const Tensor& input,
    Tensor& output,
    tt::DataFormat format) {
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    const auto tile = input.tensor_spec().tile();
    const uint32_t tile_h = tile.get_height();
    const uint32_t tile_w = tile.get_width();
    const uint32_t tile_bytes = operations::data_movement::pad_codegen::dram_aligned_tile_bytes(input, input.dtype());

    const auto& in_shape = input.logical_shape();
    const Coord4 in_tiles{in_shape[0], in_shape[1], tt::div_up(in_shape[2], tile_h), tt::div_up(in_shape[3], tile_w)};
    const auto& out_shape = attrs.output_logical_shape;
    const Coord4 out_tiles{
        out_shape[0], out_shape[1], tt::div_up(out_shape[2], tile_h), tt::div_up(out_shape[3], tile_w)};
    // The gate declines leading padding, so the data window starts at the origin of every dim.
    const Coord4 front_tiles{};

    const uint32_t total = out_tiles.n * out_tiles.c * out_tiles.h * out_tiles.w;
    const CoreSplit split = split_work(input, total);

    ProgramDescriptor desc;
    const uint32_t depth = operations::data_movement::pad_codegen::cb_depth(batches.read_batch, batches.write_batch);
    desc.cbs.push_back(make_cb(split.all_cores, kCbOut, format, tile_bytes, depth));
    desc.cbs.push_back(make_cb(split.all_cores, kCbPad, format, tile_bytes, 1));

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kSharedKernelDir) + "/reader_tile_interleaved_unified.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.named_compile_time_args = {
        {"seq_id", kSeqPad},
        {"cb_id", kCbOut},
        {"batch", batches.read_batch},
        {"src_page_pitch", 0},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct_args = {kCbOut, tile_bytes, batches.write_batch};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(kSharedKernelDir) + "/writer_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = work_for_core(split, core);
        const Coord4 p = unflatten(start, out_tiles.c, out_tiles.h, out_tiles.w);
        const uint32_t src_start = source_index_before(p, front_tiles, in_tiles);
        // Slot order is the reader's ArgsPad layout; the kernel reads these by position.
        reader_desc.emplace_runtime_args(
            core,
            RtArgs{
                src_buffer,
                n,
                src_start,
                p.w,
                p.h,
                p.c,
                p.n,
                in_tiles.w,
                in_tiles.h,
                in_tiles.c,
                in_tiles.n,
                out_tiles.w,
                out_tiles.h,
                out_tiles.c,
                out_tiles.n,
                tile_bytes,
                attrs.packed_pad_value,
                kCbPad,
                front_tiles.w,
                front_tiles.h,
                front_tiles.c,
                front_tiles.n});
        writer_desc.emplace_runtime_args(core, RtArgs{dst_buffer, n, start});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace

// The batching follows the L1 free right now, sampled after the output is allocated; compute_program_hash keys the
// same plan so a cached program never outlives the L1 it was sized for. Auto routing already declined when the live
// budget had no room, but forced codegen only checked the static window, so reaching here without room is an error.
ProgramDescriptor PadCodegenProgramFactory::create_descriptor(
    const PadCodegenParams& operation_attributes, const PadCodegenInputs& tensor_args, Tensor& tensor_return_value) {
    namespace pc = operations::data_movement::pad_codegen;
    const Tensor& input = tensor_args.input;

    const auto batches = pc::plan_batches_live(input, operation_attributes);
    if (!batches.has_value()) {
        TT_THROW(
            "ttnn.pad: codegen path needs {} B of L1 per core for its circular buffers but only {} B is free",
            pc::cb_footprint_bytes(input, operation_attributes, pc::BatchPlan{}),
            operations::data_movement::get_max_l1_space(input));
    }

    const tt::DataFormat format = datatype_to_dataformat_converter(input.dtype());
    return build_pad_tiled(operation_attributes, *batches, input, tensor_return_value, format);
}

}  // namespace ttnn::prim
