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
    const uint32_t tile_bytes = operations::data_movement::pad_codegen::tile_page_bytes(input);

    const auto& in_shape = input.logical_shape();
    const Coord4 in_tiles{in_shape[0], in_shape[1], tt::div_up(in_shape[2], tile_h), tt::div_up(in_shape[3], tile_w)};
    const Coord4 out_tiles{attrs.N_out, attrs.C_out, tt::div_up(attrs.H_out, tile_h), tt::div_up(attrs.W_out, tile_w)};
    const Coord4 front_tiles{attrs.front_n, attrs.front_c, attrs.front_h / tile_h, attrs.front_w / tile_w};

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

    std::vector<uint32_t> writer_ct_args = {kCbOut, tile_bytes};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    writer_ct_args.push_back(batches.write_batch);

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

ProgramDescriptor build_fill_partial_tile(
    const PadCodegenParams& attrs, const Tensor& input, Tensor& output, tt::DataFormat format) {
    Buffer* buffer = output.buffer();

    const auto tile = input.tensor_spec().tile();
    const uint32_t tile_h = tile.get_height();
    const uint32_t tile_w = tile.get_width();
    const uint32_t tile_bytes = operations::data_movement::pad_codegen::tile_page_bytes(input);
    const uint32_t elem = input.element_size();
    const uint32_t face_bytes = operations::data_movement::pad_codegen::fill_face_bytes(input);

    const uint32_t Ht = tt::div_up(attrs.H_out, tile_h);
    const uint32_t Wt = tt::div_up(attrs.W_out, tile_w);
    const uint32_t h_mod = attrs.H_out % tile_h;
    const uint32_t w_mod = attrs.W_out % tile_w;
    const uint32_t planes = attrs.N_out * attrs.C_out;

    // Border tiles of one plane: the bottom tile row when H is partial, then the right tile column when W is
    // partial, the corner counted once in the row. The kernel walks the same order from each core's range.
    const uint32_t bottom = (h_mod != 0) ? Wt : 0;
    const uint32_t right = (w_mod != 0) ? ((h_mod != 0) ? Ht - 1 : Ht) : 0;
    const uint32_t total = planes * (bottom + right);

    const CoreSplit split = split_work(input, total);

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(
        split.all_cores, kCbOut, format, tile_bytes, operations::data_movement::pad_codegen::kFillScratchPages));
    desc.cbs.push_back(make_cb(split.all_cores, kCbPad, format, face_bytes, 1));

    std::vector<uint32_t> ct_args = {kCbOut, tile_bytes, elem, attrs.packed_pad_value, kCbPad};
    TensorAccessorArgs(*buffer).append_to(ct_args);

    KernelDescriptor kernel_desc;
    kernel_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/pad/codegen/kernels/fill_partial_tile.cpp";
    kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel_desc.core_ranges = split.all_cores;
    kernel_desc.compile_time_args = std::move(ct_args);
    kernel_desc.config = WriterConfigDescriptor{};

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = work_for_core(split, core);
        kernel_desc.emplace_runtime_args(core, RtArgs{buffer, start, n, Ht, Wt, h_mod, w_mod});
        start += n;
    }

    desc.kernels.push_back(std::move(kernel_desc));
    return desc;
}

}  // namespace

// The batching follows the L1 free right now, sampled after the output is allocated; compute_program_hash keys the
// same plan so a cached program never outlives the L1 it was sized for. The routing gate only checked the smallest
// plan against the static window, so live occupancy can still leave no room: that is an error here, not a fallback.
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
    if (operation_attributes.fill_partial_tile) {
        return build_fill_partial_tile(operation_attributes, input, tensor_return_value, format);
    }
    return build_pad_tiled(operation_attributes, *batches, input, tensor_return_value, format);
}

}  // namespace ttnn::prim
