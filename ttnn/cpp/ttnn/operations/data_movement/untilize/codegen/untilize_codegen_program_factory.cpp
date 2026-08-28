// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_program_factory.hpp"

#include <algorithm>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.hpp"
#include "untilize_codegen_cb_plan.hpp"
#include "untilize_codegen_device_operation.hpp"
#include "untilize_codegen_supported.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Kernels were copied verbatim from the codegen builder into codegen/kernels/ in phase 3.
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels";
constexpr uint32_t kCbIn = tt::CBIndex::c_0;
constexpr uint32_t kCbOut = tt::CBIndex::c_16;
constexpr uint32_t kSeqIdentity = 0;  // mirrors common/templates/sequencers.h SEQ_IDENTITY

std::string kernel_path(const char* name) { return std::string(kKernelDir) + "/" + name; }

// Mirrors spec.py's _needs_dst_accum: 32-bit datums need 32-bit DEST accumulation in
// pack_untilize. Always false for this port's supported_by_codegen scope (bf16/bf8_b only);
// kept general to stay a faithful transliteration of the source builder.
bool needs_dst_accum(DataType dtype) {
    return dtype == DataType::FLOAT32 || dtype == DataType::INT32 || dtype == DataType::UINT32;
}

using untilize_codegen_detail::CbPlan;
using untilize_codegen_detail::choose_2d_ncol;
using untilize_codegen_detail::compute_block_ct_dim;
using untilize_codegen_detail::plan_cb_depths;

// DRAM-interleaved tile CBs must step at the device's real DRAM page pitch, not the raw tile
// byte size (a no-op for bf16/bf8_b tile sizes, both already multiples of every supported
// arch's DRAM alignment, but computed from the real device per the porting guide rather than
// a hardcoded arch constant).
uint32_t aligned_tile_page_size(uint32_t tile_bytes) {
    uint32_t align = tt::tt_metal::hal::get_dram_alignment();
    return ((tile_bytes + align - 1) / align) * align;
}

CBDescriptor make_tile_cb(
    uint32_t cb_id, tt::DataFormat fmt, uint32_t depth, uint32_t tile_bytes, const CoreRangeSet& cores) {
    uint32_t page = aligned_tile_page_size(tile_bytes);
    return CBDescriptor{
        .total_size = depth * page,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = fmt,
            .page_size = page,
        }}},
    };
}

KernelDescriptor make_reader(const CoreRangeSet& cores, Buffer* in_buf, uint32_t read_batch) {
    KernelDescriptor reader;
    reader.kernel_source = kernel_path("reader_tile_interleaved_unified.cpp");
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    TensorAccessorArgs(*in_buf).append_to(reader.compile_time_args);
    reader.named_compile_time_args = {
        {"seq_id", kSeqIdentity},
        {"cb_id", kCbIn},
        {"batch", read_batch},
        // reader_tile_interleaved_unified reads get_named_compile_time_arg_val("src_page_pitch");
        // builder_utils injects it (0 = use the accessor's page size). Absent -> JIT compile fails.
        {"src_page_pitch", 0},
    };
    reader.config = ReaderConfigDescriptor{};
    return reader;
}

KernelDescriptor make_compute(
    const CoreRangeSet& cores, uint32_t per_core_block_cnt, uint32_t wt, uint32_t max_bct, bool fp32) {
    KernelDescriptor compute;
    compute.kernel_source = kernel_path("compute_untilize.cpp");
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = {per_core_block_cnt, wt, kCbIn, kCbOut, max_bct};
    compute.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32};
    return compute;
}

struct CommonArgs {
    IDevice* device;
    Buffer* in_buf;
    Buffer* out_buf;
    tt::DataFormat in_fmt;
    tt::DataFormat out_fmt;
    uint32_t in_tile_size;
    uint32_t out_tile_size;
    uint32_t tile_size_for_planning;
    uint32_t out_elem_size;
    bool fp32;
    uint32_t max_bct;
    // Live L1 headroom, sampled once in create_descriptor -- see kUsableL1Note.
    uint64_t usable_l1;
};

// Per-tile-row split (build_untilize_tile's default path / _build_untilize_tile_cliff).
// Splits total_tile_rows across up to the device's core grid; an uneven split produces a
// second ("cliff") compute-kernel core group with its own per-core tile-row count.
std::optional<ProgramDescriptor> build_main_split(const CommonArgs& a, uint32_t wt, uint32_t total_tile_rows) {
    auto grid = a.device->compute_with_storage_grid_size();
    auto [_num_cores, core_range, cg1, cg2, tpc1, tpc2] =
        tt::tt_metal::split_work_to_cores(grid, total_tile_rows, /*row_wise=*/true);

    uint32_t block_ct_dim = compute_block_ct_dim(wt, a.fp32);
    auto maybe_plan = plan_cb_depths(a.usable_l1, wt, a.tile_size_for_planning, block_ct_dim);
    if (!maybe_plan.has_value()) {
        return std::nullopt;
    }
    const CbPlan& plan = *maybe_plan;

    // Row stride used both as the writer's TensorAccessor page pitch and the CB byte stride
    // between physical tile-rows: the FULL padded row (Wt tiles wide), never the logical
    // width. This lets the same writer template handle non-tile-aligned logical shapes
    // correctly by preserving column padding in the physical output buffer -- exactly the
    // scheme native UntilizeDeviceOperation::compute_output_specs uses (padded_shape carried
    // through unchanged; only the tensor's logical_shape metadata crops it on read).
    uint32_t row_size_bytes = wt * TILE_WIDTH * a.out_elem_size;

    ProgramDescriptor desc;
    desc.cbs.push_back(make_tile_cb(kCbIn, a.in_fmt, plan.cb_in_depth, a.in_tile_size, core_range));
    desc.cbs.push_back(make_tile_cb(kCbOut, a.out_fmt, plan.cb_out_depth, a.out_tile_size, core_range));

    KernelDescriptor reader = make_reader(core_range, a.in_buf, plan.read_batch);

    KernelDescriptor writer;
    writer.kernel_source = kernel_path("writer_untilize_interleaved.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = core_range;
    writer.compile_time_args = {kCbOut, row_size_bytes};
    TensorAccessorArgs(*a.out_buf).append_to(writer.compile_time_args);
    writer.compile_time_args.push_back(wt);
    writer.config = WriterConfigDescriptor{};

    uint32_t assigned = 0;
    auto emit_group = [&](const CoreRangeSet& group, uint32_t wpc) {
        if (group.empty()) {
            return;
        }
        for (const auto& core : corerange_to_cores(group, std::nullopt, true)) {
            uint32_t n = std::min(wpc, total_tile_rows - assigned);
            reader.emplace_runtime_args(core, {a.in_buf, n * wt, assigned * wt});
            writer.emplace_runtime_args(core, {a.out_buf, n, assigned * TILE_HEIGHT, row_size_bytes, 0u, 0u, 0u});
            assigned += n;
        }
    };
    emit_group(cg1, tpc1);
    emit_group(cg2, tpc2);

    // Kernel order [reader, writer, compute...] mirrors spec.py's build_untilize_tile: the
    // single-compute case reorders assemble()'s [reader, compute, writer] back to legacy order,
    // and the cliff case builds its kernel list in legacy order directly.
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    if (cg2.empty()) {
        desc.kernels.push_back(make_compute(core_range, tpc1, wt, a.max_bct, a.fp32));
    } else {
        desc.kernels.push_back(make_compute(cg1, tpc1, wt, a.max_bct, a.fp32));
        desc.kernels.push_back(make_compute(cg2, tpc2, wt, a.max_bct, a.fp32));
    }
    return desc;
}

// Column-parallel split (_build_untilize_column_parallel): single tile-row, Wt>1 -- splits
// tile-COLUMNS across cores instead of tile-rows.
std::optional<ProgramDescriptor> build_column_parallel(const CommonArgs& a, uint32_t wt) {
    auto grid = a.device->compute_with_storage_grid_size();
    auto [_num_cores, core_range, cg1, cg2, tpc1, tpc2] =
        tt::tt_metal::split_work_to_cores(grid, wt, /*row_wise=*/true);

    uint32_t max_tpc = std::max(tpc1, cg2.empty() ? 0u : tpc2);
    uint32_t block_ct_dim = compute_block_ct_dim(max_tpc, a.fp32);
    auto maybe_plan = plan_cb_depths(a.usable_l1, max_tpc, a.tile_size_for_planning, block_ct_dim);
    if (!maybe_plan.has_value()) {
        return std::nullopt;
    }
    const CbPlan& plan = *maybe_plan;

    uint32_t full_stick_size = wt * TILE_WIDTH * a.out_elem_size;

    ProgramDescriptor desc;
    desc.cbs.push_back(make_tile_cb(kCbIn, a.in_fmt, plan.cb_in_depth, a.in_tile_size, core_range));
    desc.cbs.push_back(make_tile_cb(kCbOut, a.out_fmt, plan.cb_out_depth, a.out_tile_size, core_range));

    KernelDescriptor reader = make_reader(core_range, a.in_buf, plan.read_batch);

    KernelDescriptor writer;
    writer.kernel_source = kernel_path("writer_untilize_col_parallel.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = core_range;
    writer.compile_time_args = {kCbOut, full_stick_size};
    TensorAccessorArgs(*a.out_buf).append_to(writer.compile_time_args);
    writer.config = WriterConfigDescriptor{};

    uint32_t assigned = 0;
    auto emit_group = [&](const CoreRangeSet& group, uint32_t wpc) {
        if (group.empty()) {
            return;
        }
        for (const auto& core : corerange_to_cores(group, std::nullopt, true)) {
            uint32_t n = std::min(wpc, wt - assigned);
            reader.emplace_runtime_args(core, {a.in_buf, n, assigned});
            writer.emplace_runtime_args(
                core,
                {a.out_buf,
                 TILE_HEIGHT,
                 assigned * TILE_WIDTH * a.out_elem_size,
                 n * TILE_WIDTH * a.out_elem_size,
                 n,
                 0u});
            assigned += n;
        }
    };
    emit_group(cg1, tpc1);
    emit_group(cg2, tpc2);

    // Kernel order [reader, writer, compute...] mirrors _build_untilize_column_parallel's
    // kernels list, built directly in that order for assemble_custom.
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    if (cg2.empty()) {
        desc.kernels.push_back(make_compute(core_range, 1, tpc1, a.max_bct, a.fp32));
    } else {
        desc.kernels.push_back(make_compute(cg1, 1, tpc1, a.max_bct, a.fp32));
        desc.kernels.push_back(make_compute(cg2, 1, tpc2, a.max_bct, a.fp32));
    }
    return desc;
}

// 2D (tile-row x column-block) split (_build_untilize_2d_column): raises core utilization
// when total_tile_rows alone would leave cores idle. Every one of total_tile_rows*ncol cores
// owns exactly one (tile-row, column-block) unit of tpc = Wt/ncol tiles.
std::optional<ProgramDescriptor> build_2d_column(
    const CommonArgs& a, uint32_t wt, uint32_t total_tile_rows, uint32_t ncol) {
    uint32_t tpc = wt / ncol;
    uint32_t num_units = total_tile_rows * ncol;

    auto grid = a.device->compute_with_storage_grid_size();
    auto [_num_cores, core_range, _cg1, _cg2, _tpc1, _tpc2] =
        tt::tt_metal::split_work_to_cores(grid, num_units, /*row_wise=*/true);

    uint32_t block_ct_dim = compute_block_ct_dim(tpc, a.fp32);
    auto maybe_plan = plan_cb_depths(a.usable_l1, tpc, a.tile_size_for_planning, block_ct_dim);
    if (!maybe_plan.has_value()) {
        return std::nullopt;
    }
    const CbPlan& plan = *maybe_plan;

    uint32_t full_stick_size = wt * TILE_WIDTH * a.out_elem_size;
    uint32_t col_chunk_bytes = tpc * TILE_WIDTH * a.out_elem_size;

    ProgramDescriptor desc;
    desc.cbs.push_back(make_tile_cb(kCbIn, a.in_fmt, plan.cb_in_depth, a.in_tile_size, core_range));
    desc.cbs.push_back(make_tile_cb(kCbOut, a.out_fmt, plan.cb_out_depth, a.out_tile_size, core_range));

    KernelDescriptor reader = make_reader(core_range, a.in_buf, plan.read_batch);

    KernelDescriptor writer;
    writer.kernel_source = kernel_path("writer_untilize_col_parallel.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = core_range;
    writer.compile_time_args = {kCbOut, full_stick_size};
    TensorAccessorArgs(*a.out_buf).append_to(writer.compile_time_args);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute = make_compute(core_range, 1, tpc, a.max_bct, a.fp32);

    auto cores = corerange_to_cores(core_range, std::nullopt, true);
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        uint32_t tile_row = i / ncol;
        uint32_t col_block = i % ncol;
        uint32_t start_tile = tile_row * wt + col_block * tpc;
        reader.emplace_runtime_args(core, {a.in_buf, tpc, start_tile});
        writer.emplace_runtime_args(
            core, {a.out_buf, TILE_HEIGHT, col_block * col_chunk_bytes, col_chunk_bytes, tpc, tile_row * TILE_HEIGHT});
    }

    // Kernel order [reader, writer, compute] mirrors _build_untilize_2d_column's kernels list.
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

// Mirrors spec.py's _count_valid_sticks: counts sticks in [start, start+count) whose position
// within a `batch_h`-sized physical batch falls below `valid_per_batch`. The with-unpadding
// writer's running out_page_offset needs this to skip padding rows and land on the same compact
// stick numbering the reference (build_untilize_with_unpadding) produces.
uint32_t count_valid_sticks(uint32_t start, uint32_t count, uint32_t batch_h, uint32_t valid_per_batch) {
    if (valid_per_batch == 0 || batch_h == 0) {
        return count;
    }
    uint32_t full_batches = count / batch_h;
    uint32_t remainder = count % batch_h;
    uint32_t pos = start % batch_h;
    uint32_t n = full_batches * valid_per_batch;
    for (uint32_t i = 0; i < remainder; ++i) {
        if ((pos + i) % batch_h < valid_per_batch) {
            ++n;
        }
    }
    return n;
}

// Interleaved TILE -> RM + unpadding (build_untilize_with_unpadding): the reference's path for any
// non-tile-aligned bf16 logical shape. Same reader/compute as the aligned per-tile-row split; the
// writer additionally skips physical pad sticks and writes only the unpadded row width, producing
// the compact output UntilizeCodegenDeviceOperation::compute_output_specs's non-aligned branch
// declares. Cliff-capable (two compute kernels), same as build_main_split.
std::optional<ProgramDescriptor> build_with_unpadding(
    const CommonArgs& a,
    uint32_t wt,
    uint32_t total_tile_rows,
    uint32_t h_unpadded_per_batch,
    uint32_t w_unpadded,
    uint32_t padded_batch_h) {
    auto grid = a.device->compute_with_storage_grid_size();
    auto [_num_cores, core_range, cg1, cg2, tpc1, tpc2] =
        tt::tt_metal::split_work_to_cores(grid, total_tile_rows, /*row_wise=*/true);

    uint32_t block_ct_dim = compute_block_ct_dim(wt, a.fp32);
    auto maybe_plan = plan_cb_depths(a.usable_l1, wt, a.tile_size_for_planning, block_ct_dim);
    if (!maybe_plan.has_value()) {
        return std::nullopt;
    }
    const CbPlan& plan = *maybe_plan;

    uint32_t unpadded_row_bytes = w_unpadded * a.out_elem_size;
    uint32_t padded_row_bytes = wt * TILE_WIDTH * a.out_elem_size;

    ProgramDescriptor desc;
    // Both CBs use the input format/tile size: build_untilize_with_unpadding only ever runs on a
    // dtype that is already RM-representable (bf8_b is cast to bf16 upstream of this port's
    // scope), so in and out CBs are the same dtype here, unlike build_untilize_tile's bf8_b->bf16
    // repack.
    desc.cbs.push_back(make_tile_cb(kCbIn, a.in_fmt, plan.cb_in_depth, a.in_tile_size, core_range));
    desc.cbs.push_back(make_tile_cb(kCbOut, a.in_fmt, plan.cb_out_depth, a.in_tile_size, core_range));

    KernelDescriptor reader = make_reader(core_range, a.in_buf, plan.read_batch);

    KernelDescriptor writer;
    writer.kernel_source = kernel_path("writer_untilize_interleaved.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = core_range;
    writer.compile_time_args = {kCbOut, unpadded_row_bytes};
    TensorAccessorArgs(*a.out_buf).append_to(writer.compile_time_args);
    writer.compile_time_args.push_back(wt);
    writer.config = WriterConfigDescriptor{};

    // out_page_offset is a running accumulator carried across cores in ascending order (mirrors
    // spec.py's stateful `_state["off"]` closure, invoked once per core by emit_per_core_rt).
    uint32_t assigned = 0;
    uint32_t out_page_offset = 0;
    auto emit_group = [&](const CoreRangeSet& group, uint32_t wpc) {
        if (group.empty()) {
            return;
        }
        for (const auto& core : corerange_to_cores(group, std::nullopt, true)) {
            uint32_t n = std::min(wpc, total_tile_rows - assigned);
            uint32_t start_stick = assigned * TILE_HEIGHT;
            reader.emplace_runtime_args(core, {a.in_buf, n * wt, assigned * wt});
            writer.emplace_runtime_args(
                core,
                {a.out_buf, n, start_stick, padded_row_bytes, h_unpadded_per_batch, padded_batch_h, out_page_offset});
            out_page_offset += count_valid_sticks(start_stick, n * TILE_HEIGHT, padded_batch_h, h_unpadded_per_batch);
            assigned += n;
        }
    };
    emit_group(cg1, tpc1);
    emit_group(cg2, tpc2);

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    if (cg2.empty()) {
        desc.kernels.push_back(make_compute(core_range, tpc1, wt, a.max_bct, a.fp32));
    } else {
        desc.kernels.push_back(make_compute(cg1, tpc1, wt, a.max_bct, a.fp32));
        desc.kernels.push_back(make_compute(cg2, tpc2, wt, a.max_bct, a.fp32));
    }
    return desc;
}

// Native untilize factories, used when no codegen CB plan fits live L1 (see kUsableL1Note).
ProgramDescriptor build_native_equivalent(
    const UntilizeCodegenOperationAttributes& operation_attributes, const Tensor& input, const Tensor& output) {
    namespace dm = ttnn::operations::data_movement;

    // Several native factories take `UntilizeTensorReturnValue&` (non-const). A Tensor is a
    // shallow, refcounted handle, so this copy aliases the same buffer the op allocated.
    Tensor out = output;

    const bool fp32_dest_acc_en = input.dtype() == DataType::INT32 || input.dtype() == DataType::UINT32 ||
                                  input.dtype() == DataType::FLOAT32;
    auto in_fmt = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(in_fmt);
    uint32_t num_tiles_per_row = input.padded_shape()[-1] / TILE_WIDTH;
    // Same derivation the native free functions do before calling their prim; note this is itself
    // a live-L1 query (get_max_l1_space), consistent with the snapshot taken in create_descriptor.
    const bool enough_space_height =
        dm::is_enough_space(input, single_tile_size, single_tile_size, num_tiles_per_row);

    const auto& logical_shape = input.logical_shape();
    const bool tile_aligned = logical_shape[-2] % TILE_HEIGHT == 0 && logical_shape[-1] % TILE_WIDTH == 0;

    if (!tile_aligned) {
        ttnn::Shape output_tensor_end(ttsl::SmallVector<uint32_t>(logical_shape.rank(), 0));
        int logical_rank = static_cast<int>(logical_shape.rank());
        for (int index = -1; index >= -logical_rank; --index) {
            output_tensor_end[index] = logical_shape[index] - 1;
        }
        UntilizeWithUnpaddingParams params{
            .output_tensor_end = output_tensor_end,
            .output_mem_config = operation_attributes.output_mem_config,
            .use_multicore = true,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .enough_space_height = enough_space_height,
            .sub_core_grids = std::nullopt};
        auto pf = UntilizeWithUnpaddingDeviceOperation::select_program_factory(params, input);
        return std::visit(
            [&](auto&& factory) { return std::decay_t<decltype(factory)>::create_descriptor(params, input, out); }, pf);
    }

    UntilizeOperationAttributes attrs{
        .output_mem_config = operation_attributes.output_mem_config,
        .use_multicore = true,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .sub_core_grids = std::nullopt,
        .enough_space_height = enough_space_height,
        // supported_by_codegen() rejects sharded output, so the sharded pf_type branches are
        // unreachable here; pass false to match what untilize_native would compute.
        .pf_type = dm::get_pf_type(/*output_is_sharded=*/false, input)};
    UntilizeTensorArgs args{.input = input};
    auto pf = UntilizeDeviceOperation::select_program_factory(attrs, args);
    return std::visit(
        [&](auto&& factory) -> ProgramDescriptor {
            using Factory = std::decay_t<decltype(factory)>;
            if constexpr (requires { Factory::create_descriptor(attrs, args, out); }) {
                return Factory::create_descriptor(attrs, args, out);
            } else {
                TT_THROW(
                    "untilize codegen: native fallback selected a program factory without descriptor support; "
                    "the live-L1 fallback must select a descriptor-backed multicore factory");
            }
        },
        pf);
}

}  // namespace

ProgramDescriptor UntilizeCodegenProgramFactory::create_descriptor(
    const UntilizeCodegenOperationAttributes& operation_attributes,
    const UntilizeCodegenTensorArgs& tensor_args,
    const Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    const Tensor& output = tensor_return_value;

    CommonArgs a{};
    a.device = input.device();
    DataType in_dtype = input.dtype();
    DataType out_dtype = output.dtype();
    a.fp32 = needs_dst_accum(in_dtype);
    a.max_bct = a.fp32 ? 4 : 8;
    a.in_fmt = tt::tt_metal::datatype_to_dataformat_converter(in_dtype);
    a.out_fmt = tt::tt_metal::datatype_to_dataformat_converter(out_dtype);
    a.in_tile_size = tt::tile_size(a.in_fmt);
    a.out_tile_size = tt::tile_size(a.out_fmt);
    a.tile_size_for_planning = std::max(a.in_tile_size, a.out_tile_size);
    a.out_elem_size = output.element_size();
    a.in_buf = input.buffer();
    a.out_buf = output.buffer();
    // kUsableL1Note: live-L1 is sampled here on a cache MISS so every builder plans against one
    // snapshot (get_max_l1_space: lowest_occupied_compute_l1_address ?: l1_size_per_core, minus
    // allocator base). The SAME chooser (choose_codegen_cb_plan) also runs on every dispatch from
    // compute_program_hash, so a later occupancy change that crosses a CB tier (or Native
    // block-split) is a new cache key rather than a hit with a frozen plan.
    //
    // get_max_l1_space is therefore on the untilize-codegen hot path (hash), not miss-only: it
    // takes the allocator mutex and walks the L1 free list per sub-device. create_descriptor
    // itself still runs only on a miss; cache hits patch addresses via apply_resolved_bindings
    // (except ENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK, OFF by default).
    a.usable_l1 = ttnn::operations::data_movement::get_max_l1_space(input);

    auto chosen = untilize_codegen_detail::choose_codegen_cb_plan(operation_attributes, tensor_args);
    if (chosen.tier == untilize_codegen_detail::CodegenCbPlan::Native) {
        return build_native_equivalent(operation_attributes, input, output);
    }

    // Wt/Ht/NC are derived from the PADDED (physical, tile-aligned) shape, which the reader/
    // compute stages need regardless of dispatch branch below (even the with-unpadding path
    // reads/untilizes the full physical tile grid; only the writer differs).
    const auto& padded_shape = input.padded_shape();
    uint32_t rank = padded_shape.rank();
    uint32_t w = padded_shape[-1];
    uint32_t h = padded_shape[-2];
    uint32_t nc = 1;
    for (uint32_t i = 0; i + 2 < rank; ++i) {
        nc *= padded_shape[i];
    }
    uint32_t wt = w / TILE_WIDTH;
    uint32_t ht = h / TILE_HEIGHT;
    uint32_t total_tile_rows = nc * ht;

    // Each branch below picks the codegen builder this shape belongs to, exactly as before. The
    // only new behaviour is that a builder may decline (nullopt) when its CB plan does not fit the
    // L1 that is free right now, in which case we emit the native-equivalent program instead of
    // failing the op. Under the opt-in ENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK build this function
    // is re-invoked on cache hits and diffed against the cached descriptor; if L1 occupancy has
    // changed enough since the miss to flip a tier, that check can report a spurious mismatch.
    // That build is a debug aid (OFF by default) and never runs in production dispatch.

    // Non-tile-aligned logical shapes (bf16 only -- see supported_by_codegen) route through the
    // with-unpadding builder instead of build_untilize_tile's variants. h == ht * TILE_HEIGHT
    // already (padded_shape is always tile-rounded), so it doubles as the physical per-batch
    // stick count the writer's pad-skip logic needs.
    const auto& logical_shape = input.logical_shape();
    bool tile_aligned = logical_shape[-2] % TILE_HEIGHT == 0 && logical_shape[-1] % TILE_WIDTH == 0;
    if (!tile_aligned) {
        if (auto desc = build_with_unpadding(a, wt, total_tile_rows, logical_shape[-2], logical_shape[-1], h)) {
            return std::move(*desc);
        }
        return build_native_equivalent(operation_attributes, input, output);
    }

    if (total_tile_rows == 1 && wt > 1) {
        if (auto desc = build_column_parallel(a, wt)) {
            return std::move(*desc);
        }
        return build_native_equivalent(operation_attributes, input, output);
    }

    auto grid = a.device->compute_with_storage_grid_size();
    uint32_t valid_cores = static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y);
    if (wt > 1) {
        uint32_t ncol = choose_2d_ncol(total_tile_rows, wt, valid_cores);
        if (ncol >= 2) {
            if (auto desc = build_2d_column(a, wt, total_tile_rows, ncol)) {
                return std::move(*desc);
            }
            return build_native_equivalent(operation_attributes, input, output);
        }
    }
    if (auto desc = build_main_split(a, wt, total_tile_rows)) {
        return std::move(*desc);
    }
    return build_native_equivalent(operation_attributes, input, output);
}

}  // namespace ttnn::prim
