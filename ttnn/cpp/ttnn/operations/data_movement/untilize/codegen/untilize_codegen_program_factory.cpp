// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_program_factory.hpp"

#include <algorithm>
#include <variant>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "untilize_codegen_device_operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Kernels were copied verbatim from the codegen builder into codegen/kernels/ in phase 3.
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/data_movement/untilize/codegen/kernels";
constexpr uint32_t kCbIn = tt::CBIndex::c_0;
constexpr uint32_t kCbOut = tt::CBIndex::c_16;
constexpr uint32_t kSeqIdentity = 0;       // mirrors common/templates/sequencers.h SEQ_IDENTITY
constexpr uint64_t kUsableL1 = 1'400'000;  // mirrors codegen builder_utils.USABLE_L1

using RtArg = std::variant<uint32_t, Buffer*>;
using RtArgs = std::vector<RtArg>;

std::string kernel_path(const char* name) { return std::string(kKernelDir) + "/" + name; }

// Mirrors spec.py's _needs_dst_accum: 32-bit datums need 32-bit DEST accumulation in
// pack_untilize. Always false for this port's supported_by_codegen scope (bf16/bf8_b only);
// kept general to stay a faithful transliteration of the source builder.
bool needs_dst_accum(DataType dtype) {
    return dtype == DataType::FLOAT32 || dtype == DataType::INT32 || dtype == DataType::UINT32;
}

// Mirrors compute_untilize.cpp's compute_num_blocks_per_column: the largest bct <= max_bct
// that evenly divides wt. The host must replicate this to size CB depths in the same units
// the kernel will actually consume per pack_untilize_block call.
uint32_t compute_block_ct_dim(uint32_t wt, bool fp32) {
    uint32_t max_bct = fp32 ? 4 : 8;
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (wt % bct == 0) {
            return bct;
        }
    }
    return 1;
}

struct CbPlan {
    uint32_t cb_in_depth;
    uint32_t cb_out_depth;
    uint32_t read_batch;
};

// Mirrors codegen_common.factory.cb_policy.plan_cb_depths: 4-tier asymmetric CB depth
// selection (double-buffer both -> double-buffer input only -> single-buffer both ->
// chunked fallback), budgeted against a fixed 1.4MB usable-L1 constant (matches source).
CbPlan plan_cb_depths(uint32_t pages_per_unit, uint32_t page_size, uint32_t block_units) {
    uint64_t p = pages_per_unit;
    uint64_t ts = page_size;
    uint64_t double_both = (2 * p + 2 * p) * ts;
    uint64_t double_in = (2 * p + p) * ts;
    uint64_t single_both = (p + p) * ts;
    if (double_both <= kUsableL1) {
        return CbPlan{static_cast<uint32_t>(2 * p), static_cast<uint32_t>(2 * p), pages_per_unit};
    }
    if (double_in <= kUsableL1) {
        return CbPlan{static_cast<uint32_t>(2 * p), pages_per_unit, pages_per_unit};
    }
    if (single_both <= kUsableL1) {
        return CbPlan{pages_per_unit, pages_per_unit, block_units};
    }
    return CbPlan{std::max(pages_per_unit, block_units), pages_per_unit, block_units};
}

// Mirrors spec.py's _choose_2d_ncol: largest divisor of wt (>=2) so that every tile-row x
// column-block unit still gets its own core; returns 1 ("don't use the 2D path") otherwise.
uint32_t choose_2d_ncol(uint32_t total_tile_rows, uint32_t wt, uint32_t valid_cores) {
    if (total_tile_rows >= valid_cores || wt < 2) {
        return 1;
    }
    uint32_t max_ncol = std::min(valid_cores / total_tile_rows, wt);
    uint32_t best = 1;
    for (uint32_t d = 2; d <= max_ncol; ++d) {
        if (wt % d == 0) {
            best = d;
        }
    }
    return best;
}

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
};

// Per-tile-row split (build_untilize_tile's default path / _build_untilize_tile_cliff).
// Splits total_tile_rows across up to the device's core grid; an uneven split produces a
// second ("cliff") compute-kernel core group with its own per-core tile-row count.
ProgramDescriptor build_main_split(const CommonArgs& a, uint32_t wt, uint32_t total_tile_rows) {
    auto grid = a.device->compute_with_storage_grid_size();
    auto split = tt::tt_metal::split_work_to_cores(grid, total_tile_rows, /*row_wise=*/true);
    const CoreRangeSet& core_range = std::get<1>(split);
    const CoreRangeSet& cg1 = std::get<2>(split);
    const CoreRangeSet& cg2 = std::get<3>(split);
    uint32_t tpc1 = std::get<4>(split);
    uint32_t tpc2 = std::get<5>(split);

    uint32_t block_ct_dim = compute_block_ct_dim(wt, a.fp32);
    CbPlan plan = plan_cb_depths(wt, a.tile_size_for_planning, block_ct_dim);

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
            reader.emplace_runtime_args(core, RtArgs{a.in_buf, n * wt, assigned * wt});
            writer.emplace_runtime_args(core, RtArgs{a.out_buf, n, assigned * TILE_HEIGHT, row_size_bytes, 0u, 0u, 0u});
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
ProgramDescriptor build_column_parallel(const CommonArgs& a, uint32_t wt) {
    auto grid = a.device->compute_with_storage_grid_size();
    auto split = tt::tt_metal::split_work_to_cores(grid, wt, /*row_wise=*/true);
    const CoreRangeSet& core_range = std::get<1>(split);
    const CoreRangeSet& cg1 = std::get<2>(split);
    const CoreRangeSet& cg2 = std::get<3>(split);
    uint32_t tpc1 = std::get<4>(split);
    uint32_t tpc2 = std::get<5>(split);

    uint32_t max_tpc = std::max(tpc1, cg2.empty() ? 0u : tpc2);
    uint32_t block_ct_dim = compute_block_ct_dim(max_tpc, a.fp32);
    CbPlan plan = plan_cb_depths(max_tpc, a.tile_size_for_planning, block_ct_dim);

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
            reader.emplace_runtime_args(core, RtArgs{a.in_buf, n, assigned});
            writer.emplace_runtime_args(
                core,
                RtArgs{
                    a.out_buf,
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
ProgramDescriptor build_2d_column(const CommonArgs& a, uint32_t wt, uint32_t total_tile_rows, uint32_t ncol) {
    uint32_t tpc = wt / ncol;
    uint32_t num_units = total_tile_rows * ncol;

    auto grid = a.device->compute_with_storage_grid_size();
    auto split = tt::tt_metal::split_work_to_cores(grid, num_units, /*row_wise=*/true);
    const CoreRangeSet& core_range = std::get<1>(split);

    uint32_t block_ct_dim = compute_block_ct_dim(tpc, a.fp32);
    CbPlan plan = plan_cb_depths(tpc, a.tile_size_for_planning, block_ct_dim);

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
        reader.emplace_runtime_args(core, RtArgs{a.in_buf, tpc, start_tile});
        writer.emplace_runtime_args(
            core,
            RtArgs{a.out_buf, TILE_HEIGHT, col_block * col_chunk_bytes, col_chunk_bytes, tpc, tile_row * TILE_HEIGHT});
    }

    // Kernel order [reader, writer, compute] mirrors _build_untilize_2d_column's kernels list.
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
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

    // Wt/Ht/NC are derived from the PADDED (physical, tile-aligned) shape rather than the
    // logical shape, so this factory also handles non-tile-aligned logical shapes correctly:
    // the padding rows/columns are preserved in the physical output buffer (matching
    // compute_output_specs below and native UntilizeDeviceOperation's own scheme), and are
    // simply cropped by the output tensor's logical_shape metadata on read.
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

    (void)operation_attributes;

    if (total_tile_rows == 1 && wt > 1) {
        return build_column_parallel(a, wt);
    }

    auto grid = a.device->compute_with_storage_grid_size();
    uint32_t valid_cores = static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y);
    if (wt > 1) {
        uint32_t ncol = choose_2d_ncol(total_tile_rows, wt, valid_cores);
        if (ncol >= 2) {
            return build_2d_column(a, wt, total_tile_rows, ncol);
        }
    }
    return build_main_split(a, wt, total_tile_rows);
}

}  // namespace ttnn::prim
