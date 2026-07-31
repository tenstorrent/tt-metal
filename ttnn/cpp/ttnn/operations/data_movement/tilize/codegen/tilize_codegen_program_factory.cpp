// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_program_factory.hpp"

#include <algorithm>
#include <map>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Transliterated from tt-dm-codegen's ops/tilize/spec.py (build_tilize_row /
// build_tilize_block / build_tilize_2d_column) and ops/tilize/builder.py (the L1-aware CB /
// 2D-split math). Only the RM-interleaved, non-sharded, same-dtype paths are ported: sharded
// (build_tilize_sharded) and value-padding (build_tilize_val_padding) are out of scope for this
// device op (see tilize_codegen_supported.cpp) and TilizeCodegenParams carries no shard/pad
// fields for them.
namespace ttnn::prim {

namespace {

constexpr uint32_t kCbInId = 0;
constexpr uint32_t kCbOutId = 16;
// ops/tilize/spec.py _DEFAULT_WRITE_BATCH
constexpr uint32_t kDefaultWriteBatch = 4;
// reader_stick_interleaved_unified.cpp MODE_TILEROW
constexpr uint32_t kModeTilerow = 1;
// ops/tilize/builder.py _BLOCK_THRESHOLD
constexpr uint32_t kBlockThreshold = 32;

constexpr const char* kReaderStickUnified =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels/reader_stick_interleaved_unified.cpp";
constexpr const char* kWriterTilizeInterleaved =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels/writer_tilize_interleaved.cpp";
constexpr const char* kReaderTilizeBlock =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels/reader_tilize_block.cpp";
constexpr const char* kWriterTilizeBlock =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels/writer_tilize_block.cpp";
constexpr const char* kComputeTilize =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/codegen/kernels/compute_tilize.cpp";

uint32_t align_up(uint32_t value, uint32_t alignment) { return ((value + alignment - 1) / alignment) * alignment; }

// The page pitches below (input stick, output tile) are rounded with the DRAM alignment for both
// placements, as ops/tilize/spec.py does. That is exact for an interleaved-L1 buffer too, not just
// a lucky over-estimate: supported_by_codegen admits only W % TILE_W == 0 with a >= 2-byte dtype, so
// a stick is a multiple of 64 bytes and a tile a multiple of 32 — already aligned under the L1
// alignment (the smaller of the two) as well, leaving the rounding a no-op either way.

// ops/tilize/builder.py _largest_divisor_le
uint32_t largest_divisor_le(uint32_t n, uint32_t limit) {
    if (n <= limit) {
        return n;
    }
    uint32_t best = 1;
    for (uint32_t d = 1; d * d <= n; ++d) {
        if (n % d == 0) {
            if (d <= limit) {
                best = std::max(best, d);
            }
            uint32_t q = n / d;
            if (q <= limit) {
                best = std::max(best, q);
            }
        }
    }
    return best;
}

// ops/tilize/builder.py _compute_cb_block_limit
uint32_t compute_cb_block_limit(uint32_t tile_size_bytes, uint32_t l1_size) { return l1_size / (2 * tile_size_bytes); }

// ops/tilize/spec.py _tilize_required_cb_out.
// The batched writer holds the previous batch un-popped while waiting on the next, so both must be
// resident at once. Compute only ever adds pages in whole tilize_block-sized groups of
// compute_chunk, so a capacity that is not a whole number of those groups is unreachable: the
// packer blocks reserving the group that would not fit while the writer blocks waiting for pages
// only that group can supply.
uint32_t required_cb_out(uint32_t write_batch, uint32_t compute_chunk) {
    if (write_batch <= 1) {
        return compute_chunk;
    }
    return align_up(2 * write_batch, compute_chunk);
}

// UnpackToDestMode vector forcing a FLOAT32 input CB to unpack straight into DEST (avoids the
// SRCA Float16_b truncation for RM fp32 tilize input) — mirrors builder_utils.unpack_to_dest_fp32_modes.
// Length is the host-side CB-count constant the native tilize factory uses for the same vector.
std::vector<UnpackToDestMode> unpack_to_dest_fp32_modes(uint32_t cb_index) {
    std::vector<UnpackToDestMode> modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    modes[cb_index] = UnpackToDestMode::UnpackToDestFp32;
    return modes;
}

struct Geometry {
    uint32_t nc = 0;
    uint32_t ht = 0;
    uint32_t wt = 0;
    uint32_t total_ht = 0;
    uint32_t h = 0;
    uint32_t w = 0;
    uint32_t elem_size = 0;
    uint32_t ts_in = 0;
    uint32_t ts_out = 0;
};

Geometry compute_geometry(const TilizeCodegenParams& attrs, const Tensor& input_tensor) {
    Geometry g;
    g.nc = attrs.NC;
    g.ht = attrs.Ht;
    g.wt = attrs.Wt;
    g.total_ht = g.nc * g.ht;
    g.h = g.ht * constants::TILE_HEIGHT;
    g.w = g.wt * constants::TILE_WIDTH;
    g.elem_size = input_tensor.element_size();
    g.ts_in = tile_size(datatype_to_dataformat_converter(attrs.input_dtype));
    g.ts_out = tile_size(datatype_to_dataformat_converter(attrs.output_dtype));
    return g;
}

// ---------------------------------------------------------------------------
// 2D block-split geometry (ops/tilize/builder.py _compute_2d_split)
// ---------------------------------------------------------------------------
struct Split2D {
    uint32_t block_wt = 0;
    uint32_t block_ht = 0;
    uint32_t cores_w = 0;
    uint32_t cores_h = 0;
    uint32_t cliff_wt = 0;
    uint32_t cliff_ht = 0;
};

Split2D compute_2d_split(uint32_t ht, uint32_t wt, uint32_t num_avail_cores, uint32_t cb_block_limit) {
    const uint32_t total_tiles = ht * wt;
    if (total_tiles <= cb_block_limit && num_avail_cores >= 1) {
        return {wt, ht, 1, 1, 0, 0};
    }

    const uint32_t max_block_wt = std::min(wt, cb_block_limit);
    bool have_best = false;
    uint32_t best_bw = 0, best_bh = 0, best_cw = 0, best_ch = 0, best_ncores = 0;
    const uint32_t upper = std::min(num_avail_cores, wt);
    for (uint32_t target_cw = 1; target_cw <= upper; ++target_cw) {
        const uint32_t bw = (wt + target_cw - 1) / target_cw;
        if (bw > max_block_wt) {
            continue;
        }
        const uint32_t cw = (wt + bw - 1) / bw;
        if (cw == 0) {
            continue;
        }
        const uint32_t max_ch = num_avail_cores / cw;
        if (max_ch < 1) {
            continue;
        }
        const uint32_t ch_cand = std::min(ht, max_ch);
        const uint32_t bh = (ht + ch_cand - 1) / ch_cand;
        const uint32_t ch = (ht + bh - 1) / bh;
        const uint32_t nc = cw * ch;
        if (nc > num_avail_cores) {
            continue;
        }
        if (nc > best_ncores || (nc == best_ncores && have_best && bw * bh < best_bw * best_bh)) {
            best_bw = bw;
            best_bh = bh;
            best_cw = cw;
            best_ch = ch;
            best_ncores = nc;
            have_best = true;
        }
    }
    if (!have_best) {
        return {wt, ht, 1, 1, 0, 0};
    }
    const uint32_t cliff_wt = wt - (wt / best_bw) * best_bw;
    const uint32_t cliff_ht = ht - (ht / best_bh) * best_bh;
    return {best_bw, best_bh, best_cw, best_ch, cliff_wt, cliff_ht};
}

// ops/tilize/spec.py uses_block_path
bool uses_block_path(
    uint32_t total_ht, uint32_t wt, uint32_t num_avail_cores, uint32_t cb_block_limit, uint32_t ts_in, uint32_t l1) {
    if (!(wt > kBlockThreshold && total_ht < num_avail_cores)) {
        return false;
    }
    const Split2D split = compute_2d_split(total_ht, wt, num_avail_cores, cb_block_limit);
    if (std::max(split.block_ht, split.cliff_ht) > 1) {
        const bool row_needs_chunking = 2 * wt * ts_in > l1;
        return row_needs_chunking && total_ht < num_avail_cores && (num_avail_cores / total_ht) >= 2;
    }
    return (split.cores_w * split.cores_h) > total_ht;
}

// ops/tilize/spec.py _choose_tilize_2d_ncol
uint32_t choose_tilize_2d_ncol(uint32_t total_ht, uint32_t wt, uint32_t valid_cores) {
    if (total_ht >= valid_cores || wt < 2) {
        return 1;
    }
    const uint32_t max_ncol = std::min(valid_cores / total_ht, wt);
    uint32_t best = 1;
    for (uint32_t d = 2; d <= max_ncol; ++d) {
        if (wt % d == 0) {
            best = d;
        }
    }
    return best;
}

// ops/tilize/spec.py uses_2d_column_path
uint32_t uses_2d_column_path(
    uint32_t total_ht, uint32_t wt, uint32_t num_avail_cores, uint32_t cb_block_limit, uint32_t ts_in, uint32_t l1) {
    if (uses_block_path(total_ht, wt, num_avail_cores, cb_block_limit, ts_in, l1)) {
        return 1;
    }
    if (wt <= 2) {
        return 1;
    }
    return choose_tilize_2d_ncol(total_ht, wt, num_avail_cores);
}

// ---------------------------------------------------------------------------
// build_tilize_row (ops/tilize/spec.py build_tilize_row)
// ---------------------------------------------------------------------------
ProgramDescriptor build_row(
    IDevice* device,
    const TilizeCodegenParams& attrs,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const Geometry& g,
    bool single_core) {
    const uint32_t dram_alignment = hal::get_dram_alignment();
    const uint32_t l1 = device->l1_size_per_core();
    const uint32_t stick_size_bytes = g.w * g.elem_size;
    const uint32_t aligned_ps = align_up(stick_size_bytes, dram_alignment);

    uint32_t cb_depth, chunk_wt, num_col_chunks;
    if (attrs.use_low_perf) {
        // ttnn's low-performance route: single-core, one-tile block, minimal CB footprint.
        cb_depth = 1;
        chunk_wt = 1;
        num_col_chunks = g.wt;
    } else if (2 * 2 * g.wt * g.ts_in <= l1) {
        cb_depth = 2;
        chunk_wt = g.wt;
        num_col_chunks = 1;
    } else if (2 * g.wt * g.ts_in <= l1) {
        cb_depth = 1;
        chunk_wt = g.wt;
        num_col_chunks = 1;
    } else {
        const uint32_t max_chunk = l1 / (2 * g.ts_in);
        chunk_wt = largest_divisor_le(g.wt, max_chunk);
        cb_depth = (2 * 2 * chunk_wt * g.ts_in <= l1) ? 2 : 1;
        num_col_chunks = g.wt / chunk_wt;
    }

    CoreRangeSet all_cores;
    uint32_t num_cores;
    std::vector<CoreCoord> cores;
    CoreRangeSet group1, group2;
    uint32_t per1 = 0, per2 = 0;
    if (single_core) {
        const CoreCoord c{0, 0};
        cores = {c};
        all_cores = CoreRangeSet(CoreRange(c, c));
        num_cores = 1;
        group1 = all_cores;
        per1 = g.total_ht;
        per2 = 0;
    } else {
        const CoreCoord grid = device->compute_with_storage_grid_size();
        std::tie(num_cores, all_cores, group1, group2, per1, per2) = split_work_to_cores(grid, g.total_ht);
        // Derive the per-core iteration list from the CoreRangeSet the kernels are actually
        // created over (all_cores), never from a second, independent enumeration of the raw grid:
        // split_work_to_cores's all_cores is column-major by default, so a row-major grid_to_cores
        // list can name a logical core outside all_cores' bounding box, indexing past the end of
        // Kernel::core_to_runtime_args_ (sized from that bounding box) and segfaulting.
        cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    }

    // MEMORY: writer_tilize_interleaved.cpp's batched branch walks CB pages linearly from the
    // read pointer, which only matches the output tile ids when a core owns ONE tile-row; with
    // several rows per core it interleaves them wrong. Force the single-write branch there.
    // A one-tile-per-core assignment has nothing to overlap either, so batch priming and
    // double-buffering would be pure per-core setup cost — spec.py's `minimal_work` clamp.
    const bool minimal_work = (num_col_chunks == 1 && chunk_wt == 1 && g.total_ht <= num_cores);
    if (minimal_work) {
        cb_depth = 1;
    }
    const bool force_single_write = attrs.use_low_perf || (g.total_ht > num_cores) || minimal_work;
    const uint32_t write_batch = force_single_write ? 1 : kDefaultWriteBatch;

    uint32_t cb_in_depth = cb_depth * chunk_wt;
    uint32_t cb_out_depth = std::max(cb_depth * chunk_wt, required_cb_out(write_batch, chunk_wt));
    if (minimal_work) {
        cb_out_depth = 1;
    }

    const uint32_t out_page = align_up(g.ts_out, dram_alignment);
    const uint32_t elem_w_bytes = constants::TILE_WIDTH * g.elem_size;

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_depth * g.ts_in,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbInId,
            .data_format = datatype_to_dataformat_converter(attrs.input_dtype),
            .page_size = g.ts_in,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_out_depth * g.ts_out,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbOutId,
            .data_format = datatype_to_dataformat_converter(attrs.output_dtype),
            .page_size = g.ts_out,
        }}},
    });

    // reader_stick_interleaved_unified.cpp is a single shared template compiled for every mode
    // that uses it; the MODE_TILEROW_PAD named CT args below are dead code under MODE_TILEROW
    // but must still be defined or JIT compilation fails ("Invalid named compile time argument").
    // Values are builder_utils.py's own injected defaults (_TILEROW_PAD_DEFAULTS), not invented.
    KernelDescriptor::NamedCompileTimeArgs reader_named_ct = {
        {"mode", kModeTilerow},
        {"cb_id", kCbInId},
        {"stick_bytes", stick_size_bytes},
        {"aligned_page_size", aligned_ps},
        {"seq_id", 0},
        {"batch", 1},
        {"nabatch", 4},
        {"elem_size", 2},
        {"tile_height", 32},
        {"tile_row_shift_bits", 0},
        {"num_pages_in_row", 1},
        {"unpadded_X_bytes", 0},
        {"valid_last_page_bytes", 0},
        {"page_size", 32},
    };

    KernelDescriptor::CompileTimeArgs reader_ct;
    TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderStickUnified;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.named_compile_time_args = std::move(reader_named_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct{kCbOutId, out_page};
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);
    writer_ct.push_back(write_batch);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterTilizeInterleaved;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    const bool fp32 = (attrs.input_dtype == DataType::FLOAT32 || attrs.output_dtype == DataType::FLOAT32);
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kComputeTilize;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {kCbInId, kCbOutId, num_col_chunks, chunk_wt};
    ComputeConfigDescriptor compute_cfg;
    compute_cfg.fp32_dest_acc_en = fp32;
    if (attrs.input_dtype == DataType::FLOAT32) {
        compute_cfg.unpack_to_dest_mode = unpack_to_dest_fp32_modes(kCbInId);
    }
    compute_desc.config = compute_cfg;

    uint32_t start = 0;
    for (const auto& core : cores) {
        const uint32_t n = group1.contains(core) ? per1 : per2;
        reader_desc.emplace_runtime_args(
            core,
            {input_tensor.buffer(),
             n,
             start * constants::TILE_HEIGHT,
             constants::TILE_HEIGHT,
             chunk_wt,
             num_col_chunks,
             elem_w_bytes});
        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{n});
        writer_desc.emplace_runtime_args(core, {output_tensor.buffer(), n * g.wt, start * g.wt});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

// ---------------------------------------------------------------------------
// build_tilize_2d_column (ops/tilize/spec.py build_tilize_2d_column)
// ---------------------------------------------------------------------------
ProgramDescriptor build_2d_column(
    IDevice* device,
    const TilizeCodegenParams& attrs,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const Geometry& g,
    uint32_t ncol) {
    TT_FATAL(
        ncol >= 2 && g.wt % ncol == 0, "tilize codegen 2D ncol={} must be a divisor of Wt={} and >= 2", ncol, g.wt);

    const uint32_t dram_alignment = hal::get_dram_alignment();
    const uint32_t l1 = device->l1_size_per_core();
    const uint32_t tpc = g.wt / ncol;
    const uint32_t num_cores = g.total_ht * ncol;

    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t core_budget = grid.x * grid.y;
    TT_FATAL(
        num_cores <= core_budget,
        "tilize codegen 2D geometry needs {} cores but the device offers {}",
        num_cores,
        core_budget);

    const uint32_t stick_size_bytes = g.w * g.elem_size;
    const uint32_t aligned_ps = align_up(stick_size_bytes, dram_alignment);

    uint32_t cb_depth = (2 * 2 * tpc * g.ts_in <= l1) ? 2 : 1;
    const bool minimal_work = (tpc == 1);
    if (minimal_work) {
        cb_depth = 1;
    }
    const uint32_t write_batch = minimal_work ? 1 : kDefaultWriteBatch;

    // sub_wt is the packer's reservation granularity, so it has to be known before the output depth.
    const uint32_t cb_block_limit = compute_cb_block_limit(g.ts_in, l1);
    const uint32_t sub_wt = (tpc <= cb_block_limit) ? tpc : largest_divisor_le(tpc, cb_block_limit);
    const uint32_t n_sub = (sub_wt == tpc) ? 1 : (tpc / sub_wt);

    uint32_t cb_in_depth = cb_depth * tpc;
    uint32_t cb_out_depth = std::max(cb_depth * tpc, required_cb_out(write_batch, sub_wt));
    if (minimal_work) {
        cb_out_depth = 1;
    }

    const uint32_t out_page = align_up(g.ts_out, dram_alignment);
    const uint32_t elem_w_bytes = constants::TILE_WIDTH * g.elem_size;

    const std::vector<CoreCoord> cores = grid_to_cores(num_cores, grid.x, grid.y, /*row_wise=*/true);
    std::vector<CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& c : cores) {
        ranges.emplace_back(c, c);
    }
    const CoreRangeSet core_grid(ranges);

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_depth * g.ts_in,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbInId,
            .data_format = datatype_to_dataformat_converter(attrs.input_dtype),
            .page_size = g.ts_in,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_out_depth * g.ts_out,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbOutId,
            .data_format = datatype_to_dataformat_converter(attrs.output_dtype),
            .page_size = g.ts_out,
        }}},
    });

    KernelDescriptor::CompileTimeArgs reader_ct{n_sub, constants::TILE_HEIGHT, sub_wt, elem_w_bytes, aligned_ps};
    TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_ct);
    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderTilizeBlock;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct{kCbOutId, out_page};
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);
    writer_ct.push_back(write_batch);
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterTilizeBlock;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    const bool fp32 = (attrs.input_dtype == DataType::FLOAT32);
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kComputeTilize;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = {kCbInId, kCbOutId, n_sub, sub_wt};
    ComputeConfigDescriptor compute_cfg;
    compute_cfg.fp32_dest_acc_en = fp32;
    if (fp32) {
        compute_cfg.unpack_to_dest_mode = unpack_to_dest_fp32_modes(kCbInId);
    }
    compute_desc.config = compute_cfg;

    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        const uint32_t tile_row = i / ncol;
        const uint32_t col_block = i % ncol;
        const uint32_t start_stick = tile_row * constants::TILE_HEIGHT;
        const uint32_t start_tc = col_block * tpc;
        const uint32_t col_byte_offset = start_tc * constants::TILE_WIDTH * g.elem_size;

        reader_desc.emplace_runtime_args(core, {input_tensor.buffer(), 1u, start_stick, col_byte_offset});
        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{1});
        const uint32_t start_tile = tile_row * g.wt + start_tc;
        writer_desc.emplace_runtime_args(core, {output_tensor.buffer(), tpc, start_tile, tpc, g.wt});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

// ---------------------------------------------------------------------------
// build_tilize_block (ops/tilize/spec.py build_tilize_block)
// ---------------------------------------------------------------------------
struct BlockAssignment {
    CoreCoord core;
    uint32_t this_ht = 0;
    uint32_t this_wt = 0;
    uint32_t start_tr = 0;
    uint32_t start_tc = 0;
};

ProgramDescriptor build_block(
    IDevice* device,
    const TilizeCodegenParams& attrs,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const Geometry& g) {
    const uint32_t dram_alignment = hal::get_dram_alignment();
    const uint32_t l1 = device->l1_size_per_core();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_avail_cores = grid.x * grid.y;
    const uint32_t cb_block_limit = compute_cb_block_limit(g.ts_in, l1);

    Split2D split = compute_2d_split(g.total_ht, g.wt, num_avail_cores, cb_block_limit);
    uint32_t block_wt = split.block_wt, block_ht = split.block_ht;
    uint32_t cores_w = split.cores_w, cores_h = split.cores_h;
    uint32_t cliff_wt = split.cliff_wt, cliff_ht = split.cliff_ht;

    // Keep one tile-row per core in the wide-short regime instead of the generic splitter's
    // tall-block + width-cliff combination (known-wrong ordering — see uses_block_path).
    if (std::max(block_ht, cliff_ht) > 1 && g.total_ht < num_avail_cores) {
        const uint32_t forced_cols = std::min(num_avail_cores / g.total_ht, g.wt);
        if (forced_cols >= 2) {
            cores_h = g.total_ht;
            block_ht = 1;
            cliff_ht = 0;
            cores_w = forced_cols;
            block_wt = (g.wt + cores_w - 1) / cores_w;
            cliff_wt = g.wt - block_wt * (cores_w - 1);
        }
    }

    const bool has_cliff_w = cliff_wt > 0;
    const bool has_cliff_h = cliff_ht > 0;
    // MEMORY: the batched writer_tilize_block.cpp is correct only for ONE tile-row per core.
    const bool multirow_core = std::max(block_ht, cliff_ht) > 1;
    const uint32_t write_batch = multirow_core ? 1 : kDefaultWriteBatch;

    const uint32_t total_cores = cores_w * cores_h;
    const std::vector<CoreCoord> valid_cores = grid_to_cores(total_cores, grid.x, grid.y, /*row_wise=*/true);

    std::map<uint32_t, std::vector<BlockAssignment>> groups;  // keyed by this_wt (block width)
    uint32_t core_idx = 0;
    for (uint32_t row_idx = 0; row_idx < cores_h; ++row_idx) {
        const bool is_cliff_h = has_cliff_h && row_idx == cores_h - 1;
        const uint32_t this_ht = is_cliff_h ? cliff_ht : block_ht;
        const uint32_t start_tile_row = row_idx * block_ht;
        for (uint32_t col_idx = 0; col_idx < cores_w; ++col_idx) {
            const bool is_cliff_w = has_cliff_w && col_idx == cores_w - 1;
            const uint32_t this_wt = is_cliff_w ? cliff_wt : block_wt;
            const uint32_t start_tile_col = col_idx * block_wt;
            groups[this_wt].push_back({valid_cores[core_idx], this_ht, this_wt, start_tile_row, start_tile_col});
            ++core_idx;
        }
    }

    uint32_t max_bw = 0;
    uint32_t max_compute_chunk = 0;
    for (const auto& [w, _] : groups) {
        max_bw = std::max(max_bw, w);
        max_compute_chunk =
            std::max(max_compute_chunk, (w <= cb_block_limit) ? w : largest_divisor_le(w, cb_block_limit));
    }
    const uint32_t cb_depth = (2 * 2 * max_bw * g.ts_in <= l1) ? 2 : 1;
    const uint32_t cb_in_depth = cb_depth * max_bw;
    const uint32_t cb_out_depth = std::max(cb_depth * max_bw, required_cb_out(write_batch, max_compute_chunk));

    const uint32_t stick_size_bytes = g.w * g.elem_size;
    const uint32_t aligned_ps = align_up(stick_size_bytes, dram_alignment);
    const uint32_t out_page = align_up(g.ts_out, dram_alignment);
    const uint32_t elem_w_bytes = constants::TILE_WIDTH * g.elem_size;

    std::vector<CoreRange> all_ranges;
    all_ranges.reserve(valid_cores.size());
    for (const auto& c : valid_cores) {
        all_ranges.emplace_back(c, c);
    }
    const CoreRangeSet all_core_grid(all_ranges);

    ProgramDescriptor desc;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_depth * g.ts_in,
        .core_ranges = all_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbInId,
            .data_format = datatype_to_dataformat_converter(attrs.input_dtype),
            .page_size = g.ts_in,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_out_depth * g.ts_out,
        .core_ranges = all_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbOutId,
            .data_format = datatype_to_dataformat_converter(attrs.output_dtype),
            .page_size = g.ts_out,
        }}},
    });

    const bool fp32 = (attrs.input_dtype == DataType::FLOAT32);

    for (const auto& [grp_wt, assignments] : groups) {
        std::vector<CoreRange> grp_ranges;
        grp_ranges.reserve(assignments.size());
        for (const auto& a : assignments) {
            grp_ranges.emplace_back(a.core, a.core);
        }
        const CoreRangeSet grp_grid(grp_ranges);

        const uint32_t sub_wt = (grp_wt <= cb_block_limit) ? grp_wt : largest_divisor_le(grp_wt, cb_block_limit);
        const uint32_t n_sub = (sub_wt == grp_wt) ? 1 : (grp_wt / sub_wt);

        KernelDescriptor::CompileTimeArgs reader_ct{n_sub, constants::TILE_HEIGHT, sub_wt, elem_w_bytes, aligned_ps};
        TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_ct);
        KernelDescriptor reader_desc;
        reader_desc.kernel_source = kReaderTilizeBlock;
        reader_desc.core_ranges = grp_grid;
        reader_desc.compile_time_args = std::move(reader_ct);
        reader_desc.config = ReaderConfigDescriptor{};

        KernelDescriptor::CompileTimeArgs writer_ct{kCbOutId, out_page};
        TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);
        writer_ct.push_back(write_batch);
        KernelDescriptor writer_desc;
        writer_desc.kernel_source = kWriterTilizeBlock;
        writer_desc.core_ranges = grp_grid;
        writer_desc.compile_time_args = std::move(writer_ct);
        writer_desc.config = WriterConfigDescriptor{};

        KernelDescriptor compute_desc;
        compute_desc.kernel_source = kComputeTilize;
        compute_desc.core_ranges = grp_grid;
        compute_desc.compile_time_args = {kCbInId, kCbOutId, n_sub, sub_wt};
        ComputeConfigDescriptor compute_cfg;
        compute_cfg.fp32_dest_acc_en = fp32;
        if (fp32) {
            compute_cfg.unpack_to_dest_mode = unpack_to_dest_fp32_modes(kCbInId);
        }
        compute_desc.config = compute_cfg;

        for (const auto& a : assignments) {
            const uint32_t start_stick = a.start_tr * constants::TILE_HEIGHT;
            const uint32_t col_byte_offset = a.start_tc * constants::TILE_WIDTH * g.elem_size;
            reader_desc.emplace_runtime_args(a.core, {input_tensor.buffer(), a.this_ht, start_stick, col_byte_offset});
            compute_desc.runtime_args.emplace_back(a.core, KernelDescriptor::CoreRuntimeArgs{a.this_ht});
            const uint32_t num_tiles = a.this_ht * a.this_wt;
            const uint32_t start_tile = a.start_tr * g.wt + a.start_tc;
            writer_desc.emplace_runtime_args(a.core, {output_tensor.buffer(), num_tiles, start_tile, a.this_wt, g.wt});
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        desc.kernels.push_back(std::move(compute_desc));
    }

    return desc;
}

}  // namespace

ProgramDescriptor TilizeCodegenProgramFactory::create_descriptor(
    const TilizeCodegenParams& operation_attributes,
    const TilizeCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    IDevice* device = tensor_return_value.device();
    const Tensor& input_tensor = tensor_args.input_tensor;
    Tensor& output_tensor = tensor_return_value;

    const Geometry g = compute_geometry(operation_attributes, input_tensor);

    // ops/tilize/tilize.py: route_single = (not use_multicore) or use_low_perf — both force the
    // single-core row path regardless of Wt/Ht, ahead of the block/2D-column dispatch decision.
    const bool route_single = !operation_attributes.use_multicore || operation_attributes.use_low_perf;
    if (route_single) {
        return build_row(device, operation_attributes, input_tensor, output_tensor, g, /*single_core=*/true);
    }

    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_avail_cores = grid.x * grid.y;
    const uint32_t l1 = device->l1_size_per_core();
    const uint32_t cb_block_limit = compute_cb_block_limit(g.ts_in, l1);

    if (uses_block_path(g.total_ht, g.wt, num_avail_cores, cb_block_limit, g.ts_in, l1)) {
        return build_block(device, operation_attributes, input_tensor, output_tensor, g);
    }
    const uint32_t ncol = uses_2d_column_path(g.total_ht, g.wt, num_avail_cores, cb_block_limit, g.ts_in, l1);
    if (ncol >= 2) {
        return build_2d_column(device, operation_attributes, input_tensor, output_tensor, g, ncol);
    }
    return build_row(device, operation_attributes, input_tensor, output_tensor, g, /*single_core=*/false);
}

}  // namespace ttnn::prim
