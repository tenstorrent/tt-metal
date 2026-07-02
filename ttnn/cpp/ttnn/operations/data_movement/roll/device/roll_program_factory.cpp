// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "roll_program_factory.hpp"

#include <algorithm>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

// Why ROW_MAJOR sharded roll needs a dedicated kernel instead of the slice + concat composite
// used for interleaved roll:
//
//   * slice cuts a tensor into two unequal halves (e.g. widths W-shift and shift). Re-sharding
//     those odd-shaped halves with the original shard spec produces invalid shards — the shard
//     dims no longer divide the sliced shape ("page size must equal buffer size").
//   * concat's sharded program factories only accept specific (axis, layout) pairs (width concat
//     on height/block-sharded, height concat on width/block-sharded, ...). The slice→concat
//     decomposition of an arbitrary roll does not fit those combinations across HEIGHT/WIDTH/
//     BLOCK sharding and every dim.
//
// So a roll on a sharded tensor cannot be expressed as composite slice + concat for the general
// case. This factory instead performs the roll as a single per-core gather directly over the
// sharded buffers: it stays in one shard spec (no intermediate slices) and is not bound by
// slice's or concat's sharded constraints.

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

struct RollTransferDesc {
    // L1 mode: source identified by (src_physical_core, src_l1_offset).
    // DRAM mode: source identified by (src_dram_shard_idx, src_l1_offset used as intra-shard offset).
    CoreCoord src_physical_core;  // L1 mode only; zeroed in DRAM mode
    uint32_t src_dram_shard_idx;  // DRAM mode only; the linear shard index of the source
    uint32_t src_l1_offset;
    uint32_t dst_offset;
    uint32_t copy_size;
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t num_rows;
};

// A contiguous cell-column run: dst cells [dst_col, dst_col+len) <- src cells [src_col, ...).
// The same column structure repeats for every cell-row, so it is computed once.
struct ColPiece {
    uint32_t dst_col;
    uint32_t src_col;
    uint32_t len;
};

}  // namespace

ProgramDescriptor RollShardedProgramFactory::create_descriptor(
    const RollParams& operation_attributes, const RollInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;

    TT_FATAL(input.is_sharded() && output.is_sharded(), "Native sharded roll requires sharded input and output");

    const bool is_dram = input.memory_config().buffer_type() == BufferType::DRAM;
    const bool is_dram_rm = is_dram && input.layout() == Layout::ROW_MAJOR;
    // DRAM-RM mode uses full-shard L1 staging: read whole source shard(s) from DRAM into L1,
    // assemble the rolled result in L1 via element-level local copies (no DRAM alignment
    // constraint on L1 ops), then write the complete shard from L1 back to DRAM.
    // This avoids the NOC DRAM write alignment problem (32-byte minimum) that would arise
    // from per-row/per-element NOC writes. The write is always one full shard = shard_size
    // bytes, which we validate is 32-byte aligned below.
    if (is_dram_rm) {
        // Each shard cell-row is a buffer page padded up to the backing memory's alignment.
        // For DRAM this is the DRAM alignment (64B on Blackhole, 32B on Wormhole), NOT the L1
        // alignment — using the wrong alignment here desyncs the staged layout from the actual
        // in-DRAM layout and corrupts the gather.
        const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
        const uint32_t shard_size_check =
            (input.shard_spec().value().shape[0] *
             ((input.shard_spec().value().shape[1] * input.element_size() + dram_alignment - 1) / dram_alignment) *
             dram_alignment);
        TT_FATAL(
            shard_size_check % dram_alignment == 0,
            "DRAM-sharded RM roll requires shard_size ({} bytes) to be {}-byte aligned for NOC DRAM writes. "
            "Adjust shard dimensions so shard_h × align_up(shard_w × elem_size, {}) is a multiple of {}.",
            shard_size_check,
            dram_alignment,
            dram_alignment,
            dram_alignment);
    }

    const auto& shape = input.padded_shape();
    const uint32_t rank = shape.rank();
    const uint32_t shift = operation_attributes.shift;
    const int32_t dim = operation_attributes.dim;
    const bool is_last_dim = (static_cast<uint32_t>(dim) == rank - 1);

    // The gather works in "cells": a cell is one element for ROW_MAJOR, one tile for TILE.
    // Tile cells are contiguous in L1 and naturally aligned, so a tile-aligned roll is just a
    // permutation/rotation of whole tiles — identical to the row-major element gather.
    const bool is_tile = input.layout() == Layout::TILE;
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t cell_h = is_tile ? tt::constants::TILE_HEIGHT : 1;
    const uint32_t cell_w = is_tile ? tt::constants::TILE_WIDTH : 1;
    const uint32_t cell_size = is_tile ? tt::tile_size(cb_data_format) : input.element_size();

    // Tile-aligned shift is required along the last two dims when tilized.
    if (is_tile) {
        if (is_last_dim) {
            TT_FATAL(shift % cell_w == 0, "Native tilized roll on the width dim requires a tile-aligned shift");
        } else if (static_cast<uint32_t>(dim) == rank - 2) {
            TT_FATAL(shift % cell_h == 0, "Native tilized roll on the height dim requires a tile-aligned shift");
        }
    }

    const uint32_t W_cells = shape[rank - 1] / cell_w;

    const auto& out_ss = output.shard_spec().value();
    const auto& in_ss = input.shard_spec().value();
    const uint32_t shard_cells_h = out_ss.shape[0] / cell_h;
    const uint32_t shard_cells_w = out_ss.shape[1] / cell_w;
    TT_FATAL(
        in_ss.shape[0] == out_ss.shape[0] && in_ss.shape[1] == out_ss.shape[1],
        "Native sharded roll expects identical input/output shard shapes");
    TT_FATAL(W_cells % shard_cells_w == 0, "Shard width must evenly divide the tensor");

    // Cell-row dim sizes (dims 0..rank-2 collapsed): the height dim is measured in tile-rows.
    std::vector<uint32_t> rd(rank, 1);
    uint32_t H_cells = 1;
    for (uint32_t i = 0; i + 1 < rank; i++) {
        rd[i] = (i == rank - 2) ? shape[i] / cell_h : shape[i];
        H_cells *= rd[i];
    }
    TT_FATAL(H_cells % shard_cells_h == 0, "Shard height must evenly divide the tensor");

    auto* device = input.device();

    // Support both ROW_MAJOR and COL_MAJOR shard orientations.
    // ROW_MAJOR: tensor height → grid y-axis, tensor width → grid x-axis.
    // COL_MAJOR: tensor height → grid x-axis, tensor width → grid y-axis.
    // This mirrors the pattern in concat_block_sharded_program_factory.cpp lines 63–66.
    const bool row_major_orient = out_ss.orientation == ShardOrientation::ROW_MAJOR;
    TT_FATAL(
        out_ss.grid.ranges().size() == 1, "Native sharded roll requires a single contiguous rectangular CoreRange");
    const auto& grid_range = *out_ss.grid.ranges().begin();
    const uint32_t grid_cols = grid_range.end_coord.x - grid_range.start_coord.x + 1;
    const uint32_t grid_rows = grid_range.end_coord.y - grid_range.start_coord.y + 1;

    // Number of shard positions in the tensor width direction (used in shard_linear).
    const uint32_t n_shard_cols = W_cells / shard_cells_w;

    // Pre-compute physical core map indexed by [gy][gx]. Cores enumerated y-outer, x-inner.
    std::vector<std::vector<CoreCoord>> physical_cores(grid_rows, std::vector<CoreCoord>(grid_cols));
    for (uint32_t gy = 0; gy < grid_rows; gy++) {
        for (uint32_t gx = 0; gx < grid_cols; gx++) {
            physical_cores[gy][gx] = device->worker_core_from_logical_core(
                CoreCoord(grid_range.start_coord.x + gx, grid_range.start_coord.y + gy));
        }
    }

    // shard_linear maps (cell_row, cell_col) → core enumeration index c.
    // The core enumeration is y-outer, x-inner, so c = gy * grid_cols + gx.
    // ROW_MAJOR: shard (sr,sc) → (gy=sr, gx=sc) → c = sr*n_shard_cols + sc.
    // COL_MAJOR: shard (sr,sc) → (gx=sr, gy=sc) → c = sc*grid_cols + sr.
    auto shard_linear = [&](uint32_t row, uint32_t col) -> uint32_t {
        const uint32_t sr = row / shard_cells_h;
        const uint32_t sc = col / shard_cells_w;
        return row_major_orient ? sr * n_shard_cols + sc : sc * grid_cols + sr;
    };

    const uint32_t num_cores = grid_rows * grid_cols;

    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    // Sharded buffers store one shard cell-row per page, padded up to the backing memory's
    // alignment. DRAM pages use the (larger) DRAM alignment — 64B on Blackhole, 32B on Wormhole —
    // whereas L1 pages use the L1 alignment. The row pitch must match whichever memory actually
    // holds the data, otherwise the staged copy and the host-computed offsets disagree.
    const uint32_t page_alignment = is_dram ? tt::tt_metal::hal::get_dram_alignment() : l1_alignment;
    const uint32_t row_pitch_bytes =
        ((shard_cells_w * cell_size + page_alignment - 1) / page_alignment) * page_alignment;

    auto local_offset = [&](uint32_t row, uint32_t col) {
        const uint32_t local_r = row % shard_cells_h;
        const uint32_t local_c = col % shard_cells_w;
        return local_r * row_pitch_bytes + local_c * cell_size;
    };

    // Strides over the cell-row dims (all dims except the last), row-major, for higher-dim rolls.
    std::vector<uint32_t> row_stride(rank, 0);
    if (!is_last_dim) {
        uint32_t s = 1;
        for (int32_t k = static_cast<int32_t>(rank) - 2; k >= 0; k--) {
            row_stride[k] = s;
            s *= rd[k];
        }
    }
    // Coordinate shift for the rolled dim, in cell-row units (tile-rows for the height dim).
    const uint32_t dim_size_cells = (static_cast<uint32_t>(dim) == rank - 2) ? rd[dim] : shape[dim];
    const uint32_t shift_cells = (static_cast<uint32_t>(dim) == rank - 2) ? shift / cell_h : shift;

    auto rolled_src_row = [&](uint32_t r) -> uint32_t {
        // Decrement the dim-th coordinate by shift (mod dim_size); other coords unchanged.
        const uint32_t coord_d = (r / row_stride[dim]) % dim_size_cells;
        const uint32_t src_coord_d = (coord_d + dim_size_cells - (shift_cells % dim_size_cells)) % dim_size_cells;
        return r + (src_coord_d - coord_d) * row_stride[dim];
    };

    // --- Column-piece structure (identical for every cell-row) ---
    // Last-dim roll rotates columns within a row; higher-dim rolls keep columns identity.
    auto split_into_pieces = [&](uint32_t dst_col0, uint32_t src_col0, uint32_t len, std::vector<ColPiece>& out) {
        uint32_t done = 0;
        while (done < len) {
            const uint32_t dcol = dst_col0 + done;
            const uint32_t scol = src_col0 + done;
            const uint32_t dst_boundary = (dcol / shard_cells_w + 1) * shard_cells_w;
            const uint32_t src_boundary = (scol / shard_cells_w + 1) * shard_cells_w;
            const uint32_t piece = std::min({len - done, dst_boundary - dcol, src_boundary - scol});
            out.push_back(ColPiece{.dst_col = dcol, .src_col = scol, .len = piece});
            done += piece;
        }
    };

    std::vector<ColPiece> col_pieces;
    if (is_last_dim) {
        const uint32_t s = (shift / cell_w) % W_cells;
        if (s > 0) {
            split_into_pieces(0, W_cells - s, s, col_pieces);  // out[:, 0:s)   <- in[:, W-s:W)
        }
        split_into_pieces(s, 0, W_cells - s, col_pieces);  // out[:, s:W)   <- in[:, 0:W-s)
    } else {
        split_into_pieces(0, 0, W_cells, col_pieces);  // identity columns, permuted rows
    }

    // --- Build transfers, coalescing consecutive cell-rows into strided runs ---
    std::vector<std::vector<RollTransferDesc>> all_transfers(num_cores);

    for (const auto& p : col_pieces) {
        const uint32_t copy_bytes = p.len * cell_size;
        bool have_run = false;
        uint32_t run_dst_core = 0, run_src_core = 0;
        uint32_t run_src_off = 0, run_dst_off = 0, run_num = 0;
        auto flush = [&]() {
            if (!have_run) {
                return;
            }
            all_transfers[run_dst_core].push_back(RollTransferDesc{
                .src_physical_core = physical_cores[run_src_core / grid_cols][run_src_core % grid_cols],
                .src_dram_shard_idx = run_src_core,
                .src_l1_offset = run_src_off,
                .dst_offset = run_dst_off,
                .copy_size = copy_bytes,
                .src_stride = row_pitch_bytes,
                .dst_stride = row_pitch_bytes,
                .num_rows = run_num,
            });
            have_run = false;
        };
        for (uint32_t r = 0; r < H_cells; r++) {
            const uint32_t src_row = is_last_dim ? r : rolled_src_row(r);
            const uint32_t dst_core = shard_linear(r, p.dst_col);
            const uint32_t src_core = shard_linear(src_row, p.src_col);
            const uint32_t src_off = local_offset(src_row, p.src_col);
            const uint32_t dst_off = local_offset(r, p.dst_col);
            // Extend the run only if cores match and both offsets advance by exactly one pitch.
            if (have_run && dst_core == run_dst_core && src_core == run_src_core &&
                dst_off == run_dst_off + run_num * row_pitch_bytes &&
                src_off == run_src_off + run_num * row_pitch_bytes) {
                run_num++;
            } else {
                flush();
                have_run = true;
                run_dst_core = dst_core;
                run_src_core = src_core;
                run_src_off = src_off;
                run_dst_off = dst_off;
                run_num = 1;
            }
        }
        flush();
    }

    // --- Circular buffers ---
    // L1 mode: cb0 backed by input buffer, cb16 backed by output buffer, cb1 scratch.
    // DRAM mode: only the scratch cb1 is allocated in L1. Input/output CBs are not used
    // because DRAM data is addressed via bank IDs in the runtime args, not CB read ptrs.
    constexpr uint32_t input_cb_id = 0;
    constexpr uint32_t output_cb_id = 16;
    constexpr uint32_t scratch_cb_id = 1;
    const uint32_t cb_page_size = is_tile ? cell_size : row_pitch_bytes;

    ProgramDescriptor desc;
    const uint32_t shard_l1_size = shard_cells_h * row_pitch_bytes;
    if (!is_dram) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = shard_l1_size,
            .core_ranges = out_ss.grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(input_cb_id),
                .data_format = cb_data_format,
                .page_size = cb_page_size,
            }}},
            .buffer = input.buffer(),
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = shard_l1_size,
            .core_ranges = out_ss.grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_id),
                .data_format = cb_data_format,
                .page_size = cb_page_size,
            }}},
            .buffer = output.buffer(),
        });
    }
    // Scratch CB: L1 mode uses it double-buffered; DRAM TILE uses it single-buffered.
    // DRAM RM allocates separate staging CBs (2/3/4) instead, so scratch is L1-only.
    const uint32_t scratch_half = row_pitch_bytes + 2 * l1_alignment;
    const uint32_t scratch_size = (is_dram && !is_dram_rm) ? shard_l1_size : 2 * scratch_half;
    if (!is_dram_rm) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = scratch_size,
            .core_ranges = out_ss.grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(scratch_cb_id),
                .data_format = cb_data_format,
                .page_size = scratch_size,
            }}},
        });
    }

    // DRAM RM staging CBs: two source slots + one destination, each = full shard size.
    constexpr uint32_t dram_rm_src0_cb_id = 2;
    constexpr uint32_t dram_rm_src1_cb_id = 3;
    constexpr uint32_t dram_rm_dst_cb_id = 4;
    if (is_dram_rm) {
        for (uint8_t cb_id : {dram_rm_src0_cb_id, dram_rm_src1_cb_id, dram_rm_dst_cb_id}) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = shard_l1_size,
                .core_ranges = out_ss.grid,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = cb_id,
                    .data_format = cb_data_format,
                    .page_size = shard_l1_size,
                }}},
            });
        }
    }

    // --- DRAM shard address helpers ---
    const uint32_t num_dram_banks = device->num_dram_channels();
    const uint32_t dram_shard_size = shard_cells_h * row_pitch_bytes;
    auto dram_bank_id = [&](uint32_t shard_idx) { return shard_idx % num_dram_banks; };
    auto dram_bank_base = [&](const Buffer* buf, uint32_t shard_idx) {
        return static_cast<uint32_t>(buf->address()) + (shard_idx / num_dram_banks) * dram_shard_size;
    };

    // --- Kernel ---
    uint32_t max_num_transfers = 0;
    for (const auto& t : all_transfers) {
        max_num_transfers = std::max(max_num_transfers, static_cast<uint32_t>(t.size()));
    }
    constexpr uint32_t runtime_args_limit = 256;
    // L1 mode:       1 + 9*N args.
    // DRAM TILE:     3 + 7*N args (dst_bank_id, dst_bank_base, count, then 7 per transfer).
    // DRAM RM:       3 + max_4_src_info + 1 + 7*N args (dst, num_src≤2, src0, src1?, count, 7*N).
    //                worst case: 3 + 4 + 1 + 7*N = 8 + 7*N.
    const uint32_t args_per_transfer = is_dram_rm ? 7u : (is_dram ? 7u : 9u);
    const uint32_t args_overhead = is_dram_rm ? 8u : (is_dram ? 3u : 1u);
    TT_FATAL(
        args_overhead + max_num_transfers * args_per_transfer <= runtime_args_limit,
        "Native sharded roll: too many copy segments per core ({}). Reduce grid/shape.",
        max_num_transfers);

    // mode: 0=L1, 1=DRAM_TILE, 2=DRAM_RM
    const uint32_t mode = is_dram_rm ? 2u : (is_dram ? 1u : 0u);
    const std::vector<uint32_t> compile_time_args = {
        output_cb_id,
        scratch_cb_id,
        l1_alignment,
        scratch_half,
        mode,
        shard_l1_size,
        dram_rm_src0_cb_id,
        dram_rm_src1_cb_id,
        dram_rm_dst_cb_id};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/roll/device/kernels/dataflow/roll_sharded_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = out_ss.grid;
    reader_desc.compile_time_args = compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    auto build_runtime_args_l1 = [&](const std::vector<RollTransferDesc>& descs) {
        KernelDescriptor::CoreRuntimeArgs args;
        args.reserve(1 + descs.size() * 9);
        args.push_back(static_cast<uint32_t>(descs.size()));
        for (const auto& td : descs) {
            args.push_back(td.src_physical_core.x);
            args.push_back(td.src_physical_core.y);
            args.push_back(input_cb_id);
            args.push_back(td.src_l1_offset);
            args.push_back(td.dst_offset);
            args.push_back(td.copy_size);
            args.push_back(td.src_stride);
            args.push_back(td.dst_stride);
            args.push_back(td.num_rows);
        }
        return args;
    };

    auto build_runtime_args_dram = [&](uint32_t dst_core_idx, const std::vector<RollTransferDesc>& descs) {
        KernelDescriptor::CoreRuntimeArgs args;
        args.reserve(3 + descs.size() * 7);
        args.push_back(dram_bank_id(dst_core_idx));
        args.push_back(dram_bank_base(output.buffer(), dst_core_idx));
        args.push_back(static_cast<uint32_t>(descs.size()));
        for (const auto& td : descs) {
            // src_bank_id, src_bank_addr (= bank_base + intra_shard_offset), dst_offset,
            // copy_size, src_stride, dst_stride, num_rows
            args.push_back(dram_bank_id(td.src_dram_shard_idx));
            args.push_back(dram_bank_base(input.buffer(), td.src_dram_shard_idx) + td.src_l1_offset);
            args.push_back(td.dst_offset);
            args.push_back(td.copy_size);
            args.push_back(td.src_stride);
            args.push_back(td.dst_stride);
            args.push_back(td.num_rows);
        }
        return args;
    };

    // DRAM RM mode: full-shard L1 staging.
    // Per core: [dst_bank_id, dst_bank_base, num_src, (src0_bank_id, src0_addr)..., num_xfers,
    //            (src_slot, src_off, dst_off, copy_size, src_stride, dst_stride, num_rows) x N]
    auto build_runtime_args_dram_rm = [&](uint32_t dst_core_idx, const std::vector<RollTransferDesc>& descs) {
        // Collect unique source shards (at most 2) and assign them to staging slots 0/1.
        std::vector<uint32_t> src_shards;
        std::unordered_map<uint32_t, uint32_t> src_to_slot;
        for (const auto& td : descs) {
            if (!src_to_slot.contains(td.src_dram_shard_idx)) {
                src_to_slot[td.src_dram_shard_idx] = static_cast<uint32_t>(src_shards.size());
                src_shards.push_back(td.src_dram_shard_idx);
            }
        }
        KernelDescriptor::CoreRuntimeArgs args;
        args.push_back(dram_bank_id(dst_core_idx));
        args.push_back(dram_bank_base(output.buffer(), dst_core_idx));
        args.push_back(static_cast<uint32_t>(src_shards.size()));
        for (uint32_t s : src_shards) {
            args.push_back(dram_bank_id(s));
            args.push_back(dram_bank_base(input.buffer(), s));
        }
        args.push_back(static_cast<uint32_t>(descs.size()));
        for (const auto& td : descs) {
            args.push_back(src_to_slot.at(td.src_dram_shard_idx));
            args.push_back(td.src_l1_offset);
            args.push_back(td.dst_offset);
            args.push_back(td.copy_size);
            args.push_back(td.src_stride);
            args.push_back(td.dst_stride);
            args.push_back(td.num_rows);
        }
        return args;
    };

    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord logical(grid_range.start_coord.x + c % grid_cols, grid_range.start_coord.y + c / grid_cols);
        KernelDescriptor::CoreRuntimeArgs args;
        if (is_dram_rm) {
            args = build_runtime_args_dram_rm(c, all_transfers[c]);
        } else if (is_dram) {
            args = build_runtime_args_dram(c, all_transfers[c]);
        } else {
            args = build_runtime_args_l1(all_transfers[c]);
        }
        reader_desc.runtime_args.emplace_back(logical, std::move(args));
    }

    desc.kernels.push_back(std::move(reader_desc));
    return desc;
}

}  // namespace ttnn::prim
