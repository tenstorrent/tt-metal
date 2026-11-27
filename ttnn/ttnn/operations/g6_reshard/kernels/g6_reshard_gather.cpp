// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_reshard PULL issuer. One instance runs per issuer RISC of every core that holds an out block.
// It reads its rows of that block from the in0 shards on the source cores straight into this
// core's out shard: no CB, no staging, no compute.
//
// Helper substitution (all raw NoC): no kernel_lib helper reads another core's L1 into an
// arbitrary local address. local_copy_helpers aim at the reading core itself, McastArgs moves one
// source to many cores, and TensorAccessor resolves per page and cannot coalesce a run across
// shard rows (op_design.md, "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t issuers_per_core = get_compile_time_arg_val(0);
constexpr uint32_t issuer_index = get_compile_time_arg_val(1);
constexpr uint32_t block_rows_tiles = get_compile_time_arg_val(2);
constexpr uint32_t block_cols_tiles = get_compile_time_arg_val(3);
constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
constexpr uint32_t tensor_rows_tiles = get_compile_time_arg_val(5);
constexpr uint32_t tensor_cols_tiles = get_compile_time_arg_val(6);
constexpr uint32_t in_shard_rows_tiles = get_compile_time_arg_val(7);
constexpr uint32_t in_shard_cols_tiles = get_compile_time_arg_val(8);
constexpr bool in_col_major = get_compile_time_arg_val(9) != 0;
constexpr uint32_t in_num_block_rows = get_compile_time_arg_val(10);
constexpr uint32_t in_num_block_cols = get_compile_time_arg_val(11);
constexpr uint32_t out_shard_rows_tiles = get_compile_time_arg_val(12);
constexpr uint32_t out_shard_cols_tiles = get_compile_time_arg_val(13);
constexpr bool out_col_major = get_compile_time_arg_val(14) != 0;
constexpr uint32_t out_x0 = get_compile_time_arg_val(15);
constexpr uint32_t out_y0 = get_compile_time_arg_val(16);

// Common runtime args: buffer addresses, then the in0 NoC tables (one coordinate per block row,
// one per block column; virtual coordinates are separable, asserted on the host).
constexpr uint32_t kInAddrArg = 0;
constexpr uint32_t kOutAddrArg = 1;
constexpr uint32_t kInRowNocArg = 2;
constexpr uint32_t kInColNocArg = kInRowNocArg + in_num_block_rows;

// The issuer's share of an out block; block_rows_tiles cuts it into blocks.
constexpr uint32_t issuer_rows_share = (out_shard_rows_tiles + issuers_per_core - 1) / issuers_per_core;

// Shard pages keep the full shard row stride even in a cut-short tail shard.
constexpr uint32_t in_row_stride_bytes = in_shard_cols_tiles * page_bytes;
constexpr uint32_t out_row_stride_bytes = out_shard_cols_tiles * page_bytes;
constexpr bool shard_widths_match = in_shard_cols_tiles == out_shard_cols_tiles;

static_assert(issuer_index < issuers_per_core);
static_assert(block_rows_tiles > 0 && block_cols_tiles > 0);

inline uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
inline uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

// This issuer's part of the out block, in tensor tile coordinates.
struct IssuerSpan {
    uint32_t r_out;
    uint32_t c_out;
    uint32_t row_lo;  // [row_lo, row_hi) tile rows this issuer owns
    uint32_t row_hi;
    uint32_t col_lo;  // [col_lo, col_hi) the core's valid tile columns
    uint32_t col_hi;
};

IssuerSpan locate_block() {
    const uint32_t x = get_absolute_logical_x();
    const uint32_t y = get_absolute_logical_y();
    IssuerSpan s;
    s.r_out = out_col_major ? x - out_x0 : y - out_y0;
    s.c_out = out_col_major ? y - out_y0 : x - out_x0;

    const uint32_t block_row0 = s.r_out * out_shard_rows_tiles;
    const uint32_t block_col0 = s.c_out * out_shard_cols_tiles;
    const uint32_t core_rows_tiles = min_u32(out_shard_rows_tiles, tensor_rows_tiles - block_row0);
    const uint32_t core_cols_tiles = min_u32(out_shard_cols_tiles, tensor_cols_tiles - block_col0);

    const uint32_t local_lo = min_u32(issuer_index * issuer_rows_share, core_rows_tiles);
    const uint32_t local_hi = min_u32(local_lo + issuer_rows_share, core_rows_tiles);
    s.row_lo = block_row0 + local_lo;
    s.row_hi = block_row0 + local_hi;
    s.col_lo = block_col0;
    s.col_hi = block_col0 + core_cols_tiles;
    return s;
}

// Issues every run of one block: tile rows [row_lo, row_hi) x tile columns [col_lo, col_hi).
// A piece is where the block meets one in0 block (one source core); a run is one NoC read of
// tiles contiguous in both the source and the target shard. No barrier: the reads stay in flight.
void gather_block(
    const IssuerSpan& span,
    uint32_t row_lo,
    uint32_t row_hi,
    uint32_t col_lo,
    uint32_t col_hi,
    uint32_t in_addr,
    uint32_t out_addr,
    const tt_l1_ptr uint32_t* in_row_noc,
    const tt_l1_ptr uint32_t* in_col_noc) {
    const uint32_t out_row0 = span.r_out * out_shard_rows_tiles;
    const uint32_t out_col0 = span.c_out * out_shard_cols_tiles;

    for (uint32_t br = row_lo / in_shard_rows_tiles; br * in_shard_rows_tiles < row_hi; ++br) {
        const uint32_t in_row0 = br * in_shard_rows_tiles;
        const uint32_t piece_row_lo = max_u32(row_lo, in_row0);
        const uint32_t piece_rows = min_u32(row_hi, in_row0 + in_shard_rows_tiles) - piece_row_lo;
        const uint32_t row_noc = in_row_noc[br];

        for (uint32_t bc = col_lo / in_shard_cols_tiles; bc * in_shard_cols_tiles < col_hi; ++bc) {
            const uint32_t in_col0 = bc * in_shard_cols_tiles;
            const uint32_t piece_col_lo = max_u32(col_lo, in_col0);
            const uint32_t piece_cols = min_u32(col_hi, in_col0 + in_shard_cols_tiles) - piece_col_lo;
            const uint32_t col_noc = in_col_noc[bc];
            const uint32_t noc_x = in_col_major ? row_noc : col_noc;
            const uint32_t noc_y = in_col_major ? col_noc : row_noc;

            uint32_t src =
                in_addr + ((piece_row_lo - in_row0) * in_shard_cols_tiles + (piece_col_lo - in_col0)) * page_bytes;
            uint32_t dst =
                out_addr + ((piece_row_lo - out_row0) * out_shard_cols_tiles + (piece_col_lo - out_col0)) * page_bytes;

            // Rows coalesce only when the piece spans the full width of both shards; comparing the
            // piece against a tail block's actual width would read across a shard row.
            const bool one_run = piece_rows == 1 || (shard_widths_match && piece_cols == in_shard_cols_tiles);
            if (one_run) {
                noc_async_read(get_noc_addr(noc_x, noc_y, src), dst, piece_rows * piece_cols * page_bytes);
            } else {
                const uint32_t run_bytes = piece_cols * page_bytes;
                for (uint32_t r = 0; r < piece_rows; ++r) {
                    noc_async_read(get_noc_addr(noc_x, noc_y, src), dst, run_bytes);
                    src += in_row_stride_bytes;
                    dst += out_row_stride_bytes;
                }
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t in_addr = get_common_arg_val<uint32_t>(kInAddrArg);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(kOutAddrArg);
    const tt_l1_ptr uint32_t* in_row_noc =
        reinterpret_cast<const tt_l1_ptr uint32_t*>(get_common_arg_addr(kInRowNocArg));
    const tt_l1_ptr uint32_t* in_col_noc =
        reinterpret_cast<const tt_l1_ptr uint32_t*>(get_common_arg_addr(kInColNocArg));

    {
        DeviceZoneScopedN("G6-GATHER");
        const IssuerSpan span = locate_block();
        // Phase 0: one block per issuer (its whole share); an issuer with no rows issues nothing.
        const uint32_t num_row_blocks = (span.row_hi - span.row_lo + block_rows_tiles - 1) / block_rows_tiles;
        const uint32_t num_col_blocks = (span.col_hi - span.col_lo + block_cols_tiles - 1) / block_cols_tiles;
        for (uint32_t row_block = 0; row_block < num_row_blocks; ++row_block) {
            const uint32_t row_lo = span.row_lo + row_block * block_rows_tiles;
            const uint32_t row_hi = min_u32(row_lo + block_rows_tiles, span.row_hi);
            for (uint32_t col_block = 0; col_block < num_col_blocks; ++col_block) {
                const uint32_t col_lo = span.col_lo + col_block * block_cols_tiles;
                const uint32_t col_hi = min_u32(col_lo + block_cols_tiles, span.col_hi);
                gather_block(span, row_lo, row_hi, col_lo, col_hi, in_addr, out_addr, in_row_noc, in_col_noc);
            }
        }
    }
    {
        // land_blocks: the only wait, and every NoC transaction this issuer started ends here.
        DeviceZoneScopedN("G6-LAND");
        noc_async_read_barrier();
    }
}
