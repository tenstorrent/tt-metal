// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_reshard issuer kernel (PULL regime). One instance per data-movement RISC of every core that
// holds an out block. Each issuer pulls the tiles of its row range of this core's out block from
// the cores holding them in in0, straight into this core's out shard; no CB, no compute.
//
// Raw NoC reads instead of a kernel_lib helper: local_copy_helpers_dataflow is self-aimed (source
// is always this core), McastArgs moves one source to many targets (no tile of a reshard goes to
// two cores), and TensorAccessor resolves addresses per page and never coalesces across shard
// rows. The piece walk below needs the in0 block boundaries anyway.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

constexpr uint32_t issuer_index = get_compile_time_arg_val(0);
constexpr uint32_t issuers_per_core = get_compile_time_arg_val(1);
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

static_assert(issuer_index < issuers_per_core, "issuer_index out of range");
static_assert(block_rows_tiles > 0 && block_cols_tiles > 0, "block extents must be non-zero");

// The issuer split is by out-shard rows only: a column split would cut runs.
constexpr uint32_t issuer_rows_tiles = (out_shard_rows_tiles + issuers_per_core - 1) / issuers_per_core;

// Shards keep their full row stride even when cut short at the tensor edge.
constexpr uint32_t in_row_stride_bytes = in_shard_cols_tiles * page_bytes;
constexpr uint32_t out_row_stride_bytes = out_shard_cols_tiles * page_bytes;

// Common RT args: in_addr, out_addr, then the in0 NoC coordinate tables.
constexpr uint32_t in_addr_arg = 0;
constexpr uint32_t out_addr_arg = 1;
constexpr uint32_t in_block_row_noc_arg = 2;
constexpr uint32_t in_block_col_noc_arg = in_block_row_noc_arg + in_num_block_rows;

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
FORCE_INLINE uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

struct OutBlock {
    uint32_t row0;           // first tensor tile row of this core's out block
    uint32_t col0;           // first tensor tile column of this core's out block
    uint32_t cols;           // valid tile columns (cut short only on the last block column)
    uint32_t issuer_row_lo;  // this issuer's tensor tile rows [lo, hi)
    uint32_t issuer_row_hi;
};

FORCE_INLINE OutBlock locate_block() {
    const uint32_t gx = get_absolute_logical_x() - out_x0;
    const uint32_t gy = get_absolute_logical_y() - out_y0;
    const uint32_t r_out = out_col_major ? gx : gy;
    const uint32_t c_out = out_col_major ? gy : gx;

    OutBlock blk;
    blk.row0 = r_out * out_shard_rows_tiles;
    blk.col0 = c_out * out_shard_cols_tiles;
    const uint32_t core_rows_tiles = min_u32(out_shard_rows_tiles, tensor_rows_tiles - blk.row0);
    blk.cols = min_u32(out_shard_cols_tiles, tensor_cols_tiles - blk.col0);

    const uint32_t local_lo = min_u32(issuer_index * issuer_rows_tiles, core_rows_tiles);
    const uint32_t local_hi = min_u32(local_lo + issuer_rows_tiles, core_rows_tiles);
    blk.issuer_row_lo = blk.row0 + local_lo;
    blk.issuer_row_hi = blk.row0 + local_hi;
    return blk;
}

// Issues one NoC read per run for every piece where the block [row_lo,row_hi) x [col_lo,col_hi)
// meets an in0 block. No barrier: every run stays in flight until land_blocks().
FORCE_INLINE void gather_block(
    const OutBlock& blk,
    uint32_t in_addr,
    uint32_t out_addr,
    uint32_t row_lo,
    uint32_t row_hi,
    uint32_t col_lo,
    uint32_t col_hi) {
    for (uint32_t br = row_lo / in_shard_rows_tiles; br * in_shard_rows_tiles < row_hi; ++br) {
        const uint32_t in_row0 = br * in_shard_rows_tiles;
        const uint32_t piece_row_lo = max_u32(row_lo, in_row0);
        const uint32_t piece_rows = min_u32(row_hi, in_row0 + in_shard_rows_tiles) - piece_row_lo;
        const uint32_t row_noc = get_common_arg_val<uint32_t>(in_block_row_noc_arg + br);

        for (uint32_t bc = col_lo / in_shard_cols_tiles; bc * in_shard_cols_tiles < col_hi; ++bc) {
            const uint32_t in_col0 = bc * in_shard_cols_tiles;
            const uint32_t piece_col_lo = max_u32(col_lo, in_col0);
            const uint32_t piece_cols = min_u32(col_hi, in_col0 + in_shard_cols_tiles) - piece_col_lo;
            const uint32_t col_noc = get_common_arg_val<uint32_t>(in_block_col_noc_arg + bc);
            const uint32_t noc_x = in_col_major ? row_noc : col_noc;
            const uint32_t noc_y = in_col_major ? col_noc : row_noc;

            const uint32_t src_l1 =
                in_addr + ((piece_row_lo - in_row0) * in_shard_cols_tiles + (piece_col_lo - in_col0)) * page_bytes;
            uint32_t dst_l1 =
                out_addr + ((piece_row_lo - blk.row0) * out_shard_cols_tiles + (piece_col_lo - blk.col0)) * page_bytes;
            uint64_t src_noc = get_noc_addr(noc_x, noc_y, src_l1);

            // Rows of a piece are contiguous on both ends only when the piece spans the full
            // shard width of both shards; comparing against the cut-short block width would
            // coalesce across a stride gap.
            const bool one_run =
                piece_rows == 1 || (piece_cols == in_shard_cols_tiles && piece_cols == out_shard_cols_tiles);
            if (one_run) {
                noc_async_read(src_noc, dst_l1, piece_rows * piece_cols * page_bytes);
            } else {
                const uint32_t run_bytes = piece_cols * page_bytes;
                for (uint32_t r = 0; r < piece_rows; ++r) {
                    noc_async_read(src_noc, dst_l1, run_bytes);
                    src_noc += in_row_stride_bytes;
                    dst_l1 += out_row_stride_bytes;
                }
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t in_addr = get_common_arg_val<uint32_t>(in_addr_arg);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(out_addr_arg);

    {
        DeviceZoneScopedN("G6-GATHER");
        const OutBlock blk = locate_block();
        const uint32_t num_row_blocks =
            (blk.issuer_row_hi - blk.issuer_row_lo + block_rows_tiles - 1) / block_rows_tiles;
        const uint32_t num_col_blocks = (blk.cols + block_cols_tiles - 1) / block_cols_tiles;
        const uint32_t num_blocks_this_issuer = num_row_blocks * num_col_blocks;
        const uint32_t col_end = blk.col0 + blk.cols;
        for (uint32_t block_idx = 0; block_idx < num_blocks_this_issuer; ++block_idx) {
            const uint32_t row_lo = blk.issuer_row_lo + (block_idx / num_col_blocks) * block_rows_tiles;
            const uint32_t col_lo = blk.col0 + (block_idx % num_col_blocks) * block_cols_tiles;
            gather_block(
                blk,
                in_addr,
                out_addr,
                row_lo,
                min_u32(row_lo + block_rows_tiles, blk.issuer_row_hi),
                col_lo,
                min_u32(col_lo + block_cols_tiles, col_end));
        }
    }

    {
        DeviceZoneScopedN("G6-LAND");
        noc_async_read_barrier();
    }
}
