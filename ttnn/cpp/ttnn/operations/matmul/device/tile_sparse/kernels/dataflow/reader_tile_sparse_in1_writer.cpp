// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-sparse matmul: in1 (B matrix) reader + output writer with K-block skip.
//
// For each K-block, checks the global K-active bitmask:
//   - Active  (bit=1): reads B tile from DRAM and pushes to CB.
//   - Inactive(bit=0): skips the K-block entirely — no DRAM read, no CB push.
//
// Supports up to 32 K-blocks (uint32_t bitmask). Pass 0xFFFFFFFF for dense path.
//
// Compile-time args:
//   0  in1_stride_w         -- tile stride in N direction (usually 1)
//   1  in1_stride_h         -- tile stride in K direction (= Nt)
//   2  in1_k_stride         -- tile stride between K-blocks in B (= in0_block_w * Nt)
//   3  in1_n_stride         -- tile stride between N-blocks (= out_block_w)
//   4  in1_block_w          -- N-dim tiles per block (= out_block_w)
//   5  in1_block_h          -- K-dim tiles per block (= in0_block_w)
//   6  in1_block_num_tiles  -- in1_block_w * in1_block_h
//   7  num_k_blocks         -- total K-blocks = Kt / in0_block_w
//   8  num_n_blocks         -- N outer blocks = per_core_N / out_block_w (usually 1)
//   9  num_m_blocks         -- M outer blocks = per_core_M / out_block_h
//  10  out_stride_w         -- output tile stride in N (= 1)
//  11  out_stride_h         -- output tile stride in M (= Nt)
//  12  out_sb_stride_w      -- output subblock stride in N (= out_subblock_w)
//  13  out_sb_stride_h      -- output subblock stride in M (= out_subblock_h * Nt)
//  14  out_blk_stride_w     -- output block stride in N (= out_block_w)
//  15  out_blk_stride_h     -- output block stride in M (= out_block_h * Nt)
//  16  out_subblock_w       -- output subblock width
//  17  out_subblock_h       -- output subblock height
//  18  out_sb_tiles         -- out_subblock_w * out_subblock_h
//  19  MtNt                 -- M * N tile count (not currently used)
//  20+ TensorAccessorArgs for in1
//  ..  TensorAccessorArgs for out  (after in1 args)
//
// Runtime args (must match factory writer_runtime_args layout):
//   0  in1_tensor_addr
//   1  in1_start_tile       -- per_core_N * output_idx_x
//   2-5 mcast placeholders  -- ignored (no in1 mcast)
//   6  k_active_bitmask     -- bit k = 1 → K-block k is active for this core's N column
//   7  out_tensor_addr
//   8  out_start_tile       -- output_idx_x * per_core_N + output_idx_y * per_core_M * Nt
//   9  out_num_subblocks_h  -- out_block_h / out_subblock_h
//  10  out_last_subblock_h  -- out_subblock_h (for non-padded; last subblock height)
//  11  pad_tiles_h          -- 0 for non-padded
//  12  out_num_subblocks_w  -- out_block_w / out_subblock_w
//  13  out_last_nz_sbw      -- same as above (for last N-block)
//  14  out_last_subblock_w  -- out_subblock_w (for non-padded)
//  15  pad_sb_tiles         -- 0 for non-padded
//  16  pad_tiles_w          -- 0 for non-padded
//  17-18 bias placeholders  -- ignored
//  19  last_blocks_w_dim    -- ignored (no multi-N-block support needed)
//  ...  additional padding  -- ignored

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // ---- Runtime args ----
    uint32_t rt = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t in1_start = get_arg_val<uint32_t>(rt++);
    rt += 4;  // skip mcast placeholders
    const uint32_t k_mask = get_arg_val<uint32_t>(rt++);
    const uint32_t out_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t out_start = get_arg_val<uint32_t>(rt++);
    const uint32_t out_sh_h = get_arg_val<uint32_t>(rt++);   // subblocks per M-block
    const uint32_t out_lsh_h = get_arg_val<uint32_t>(rt++);  // last subblock height
    const uint32_t pad_h = get_arg_val<uint32_t>(rt++);      // padded tiles h
    const uint32_t out_sh_w = get_arg_val<uint32_t>(rt++);   // subblocks per N-block
    const uint32_t out_lsh_w = get_arg_val<uint32_t>(rt++);  // last nz subblocks w (last N-block)
    const uint32_t out_lsw = get_arg_val<uint32_t>(rt++);    // last subblock width
    const uint32_t pad_sb = get_arg_val<uint32_t>(rt++);     // padded subblock tiles
    const uint32_t pad_w = get_arg_val<uint32_t>(rt++);      // padded block tiles w

    // ---- Compile-time args ----
    constexpr uint32_t in1_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in1_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in1_k_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in1_n_stride = get_compile_time_arg_val(3);
    constexpr uint32_t in1_bw = get_compile_time_arg_val(4);
    constexpr uint32_t in1_bh = get_compile_time_arg_val(5);
    constexpr uint32_t in1_btn = get_compile_time_arg_val(6);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t out_stride_w = get_compile_time_arg_val(10);
    constexpr uint32_t out_stride_h = get_compile_time_arg_val(11);
    constexpr uint32_t out_sb_sw = get_compile_time_arg_val(12);
    constexpr uint32_t out_sb_sh = get_compile_time_arg_val(13);
    constexpr uint32_t out_blk_sw = get_compile_time_arg_val(14);
    constexpr uint32_t out_blk_sh = get_compile_time_arg_val(15);
    constexpr uint32_t out_sbw = get_compile_time_arg_val(16);
    constexpr uint32_t out_sbh = get_compile_time_arg_val(17);
    constexpr uint32_t out_sb_tiles = get_compile_time_arg_val(18);
    // constexpr uint32_t MtNt = get_compile_time_arg_val(19);  // not needed at runtime

    constexpr auto in1_ta = TensorAccessorArgs<20>();
    constexpr auto out_ta = TensorAccessorArgs<in1_ta.next_compile_time_args_offset()>();

    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t in1_tbytes = get_tile_size(cb_in1);
    constexpr uint32_t out_tbytes = get_tile_size(cb_out);

    const auto s1 = TensorAccessor(in1_ta, in1_addr, in1_tbytes);
    const auto sout = TensorAccessor(out_ta, out_addr, out_tbytes);

    uint32_t out_h_tile = out_start;

    for (uint32_t bm = 0; bm < num_m_blocks; ++bm) {
        uint32_t out_w_tile = out_h_tile;

        for (uint32_t bn = 0; bn < num_n_blocks; ++bn) {
            uint32_t in1_k_tile_start = in1_start + bn * in1_n_stride;

            // ---- Read in1 (B matrix) K-blocks ----
            for (uint32_t bk = 0; bk < num_k_blocks; ++bk) {
                if (!((k_mask >> bk) & 1u)) {
                    continue;  // Skip inactive K-block entirely
                }

                cb_reserve_back(cb_in1, in1_btn);
                uint32_t l1_addr = get_write_ptr(cb_in1);

                // Read B[bk, bn] block from DRAM
                uint32_t dst = l1_addr;
                uint32_t k_tile = in1_k_tile_start + bk * in1_k_stride;
                for (uint32_t h = 0; h < in1_bh; ++h) {
                    uint32_t row_tile = k_tile + h * in1_stride_h;
                    for (uint32_t w = 0; w < in1_bw; ++w) {
                        noc_async_read_tile(row_tile + w * in1_stride_w, s1, dst);
                        dst += in1_tbytes;
                    }
                }
                noc_async_read_barrier();

                cb_push_back(cb_in1, in1_btn);
            }  // bk loop

            // ---- Write output tiles to DRAM ----
            // After compute has processed all K-blocks for this (bm, bn),
            // the result subblocks are packed into cb_out. Write them to DRAM.
            uint32_t out_sbh_tile = out_w_tile;
            for (uint32_t sbh = 0; sbh < out_sh_h; ++sbh) {
                uint32_t out_sbw_tile = out_sbh_tile;
                uint32_t sb_nzw = (bn == num_n_blocks - 1) ? out_lsh_w : out_sh_w;

                for (uint32_t sbw = 0; sbw < sb_nzw; ++sbw) {
                    // Determine effective subblock dims (handles last-subblock padding)
                    uint32_t eff_sbh = (bm == num_m_blocks - 1 && sbh == out_sh_h - 1) ? out_lsh_h : out_sbh;
                    uint32_t eff_sbw = (bn == num_n_blocks - 1 && sbw == sb_nzw - 1) ? out_lsw : out_sbw;
                    uint32_t sb_addr_skip = (bn == num_n_blocks - 1 && sbw == sb_nzw - 1) ? pad_sb : 0;

                    cb_wait_front(cb_out, out_sb_tiles);
                    uint32_t l1_read = get_read_ptr(cb_out);

                    for (uint32_t h = 0; h < eff_sbh; ++h) {
                        uint32_t out_row = out_sbw_tile + h * out_stride_h;
                        uint32_t l1_row = l1_read + h * out_sbw * out_tbytes;
                        for (uint32_t w = 0; w < eff_sbw; ++w) {
                            noc_async_write_tile(out_row + w * out_stride_w, sout, l1_row + w * out_tbytes);
                        }
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, out_sb_tiles);
                    (void)sb_addr_skip;  // skip address used for padded case

                    out_sbw_tile += out_sb_sw;
                }  // sbw loop

                // Pop fully padded subblocks along the N direction
                if (bn == num_n_blocks - 1 && pad_w > 0) {
                    cb_wait_front(cb_out, pad_w);
                    cb_pop_front(cb_out, pad_w);
                }

                out_sbh_tile += out_sb_sh;
            }  // sbh loop

            // Pop row(s) of fully padded subblocks
            if (bm == num_m_blocks - 1 && pad_h > 0) {
                cb_wait_front(cb_out, pad_h);
                cb_pop_front(cb_out, pad_h);
            }

            out_w_tile += out_blk_sw;
        }  // bn loop

        out_h_tile += out_blk_sh;
    }  // bm loop

    noc_async_write_barrier();
}
