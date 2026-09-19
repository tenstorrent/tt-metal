// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The mcast matmul output write, factored out of
// reader_bmm_tile_layout_in1_receiver_writer_padding.cpp so that the in0 RISC can issue it instead
// of the in1 RISC. The body is that writer lifted verbatim: same subblock loops, same barrier per
// subblock, same padding handling, same compile time arg values in the same order. Only the thread
// that runs it changes.
//
// Compile time args from CTA onward, matching append_writer_ct_args in the program factory:
//   +0 out_tensor_stride_w                +5 out_tensor_next_h_dim_block_stride
//   +1 out_tensor_stride_h                +6 out_subblock_w
//   +2 out_tensor_next_subblock_stride_w  +7 out_subblock_h
//   +3 out_tensor_next_subblock_stride_h  +8 out_subblock_tile_count
//   +4 out_tensor_next_w_dim_block_stride +9 MtNt
//   +10..                                 TensorAccessorArgs(out)
// +4, +5 and +9 are read by the caller, which owns the block tile-id walk.
#pragma once

#include <stdint.h>

struct MatmulOutWriterArgs {
    uint32_t out_tensor_addr;
    uint32_t out_tensor_start_tile_id;
    uint32_t out_num_nonzero_subblocks_h;
    uint32_t out_last_num_nonzero_subblocks_h;
    uint32_t out_last_subblock_h;
    uint32_t padded_block_tiles_h_skip;
    uint32_t out_num_nonzero_subblocks_w;
    uint32_t out_last_num_nonzero_subblocks_w;
    uint32_t out_last_subblock_w;
    uint32_t padded_subblock_tiles_addr_skip;
    uint32_t padded_block_tiles_w_skip;
    uint32_t last_num_blocks_h_dim;
    uint32_t last_num_blocks_w_dim;
};

inline MatmulOutWriterArgs matmul_out_writer_args(uint32_t& rt_args_idx) {
    MatmulOutWriterArgs a;
    a.out_tensor_addr = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.out_tensor_start_tile_id = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.out_last_num_nonzero_subblocks_h = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.out_last_subblock_h = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.padded_block_tiles_h_skip = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.out_last_num_nonzero_subblocks_w = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.out_last_subblock_w = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.padded_subblock_tiles_addr_skip = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.padded_block_tiles_w_skip = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.last_num_blocks_h_dim = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    a.last_num_blocks_w_dim = get_arg_val<uint32_t>(static_cast<int>(rt_args_idx++));
    return a;
}

template <uint32_t CTA, uint32_t num_blocks_h_dim, uint32_t num_blocks_w_dim, typename Accessor>
inline void matmul_write_out_block(
    const Noc& noc,
    DataflowBuffer& dfb_out,
    const Accessor& s,
    const MatmulOutWriterArgs& wargs,
    uint32_t bh,
    uint32_t bw,
    uint32_t out_tensor_current_w_dim_block_tile_id) {
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(CTA + 0);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(CTA + 1);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(CTA + 2);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(CTA + 3);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(CTA + 6);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(CTA + 7);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(CTA + 8);

    const uint32_t output_single_tile_size_bytes = dfb_out.get_tile_size();

    const uint32_t num_blocks_h_dim_ =
        bh >= wargs.last_num_blocks_h_dim - 1 ? wargs.last_num_blocks_h_dim : num_blocks_h_dim;
    const uint32_t num_blocks_w_dim_ =
        bw >= wargs.last_num_blocks_w_dim - 1 ? wargs.last_num_blocks_w_dim : num_blocks_w_dim;
    uint32_t out_num_nonzero_subblocks_h_ = wargs.out_num_nonzero_subblocks_h;
    uint32_t out_num_nonzero_subblocks_w_ = wargs.out_num_nonzero_subblocks_w;
    if (bh == num_blocks_h_dim_ - 1) {
        out_num_nonzero_subblocks_h_ = wargs.out_last_num_nonzero_subblocks_h;
    }
    if (bw == num_blocks_w_dim_ - 1) {
        out_num_nonzero_subblocks_w_ = wargs.out_last_num_nonzero_subblocks_w;
    }
    uint32_t out_tensor_sbh_start_tile_id = out_tensor_current_w_dim_block_tile_id;
    for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h_; ++sbh) {
        uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
        for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w_; ++sbw) {
            uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

            uint32_t out_subblock_h_ = out_subblock_h;
            uint32_t out_subblock_w_ = out_subblock_w;
            uint32_t subblock_tiles_addr_skip = 0;
            if (bh == num_blocks_h_dim_ - 1 && sbh == out_num_nonzero_subblocks_h_ - 1) {
                out_subblock_h_ = wargs.out_last_subblock_h;
            }
            if (bw == num_blocks_w_dim_ - 1 && sbw == out_num_nonzero_subblocks_w_ - 1) {
                out_subblock_w_ = wargs.out_last_subblock_w;
                subblock_tiles_addr_skip = wargs.padded_subblock_tiles_addr_skip;
            }

            dfb_out.wait_front(out_subblock_tile_count);
            uint32_t out_read_offset = 0;

            for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                    if (bh < num_blocks_h_dim_ && bw < num_blocks_w_dim_) {
                        noc.async_write(
                            dfb_out,
                            s,
                            output_single_tile_size_bytes,
                            {.offset_bytes = out_read_offset},
                            {.page_id = out_tensor_tile_id});
                    }

                    out_read_offset += output_single_tile_size_bytes;

                    out_tensor_tile_id += out_tensor_stride_w;
                }
                // Skip padded tiles in subblock along row
                out_read_offset += subblock_tiles_addr_skip;
                out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
            }

            noc.async_write_barrier();

            dfb_out.pop_front(out_subblock_tile_count);
            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
        }
        // Pop fully padded subblocks along the row
        if (bw == num_blocks_w_dim_ - 1) {
            dfb_out.wait_front(static_cast<uint16_t>(wargs.padded_block_tiles_w_skip));
            dfb_out.pop_front(static_cast<uint16_t>(wargs.padded_block_tiles_w_skip));
        }
        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
    }
    // Pop row(s) of fully padded subblocks
    if (bh == num_blocks_h_dim_ - 1) {
        dfb_out.wait_front(static_cast<uint16_t>(wargs.padded_block_tiles_h_skip));
        dfb_out.pop_front(static_cast<uint16_t>(wargs.padded_block_tiles_h_skip));
    }
}
