// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

inline uint32_t out_writer_wall_clock() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

struct DeferredBlockWriter {
    DataflowBuffer* od = nullptr;
    bool active = false;
    bool valid = true;
    bool w_last = false;
    bool h_last = false;
    bool h_pop = false;
    uint32_t nsub_h = 0, nsub_w = 0, sbh = 0, sbw = 0;
    uint32_t sbh_start = 0, sbw_start = 0;
    uint32_t last_sub_h = 0, last_sub_w = 0, addr_skip = 0, pad_w_skip = 0, pad_h_skip = 0;
    uint32_t tile_bytes = 0;

    void arm(
        DataflowBuffer* od_,
        bool valid_,
        uint32_t start_tile_id,
        uint32_t nsub_h_,
        uint32_t nsub_w_,
        bool w_last_,
        bool h_last_,
        bool h_pop_,
        uint32_t last_sub_h_,
        uint32_t last_sub_w_,
        uint32_t addr_skip_,
        uint32_t pad_w_skip_,
        uint32_t pad_h_skip_,
        uint32_t tile_bytes_) {
        od = od_;
        valid = valid_;
        nsub_h = nsub_h_;
        nsub_w = nsub_w_;
        w_last = w_last_;
        h_last = h_last_;
        h_pop = h_pop_;
        last_sub_h = last_sub_h_;
        last_sub_w = last_sub_w_;
        addr_skip = addr_skip_;
        pad_w_skip = pad_w_skip_;
        pad_h_skip = pad_h_skip_;
        tile_bytes = tile_bytes_;
        sbh = 0;
        sbw = 0;
        sbh_start = start_tile_id;
        sbw_start = start_tile_id;
        active = nsub_h > 0 && nsub_w > 0;
        if (!active) {
            finish_block();
        }
    }

    template <
        uint32_t out_subblock_h,
        uint32_t out_subblock_w,
        uint32_t out_subblock_tile_count,
        uint32_t stride_w,
        uint32_t stride_h,
        uint32_t next_sub_stride_w,
        uint32_t next_sub_stride_h,
        typename NocT,
        typename AccT>
    bool step(const NocT& noc, const AccT& s) {
        if (!active) {
            return false;
        }
        uint32_t sh = out_subblock_h;
        uint32_t sw = out_subblock_w;
        uint32_t skip = 0;
        if (h_last && sbh == nsub_h - 1) {
            sh = last_sub_h;
        }
        if (w_last && sbw == nsub_w - 1) {
            sw = last_sub_w;
            skip = addr_skip;
        }
        od->wait_front(out_subblock_tile_count);
        uint32_t off = 0;
        uint32_t row_start = sbw_start;
        for (uint32_t h = 0; h < sh; ++h) {
            uint32_t tid = row_start;
            for (uint32_t w = 0; w < sw; ++w) {
                if (valid) {
                    noc.async_write(*od, s, tile_bytes, {.offset_bytes = off}, {.page_id = tid});
                }
                off += tile_bytes;
                tid += stride_w;
            }
            off += skip;
            row_start += stride_h;
        }
#ifdef OUT_FLUSH_PER_BLOCK
        noc.async_writes_flushed();
#else
        noc.async_write_barrier();
#endif
        od->pop_front(out_subblock_tile_count);
        sbw_start += next_sub_stride_w;
        if (++sbw == nsub_w) {
            if (w_last) {
                od->wait_front(static_cast<uint16_t>(pad_w_skip));
                od->pop_front(static_cast<uint16_t>(pad_w_skip));
            }
            sbh_start += next_sub_stride_h;
            sbw_start = sbh_start;
            sbw = 0;
            if (++sbh == nsub_h) {
                finish_block();
            }
        }
        return true;
    }

    void finish_block() {
        if (h_pop) {
            od->wait_front(static_cast<uint16_t>(pad_h_skip));
            od->pop_front(static_cast<uint16_t>(pad_h_skip));
        }
        active = false;
    }
};

inline void write_output_blocks(const Noc& noc) {
    constexpr uint32_t num_blocks_w = get_arg(args::num_blocks_w_dim);
    constexpr uint32_t num_blocks_h = get_arg(args::num_blocks_h_dim);
    constexpr uint32_t sb_w = get_arg(args::out_subblock_w);
    constexpr uint32_t sb_h = get_arg(args::out_subblock_h);
    constexpr uint32_t sb_tiles = get_arg(args::out_subblock_tile_count);
    constexpr uint32_t stride_w = get_arg(args::out_tensor_stride_w);
    constexpr uint32_t stride_h = get_arg(args::out_tensor_stride_h);
    constexpr uint32_t next_sb_w = get_arg(args::out_tensor_next_subblock_stride_w);
    constexpr uint32_t next_sb_h = get_arg(args::out_tensor_next_subblock_stride_h);
    constexpr uint32_t next_block_w = get_arg(args::out_tensor_next_w_dim_block_stride);
    constexpr uint32_t next_block_h = get_arg(args::out_tensor_next_h_dim_block_stride);
    constexpr uint32_t pace_cycles = get_arg(args::out_write_pace_cycles);
    constexpr uint32_t pace_barrier = get_arg(args::out_write_pace_barrier);
    const uint32_t start = get_arg(args::out_tensor_start_tile_id);
    const uint32_t nsh = get_arg(args::out_num_nonzero_subblocks_h);
    const uint32_t last_nsh = get_arg(args::out_last_num_nonzero_subblocks_h);
    const uint32_t last_sh = get_arg(args::out_last_subblock_h);
    const uint32_t h_skip = get_arg(args::padded_block_tiles_h_skip);
    const uint32_t nsw = get_arg(args::out_num_nonzero_subblocks_w);
    const uint32_t last_nsw = get_arg(args::out_last_num_nonzero_subblocks_w);
    const uint32_t last_sw = get_arg(args::out_last_subblock_w);
    const uint32_t addr_skip = get_arg(args::padded_subblock_tiles_addr_skip);
    const uint32_t w_skip = get_arg(args::padded_block_tiles_w_skip);
    const uint32_t last_nbh = get_arg(args::last_num_blocks_h_dim);
    const uint32_t last_nbw = get_arg(args::last_num_blocks_w_dim);
#ifdef OUT_SWIGLU
    DataflowBuffer dfb_out(dfb::swiglu_out);
#else
    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_out_b(dfb::out_b);
#endif
    const auto s = TensorAccessor(tensor::out);
    const uint32_t tile_bytes = dfb_out.get_tile_size();
    DeferredBlockWriter w;
    uint32_t h_start = start;
    for (uint32_t bh = 0; bh < num_blocks_h; ++bh) {
        uint32_t w_start = h_start;
        const uint32_t nbh = bh >= last_nbh - 1 ? last_nbh : num_blocks_h;
        for (uint32_t bw = 0; bw < num_blocks_w; ++bw) {
            const uint32_t nbw = bw >= last_nbw - 1 ? last_nbw : num_blocks_w;
#ifdef OUT_SWIGLU
            DataflowBuffer* const cur = &dfb_out;
#else
            DataflowBuffer* const cur = (bw & 1u) ? &dfb_out_b : &dfb_out;
#endif
            w.arm(
                cur,
                bh < nbh && bw < nbw,
                w_start,
                bh == nbh - 1 ? last_nsh : nsh,
                bw == nbw - 1 ? last_nsw : nsw,
                bw == nbw - 1,
                bh == nbh - 1,
                bh == nbh - 1,
                last_sh,
                last_sw,
                addr_skip,
                w_skip,
                h_skip,
                tile_bytes);
            uint32_t paced = 0;
            const uint32_t pace_start = pace_cycles > 0 ? out_writer_wall_clock() : 0;
            while (w.step<sb_h, sb_w, sb_tiles, stride_w, stride_h, next_sb_w, next_sb_h>(noc, s)) {
                if constexpr (pace_cycles > 0) {
                    if (bw + 1 < num_blocks_w) {
                        if (++paced % pace_barrier == 0) {
                            noc.async_write_barrier();
                        }
                        while (out_writer_wall_clock() - pace_start < paced * pace_cycles) {
                        }
                    }
                }
            }
            w_start += next_block_w;
        }
        h_start += next_block_h;
    }
    noc.async_write_barrier();
}
