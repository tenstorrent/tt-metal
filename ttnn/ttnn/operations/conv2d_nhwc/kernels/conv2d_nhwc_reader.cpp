// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc reader (NCRISC) — implicit im2col gather.
//
// The im2col matrix is never materialized in DRAM. For each K-block (one
// (tap, channel-slice) pair) and each output-row tile, this kernel issues 32
// NoC reads — one per output position — each pulling `stick_bytes` contiguous
// channel values out of the NHWC row-major activation. Positions whose input
// coordinate falls outside the padded image (or which are M-padding beyond
// M_total) read from a zero-filled scratch page instead.
//
// Pushes `cb_act_rm` in units of 32 row-major sticks; the compute kernel's
// tilize hook consumes them as one Kb-wide tile-row (asymmetric page mode).
//
// Multi-core: the M-block index space is split across the grid, so the outer
// loop bound is a runtime arg (`num_m_blocks_here`) and the loop is offset by
// this core's `start_m_block`. Everything else is grid-uniform.
//
// Channel alignment (Refinement 3): the gather window per tap is `Ct*32`
// channels, where `Ct = ceil(chans_cb / 32)`. When the real channel run is not
// a multiple of 32 the last channel-slice of each tap runs past the end of the
// activation stick, so this kernel reads only the bytes the stick actually
// owns and zero-fills the rest of the L1 stick. The matching weight K-rows are
// zero (prepare_conv2d_weights pads them), so those lanes contribute 0*0 to the
// matmul; zeroing the activation side too keeps a NaN/Inf bit pattern in the
// stick's alignment padding from poisoning the product. When the channel run IS
// tile-aligned, `has_chan_tail` is false and every masking branch below
// compiles out.
//
// Grouped / depthwise (Refinement 4): this is the one kernel where the im2col
// gather depends on *which output columns* are being computed. The C_out axis
// is partitioned into column blocks of `n_sub_per_cblock` consecutive N-blocks;
// column block `cblock` reads the channel window
// `[cblock*chans_cb, cblock*chans_cb + Ct*32)` of every activation stick. For
// dense conv there is exactly one column block and the base offset is always
// 0, so the coupling costs a single integer divide outside the inner loops.
// The block-diagonal structure *inside* a column block lives entirely in the
// prepared weight (zeros for channels outside a column's own group), so
// nothing about the gather changes per column.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

// Zero `nbytes` of L1 starting at `addr`. `addr` may be 2-byte aligned (bf16
// activation with an odd channel count); the end is always 4-byte aligned
// because stick_bytes is a multiple of 64.
inline void zero_l1_range(uint32_t addr, uint32_t nbytes) {
    while (nbytes != 0 && (addr & 3u) != 0) {
        *reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr) = 0;
        ++addr;
        --nbytes;
    }
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    const uint32_t nwords = nbytes >> 2;
    for (uint32_t i = 0; i < nwords; ++i) {
        w[i] = 0;
    }
    uint32_t tail = nbytes & 3u;
    volatile tt_l1_ptr uint8_t* b = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr + (nwords << 2));
    for (uint32_t i = 0; i < tail; ++i) {
        b[i] = 0;
    }
}

}  // namespace

void kernel_main() {
    // ---------------- compile-time args ----------------
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t H_out = get_compile_time_arg_val(2);
    constexpr uint32_t W_out = get_compile_time_arg_val(3);
    constexpr uint32_t M_total = get_compile_time_arg_val(4);
    constexpr uint32_t Mt = get_compile_time_arg_val(5);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t sub_per_tap = get_compile_time_arg_val(8);
    constexpr uint32_t kernel_size = get_compile_time_arg_val(9);
    constexpr uint32_t padding = get_compile_time_arg_val(10);
    constexpr uint32_t stride = get_compile_time_arg_val(11);
    constexpr uint32_t dilation = get_compile_time_arg_val(12);
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(13);       // Kb*32 channels * elem_size
    constexpr uint32_t c_in_bytes = get_compile_time_arg_val(14);        // C_in * elem_size (real bytes/stick)
    constexpr uint32_t chans_cb_bytes = get_compile_time_arg_val(15);    // per-column-block channel base stride
    constexpr uint32_t n_sub_per_cblock = get_compile_time_arg_val(16);  // N-blocks per column block
    // True iff any (column block, channel-slice) gather window overhangs the
    // real channel axis. Everything guarded by it compiles out otherwise.
    constexpr bool has_chan_tail = get_compile_time_arg_val(17) == 1;

    constexpr auto act_args = TensorAccessorArgs<18>();

    constexpr uint32_t cb_act_rm = 0;
    constexpr uint32_t cb_zero_scratch = 3;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t HW_out = H_out * W_out;

    // ---------------- runtime args ----------------
    // The M-block range is per-core (grid split) and therefore runtime, not CT.
    const uint32_t act_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_m_block = get_arg_val<uint32_t>(1);
    const uint32_t num_m_blocks_here = get_arg_val<uint32_t>(2);

    const auto act = TensorAccessor(act_args, act_base_addr);

    // ---------------- zero page for out-of-bounds taps ----------------
    const uint32_t zero_l1 = get_write_ptr(cb_zero_scratch);
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
        for (uint32_t i = 0; i < stick_bytes / 4; ++i) {
            p[i] = 0;
        }
    }
    const uint64_t zero_noc = get_noc_addr(zero_l1);

    // ---------------- gather loop ----------------
    // Every m_block covers a FULL Mt tile-rows, even the tail one: metal CBs cannot
    // pop/reserve a block that straddles fifo_limit, so a short tail block would drift
    // every downstream CB off its size-aligned boundary. Rows past M_total are zeros.
    for (uint32_t mb = 0; mb < num_m_blocks_here; ++mb) {
        const uint32_t mt_base = (start_m_block + mb) * Mt;

        for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
            // Which group bundle these output columns belong to (Refinement 4).
            // Collapses to 0 for dense conv (n_sub_per_cblock == num_n_blocks).
            const uint32_t cblock_base_bytes = (n_block / n_sub_per_cblock) * chans_cb_bytes;

            for (uint32_t k_block = 0; k_block < num_k_blocks; ++k_block) {
                const uint32_t tap = k_block / sub_per_tap;
                const uint32_t sub = k_block - tap * sub_per_tap;
                const uint32_t kh = tap / kernel_size;
                const uint32_t kw = tap - kh * kernel_size;
                // Byte offset of this K-block's channel run inside a stick.
                // Both terms are multiples of 64, so the NoC source keeps the
                // same alignment residue as the 64-aligned L1 write pointer.
                const uint32_t chan_off_bytes = cblock_base_bytes + sub * stick_bytes;

                // How much of this K-block's stick is real activation data.
                uint32_t valid_bytes = stick_bytes;
                uint32_t tail_bytes = 0;
                if constexpr (has_chan_tail) {
                    const uint32_t remaining = (c_in_bytes > chan_off_bytes) ? (c_in_bytes - chan_off_bytes) : 0;
                    valid_bytes = (remaining < stick_bytes) ? remaining : stick_bytes;
                    tail_bytes = stick_bytes - valid_bytes;
                }

                for (uint32_t t = 0; t < Mt; ++t) {
                    cb_reserve_back(cb_act_rm, TILE_H);
                    uint32_t wptr = get_write_ptr(cb_act_rm);

                    const uint32_t m_row_base = (mt_base + t) * TILE_H;
                    for (uint32_t r = 0; r < TILE_H; ++r) {
                        const uint32_t m = m_row_base + r;
                        bool valid = (m < M_total);
                        uint32_t page_id = 0;
                        if (valid) {
                            const uint32_t n = m / HW_out;
                            const uint32_t rem = m - n * HW_out;
                            const uint32_t ho = rem / W_out;
                            const uint32_t wo = rem - ho * W_out;

                            const int32_t hi = (int32_t)(ho * stride) + (int32_t)(dilation * kh) - (int32_t)padding;
                            const int32_t wi = (int32_t)(wo * stride) + (int32_t)(dilation * kw) - (int32_t)padding;

                            if (hi < 0 || wi < 0 || (uint32_t)hi >= H || (uint32_t)wi >= W) {
                                valid = false;
                            } else {
                                page_id = (n * H + (uint32_t)hi) * W + (uint32_t)wi;
                            }
                        }

                        if constexpr (has_chan_tail) {
                            // A ragged last column block can leave a K-block
                            // with no real channels at all; then the stick is
                            // pure zero-fill and there is nothing to read.
                            if (valid_bytes == 0) {
                                valid = false;
                            }
                        }

                        if (valid) {
                            // The NoC read and the tail zero-fill target
                            // disjoint byte ranges of the same stick, so the
                            // in-flight read cannot race the stores.
                            noc_async_read(act.get_noc_addr(page_id, chan_off_bytes), wptr, valid_bytes);
                            if constexpr (has_chan_tail) {
                                if (tail_bytes != 0) {
                                    zero_l1_range(wptr + valid_bytes, tail_bytes);
                                }
                            }
                        } else {
                            // Out-of-image tap (or M-padding row): the whole
                            // stick is zero, tail included.
                            noc_async_read(zero_noc, wptr, stick_bytes);
                        }
                        wptr += stick_bytes;
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_act_rm, TILE_H);
                }
            }
        }
    }
}
