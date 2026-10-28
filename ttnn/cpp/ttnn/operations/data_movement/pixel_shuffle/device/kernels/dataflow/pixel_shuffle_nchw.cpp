// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_shuffle on NCHW ROW_MAJOR tensors (mirror of pixel_unshuffle_nchw.cpp). Runs on BOTH dataflow RISCs
// over disjoint output rows; each RISC does read -> interleave -> write with private L1 buffers.
//
// Work unit is one OUTPUT ROW p = (n, c, h_out): h_in = h_out / r, rh = h_out % r, and
//   out[n, c, h_out, w*r + rw] = in[n, c_in(rw), h_in, w]
//   CHANNEL_MAJOR: c_in(rw) = c*r^2 + rh*r + rw        -> input page stride over rw = H
//   SPATIAL_MAJOR: c_in(rw) = rh*(r*C) + rw*C + c      -> input page stride over rw = C*H
//   input page(rw) = n*C_in*H + c_in(rw)*H + h_in,  output page = p
// so each output stick is the interleave of r input sticks: out[w*r + rw] = in_rw[w].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t stick_nbytes_in = get_compile_time_arg_val(0);
    constexpr uint32_t aligned_stick_nbytes_in = get_compile_time_arg_val(1);
    constexpr uint32_t stick_nbytes_out = get_compile_time_arg_val(2);
    constexpr uint32_t aligned_stick_nbytes_out = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_scratch = get_compile_time_arg_val(5);
    constexpr uint32_t r = get_compile_time_arg_val(6);
    constexpr uint32_t W = get_compile_time_arg_val(7);
    constexpr uint32_t C = get_compile_time_arg_val(8);
    constexpr uint32_t H = get_compile_time_arg_val(9);
    constexpr uint32_t channel_order = get_compile_time_arg_val(10);
    constexpr uint32_t depth = get_compile_time_arg_val(11);
    constexpr auto src_args = TensorAccessorArgs<12, 0>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset(), src_args.num_common_runtime_args()>();

    constexpr uint32_t SPATIAL_MAJOR = 1;
    constexpr uint32_t datum_nbytes = stick_nbytes_in / W;
    constexpr uint32_t r2 = r * r;
    constexpr uint32_t Ho = H * r;
    constexpr uint32_t C_in = C * r2;
    constexpr uint32_t CHo = C * Ho;
    constexpr uint32_t rw_page_stride = (channel_order == SPATIAL_MAJOR) ? (C * H) : H;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);
    const uint32_t row_stride = get_arg_val<uint32_t>(4);
    if (num_rows == 0) {
        return;
    }

    const auto s_in = TensorAccessor(src_args, src_addr);
    const auto s_out = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    experimental::CB cb_in(cb_id_in);
    experimental::CB cb_scratch(cb_id_scratch);
    const uint32_t in_base = cb_in.get_write_ptr();
    const uint32_t scratch_base = cb_scratch.get_write_ptr();

    // output row p -> first input page (rw = 0)
    auto in_page0 = [&](uint32_t p) -> uint32_t {
        const uint32_t n = p / CHo;
        const uint32_t rem = p - n * CHo;
        const uint32_t c = rem / Ho;
        const uint32_t h_out = rem - c * Ho;
        const uint32_t h_in = h_out / r;
        const uint32_t rh = h_out - h_in * r;
        uint32_t c_in0;
        if constexpr (channel_order == SPATIAL_MAJOR) {
            c_in0 = rh * (r * C) + c;  // + rw*C
        } else {
            c_in0 = c * r2 + rh * r;  // + rw
        }
        return n * (C_in * H) + c_in0 * H + h_in;
    };
    auto issue_reads = [&](uint32_t p, uint32_t slot) {
        const uint32_t page0 = in_page0(p);
        for (uint32_t rw = 0; rw < r; rw++) {
            noc.async_read(
                s_in, cb_in, stick_nbytes_in, {.page_id = page0 + rw * rw_page_stride},
                {.offset_bytes = (slot * r + rw) * aligned_stick_nbytes_in});
        }
    };

    uint32_t row = start_row;
    issue_reads(row, 0);  // prefetch row 0; row i+1 is issued while row i is copied (one row's reads in flight)
    for (uint32_t i = 0; i < num_rows; i++) {
        const uint32_t slot = i % depth;
        noc.async_read_barrier();  // row i's reads are the only ones outstanding
        if (i >= depth) {
            noc.async_write_barrier();  // the write that used this scratch slot has to be out before it is refilled
        }
        const uint32_t src_l1 = in_base + slot * r * aligned_stick_nbytes_in;
        const uint32_t dst_l1 = scratch_base + slot * aligned_stick_nbytes_out;
        asm volatile("" ::: "memory");

        if constexpr (datum_nbytes == 2 && (W % 2) == 0) {
            // 32-bit interleave. Input word in32[j][m] holds in_j[2m] (lo) and in_j[2m+1] (hi). Output group m
            // is the r words out32[r*m + t] = {out[2r*m + 2t] (lo), out[2r*m + 2t + 1] (hi)}; element
            // e = 2r*m + q (q < 2r) comes from input stick j = q % r, position 2m + q / r, i.e. half (q / r) of
            // in32[q % r][m]. One aligned load per input word, one aligned store per output word.
            tt_l1_ptr uint32_t* s32[r];
            for (uint32_t j = 0; j < r; j++) {
                s32[j] = (tt_l1_ptr uint32_t*)(src_l1 + j * aligned_stick_nbytes_in);
            }
            tt_l1_ptr uint32_t* d32 = (tt_l1_ptr uint32_t*)dst_l1;
            for (uint32_t m = 0; m < W / 2; m++) {
                uint32_t w[r];
#pragma GCC unroll 16
                for (uint32_t j = 0; j < r; j++) {
                    w[j] = s32[j][m];
                }
#pragma GCC unroll 16
                for (uint32_t t = 0; t < r; t++) {
                    const uint32_t ql = 2 * t;
                    const uint32_t qh = 2 * t + 1;
                    const uint32_t wl = w[ql % r];
                    const uint32_t lo = (ql / r) ? (wl >> 16) : (wl & 0xffffu);
                    const uint32_t wh = w[qh % r];
                    const uint32_t hi = (qh / r) ? (wh & 0xffff0000u) : (wh << 16);
                    d32[r * m + t] = lo | hi;
                }
            }
        } else if constexpr (datum_nbytes == 2) {
            tt_l1_ptr uint16_t* d16 = (tt_l1_ptr uint16_t*)dst_l1;
            for (uint32_t j = 0; j < r; j++) {
                tt_l1_ptr uint16_t* sj = (tt_l1_ptr uint16_t*)(src_l1 + j * aligned_stick_nbytes_in);
                uint32_t di = j;
                for (uint32_t k = 0; k < W; k++) {
                    d16[di] = sj[k];
                    di += r;
                }
            }
        } else {
            tt_l1_ptr uint32_t* d32 = (tt_l1_ptr uint32_t*)dst_l1;
            for (uint32_t j = 0; j < r; j++) {
                tt_l1_ptr uint32_t* sj = (tt_l1_ptr uint32_t*)(src_l1 + j * aligned_stick_nbytes_in);
                uint32_t di = j;
                for (uint32_t k = 0; k < W; k++) {
                    d32[di] = sj[k];
                    di += r;
                }
            }
        }

        if (i + 1 < num_rows) {
            issue_reads(row + row_stride, (slot + 1) % depth);  // overlaps the write below and the next copy
        }
        noc.async_write(
            use<experimental::CB::AddrSelector::WRITE_PTR>(cb_scratch), s_out, stick_nbytes_out,
            {.offset_bytes = slot * aligned_stick_nbytes_out}, {.page_id = row});
        row += row_stride;
    }
    noc.async_write_barrier();
}
