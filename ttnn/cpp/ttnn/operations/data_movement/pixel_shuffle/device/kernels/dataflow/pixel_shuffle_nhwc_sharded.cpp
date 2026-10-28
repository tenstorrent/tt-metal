// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_shuffle, channels_last: input stick j of this core's shard is pixel p = pix0 + j = (n, h, w) with C*r^2
// channels; its r^2 output pixels are (n, h*r + rh, w*r + rw), output row base(n, h, w) + rh*(W*r) + rw.
//   fast (SPATIAL_MAJOR, no padding): for every rh the r chunks rw = 0..r-1 are contiguous in the stick
//     (offset (rh*r + rw)*C) and go to r consecutive output sticks: one NoC write of r*C elements.
//   general: every output stick is assembled in a scratch slot (C elements, then zero padding) and written
//     with one NoC write; CHANNEL_MAJOR gathers element c*r^2 + rh*r + rw for c = 0..C-1 (stride r^2),
//     SPATIAL_MAJOR copies the contiguous chunk.
// The two dataflow RISCs take the two halves of the shard's sticks.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t r = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3);
    constexpr uint32_t Cp = get_compile_time_arg_val(4);
    constexpr uint32_t datum = get_compile_time_arg_val(5);
    constexpr uint32_t in_stick_nbytes = get_compile_time_arg_val(6);
    constexpr uint32_t out_stick_nbytes = get_compile_time_arg_val(7);
    constexpr uint32_t cb_in = get_compile_time_arg_val(8);
    constexpr uint32_t half = get_compile_time_arg_val(9);
    constexpr uint32_t channel_order = get_compile_time_arg_val(10);
    constexpr uint32_t fast = get_compile_time_arg_val(11);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(12);
    constexpr uint32_t scratch_slots = get_compile_time_arg_val(13);
    constexpr auto dst_args = TensorAccessorArgs<14, 0>();
    constexpr uint32_t SPATIAL_MAJOR = 1;
    constexpr uint32_t r2 = r * r;
    constexpr uint32_t out_W = W * r;
    constexpr uint32_t out_HW = H * r * out_W;
    constexpr uint32_t HW = H * W;
    constexpr uint32_t run_nbytes = r * C * datum;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t pix0 = get_arg_val<uint32_t>(1);
    const uint32_t npix = get_arg_val<uint32_t>(2);
    const uint32_t j_begin = half == 0 ? 0 : npix / 2;
    const uint32_t j_end = half == 0 ? npix / 2 : npix;
    if (j_begin >= j_end) {
        return;
    }
    const auto dst = TensorAccessor(dst_args, dst_addr);
    const uint32_t in_base = get_read_ptr(cb_in);

    uint32_t scratch_base = 0;
    if constexpr (!fast) {
        scratch_base = get_write_ptr(cb_scratch);
        // zero the padding tail of every slot once; the C data elements are rewritten per stick
        for (uint32_t sl = 0; sl < scratch_slots; sl++) {
            tt_l1_ptr uint8_t* pad = (tt_l1_ptr uint8_t*)(scratch_base + sl * out_stick_nbytes + C * datum);
            for (uint32_t b = 0; b < out_stick_nbytes - C * datum; b++) {
                pad[b] = 0;
            }
        }
    }

    uint32_t p = pix0 + j_begin;
    uint32_t n = p / HW;
    uint32_t rem = p - n * HW;
    uint32_t h = rem / W;
    uint32_t w = rem - h * W;
    uint32_t slot = 0;
    for (uint32_t j = j_begin; j < j_end; ++j) {
        const uint32_t src = in_base + j * in_stick_nbytes;
        uint32_t out_row = n * out_HW + (h * r) * out_W + w * r;
        if constexpr (fast) {
            uint32_t s = src;
            for (uint32_t dy = 0; dy < r; ++dy) {
                noc_async_write(s, dst.get_noc_addr(out_row), run_nbytes);
                s += run_nbytes;
                out_row += out_W;
            }
        } else {
            for (uint32_t rh = 0; rh < r; ++rh) {
                for (uint32_t rw = 0; rw < r; ++rw) {
                    const uint32_t d = scratch_base + slot * out_stick_nbytes;
                    if constexpr (channel_order == SPATIAL_MAJOR) {
                        tt_l1_ptr uint32_t* s32 = (tt_l1_ptr uint32_t*)(src + (rh * r + rw) * C * datum);
                        tt_l1_ptr uint32_t* d32 = (tt_l1_ptr uint32_t*)d;
                        for (uint32_t k = 0; k < C * datum / 4; k++) {
                            d32[k] = s32[k];
                        }
                    } else if constexpr (datum == 2) {
                        tt_l1_ptr uint16_t* s16 = (tt_l1_ptr uint16_t*)src;
                        tt_l1_ptr uint16_t* d16 = (tt_l1_ptr uint16_t*)d;
                        uint32_t si = rh * r + rw;
                        for (uint32_t c = 0; c < C; c++) {
                            d16[c] = s16[si];
                            si += r2;
                        }
                    } else {
                        tt_l1_ptr uint32_t* s32 = (tt_l1_ptr uint32_t*)src;
                        tt_l1_ptr uint32_t* d32 = (tt_l1_ptr uint32_t*)d;
                        uint32_t si = rh * r + rw;
                        for (uint32_t c = 0; c < C; c++) {
                            d32[c] = s32[si];
                            si += r2;
                        }
                    }
                    noc_async_write(d, dst.get_noc_addr(out_row + rh * out_W + rw), out_stick_nbytes);
                    if (++slot == scratch_slots) {  // all slots issued: drain before reusing the first one
                        noc_async_write_barrier();
                        slot = 0;
                    }
                }
            }
        }
        if (++w == W) {
            w = 0;
            if (++h == H) {
                h = 0;
                ++n;
            }
        }
    }
    noc_async_write_barrier();
}
