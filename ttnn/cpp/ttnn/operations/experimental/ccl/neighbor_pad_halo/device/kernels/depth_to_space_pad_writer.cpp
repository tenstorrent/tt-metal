// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Fused depth-to-space + pad, writing DIRECTLY into a padded output buffer's INTERIOR (no dense
// intermediate, no separate pad copy). Input is a conv output [B,T,H,W, p1*p2*p3*C] row-major with
// channel order (p1,p2,p3,C); output is padded [B, T_out, H*p2+2pH, W*p3+2pW, C] with
// T_out = T*p1 - drop_first. Only the interior is written; the border is left for a later
// neighbor_pad_halo + halo_scatter(border_only) to fill (same contract as conv padded-output).
//
//   out[b, (t*p1+a)-drop, pH + h*p2+j, pW + w*p3+k, c]
//       = in[b, t, h, w, ((a*p2+j)*p3+k)*C + c]
//
// The depth-to-space is the inverse of _depth_to_space_channels_last() in vae_ltx.py (reshape to
// (B,T,H,W,p1,p2,p3,C), permute (0,1,4,2,5,3,6,7), reshape). Each output interior stick is C
// contiguous elements read as a sub-slice of the larger input page (offset slice_idx*c_bytes within
// it). Parallelized by a single global output-interior stick index; each core owns a contiguous
// [start, start+count) range. Throughput comes from BATCHING async reads/writes (one barrier/batch)
// since DRAM pages are bank-distributed.
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);   // conv output [B,T,H,W,p1*p2*p3*C]
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);  // padded output (interior written)
    const uint32_t stick_start = get_arg_val<uint32_t>(2);
    const uint32_t stick_count = get_arg_val<uint32_t>(3);

    constexpr uint32_t c_bytes = get_compile_time_arg_val(0);        // C * elem_size (data moved per stick)
    constexpr uint32_t in_page_size = get_compile_time_arg_val(1);   // input accessor page (aligned p1p2p3C row)
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(2);  // dst accessor page (aligned C row) = L1 slot
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t T = get_compile_time_arg_val(4);  // input frames
    constexpr uint32_t H = get_compile_time_arg_val(5);  // input rows (per device)
    constexpr uint32_t W = get_compile_time_arg_val(6);  // input cols (per device)
    constexpr uint32_t p1 = get_compile_time_arg_val(7);
    constexpr uint32_t p2 = get_compile_time_arg_val(8);
    constexpr uint32_t p3 = get_compile_time_arg_val(9);
    constexpr uint32_t pH = get_compile_time_arg_val(10);
    constexpr uint32_t pW = get_compile_time_arg_val(11);
    constexpr uint32_t drop_first = get_compile_time_arg_val(12);  // 1 if first output frame dropped
    constexpr uint32_t cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t batch = get_compile_time_arg_val(14);

    constexpr auto in_args = TensorAccessorArgs<15>();
    constexpr auto dst_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto in_acc = TensorAccessor(in_args, in_addr, in_page_size);
    const auto dst_acc = TensorAccessor(dst_args, dst_addr, dst_page_size);

    constexpr uint32_t Hd_out = H * p2;  // interior output rows
    constexpr uint32_t Wd_out = W * p3;  // interior output cols
    constexpr uint32_t Hp_out = Hd_out + 2 * pH;
    constexpr uint32_t Wp_out = Wd_out + 2 * pW;
    constexpr uint32_t T_out = T * p1 - drop_first;
    constexpr uint32_t frame_stride = Hp_out * Wp_out;  // output pages per frame

    const uint32_t l1_base = get_write_ptr(cb_id);
    const uint32_t end = stick_start + stick_count;

    for (uint32_t gi = stick_start; gi < end; gi += batch) {
        const uint32_t n = (end - gi < batch) ? (end - gi) : batch;
        // Batch of reads: input C-slice -> L1 ring slots.
        for (uint32_t kk = 0; kk < n; kk++) {
            const uint32_t idx = gi + kk;
            const uint32_t frame_g = idx / (Hd_out * Wd_out);
            const uint32_t rem = idx - frame_g * (Hd_out * Wd_out);
            const uint32_t h_out = rem / Wd_out;
            const uint32_t w_out = rem - h_out * Wd_out;
            const uint32_t b = frame_g / T_out;
            const uint32_t frame_out = frame_g - b * T_out;
            const uint32_t actual_frame = frame_out + drop_first;
            const uint32_t t = actual_frame / p1;
            const uint32_t a = actual_frame - t * p1;
            const uint32_t h = h_out / p2;
            const uint32_t j = h_out - h * p2;
            const uint32_t w = w_out / p3;
            const uint32_t k = w_out - w * p3;
            const uint32_t slice_idx = (a * p2 + j) * p3 + k;
            const uint32_t in_page = ((b * T + t) * H + h) * W + w;
            const uint64_t src_noc = get_noc_addr(in_page, in_acc) + slice_idx * c_bytes;
            noc_async_read(src_noc, l1_base + kk * dst_page_size, c_bytes);
        }
        noc_async_read_barrier();
        // Batch of writes: L1 ring slots -> padded interior pages.
        for (uint32_t kk = 0; kk < n; kk++) {
            const uint32_t idx = gi + kk;
            const uint32_t frame_g = idx / (Hd_out * Wd_out);
            const uint32_t rem = idx - frame_g * (Hd_out * Wd_out);
            const uint32_t h_out = rem / Wd_out;
            const uint32_t w_out = rem - h_out * Wd_out;
            const uint32_t dst_page = frame_g * frame_stride + (pH + h_out) * Wp_out + (pW + w_out);
            noc_async_write(l1_base + kk * dst_page_size, get_noc_addr(dst_page, dst_acc), c_bytes);
        }
        noc_async_write_barrier();
    }
}
