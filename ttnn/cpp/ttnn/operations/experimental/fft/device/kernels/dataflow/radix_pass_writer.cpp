// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// radix_pass_writer.cpp — BRISC1 / writer for ttnn::prim::fft_radix_pass.
//
// Identical to batch_fft_writer.cpp PLUS, when APPLY_POST_TWIDDLE=1,
// an in-place scalar fp32 complex-multiply of STATE_R/I against the
// post-twiddle tile pre-loaded by the reader into CB_PT_R/I.  We do
// the cmul HERE (on BRISC1, the same RISC that issues the subsequent
// noc_async_write_tile) instead of on the reader because:
//   * BRISC1's scalar L1 stores are trivially visible to BRISC1's
//     subsequent NoC reads (single-thread, single-pipeline).
//   * Doing the cmul on BRISC0 (reader) and then having BRISC1 read
//     STATE turns out to be flaky for some core placements — the
//     late scalar stores aren't always visible to the writer's NoC
//     read in time (observed: row 12 / grid (4,1) on WH 8×8).
//
// Compile-time args:
//   0: OUTPUT_BF16           — 0 fp32 fast path, 1 = bf16 trunc on output
//   1: SUB_N                 — number of valid floats per row (= P)
//   2: APPLY_POST_TWIDDLE    — 0 = pure FFT writer, 1 = fused cmul
//   3: APPLY_SCALE           — 0 = no output scale, 1 = multiply each
//                              STATE element by runtime arg 5 (used by
//                              the IFFT path to fold the 1/N scale).
//
// Runtime args:
//   5: output_scale_bits     — uint32_t bit-pattern of float, only read
//                              when APPLY_SCALE=1.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "radix_pass_common.h"

void kernel_main() {
    const uint32_t out_r_addr      = get_arg_val<uint32_t>(0);
    const uint32_t out_i_addr      = get_arg_val<uint32_t>(1);
    const uint32_t base_tile_idx   = get_arg_val<uint32_t>(2);
    const uint32_t batch_per_core  = get_arg_val<uint32_t>(3);
    const uint32_t out_page_size_override = get_arg_val<uint32_t>(4);

    constexpr uint32_t OUTPUT_BF16        = get_compile_time_arg_val(0);
    constexpr uint32_t SUB_N              = get_compile_time_arg_val(1);
    constexpr uint32_t APPLY_POST_TWIDDLE = get_compile_time_arg_val(2);
    constexpr uint32_t APPLY_SCALE        = get_compile_time_arg_val(3);

    const uint32_t ts = get_tile_size(CB_STATE_R);

    uint32_t fallback_ts;
    if constexpr (OUTPUT_BF16) {
        fallback_ts = get_tile_size(CB_OUT_R_BF16);
    } else {
        fallback_ts = ts;
    }
    const uint32_t out_ps = out_page_size_override ? out_page_size_override : fallback_ts;
    const InterleavedAddrGen<true> out_r_gen = {
            .bank_base_address = out_r_addr, .page_size = out_ps};
    const InterleavedAddrGen<true> out_i_gen = {
            .bank_base_address = out_i_addr, .page_size = out_ps};

    for (uint32_t k = 0; k < batch_per_core; ++k) {
        const uint32_t tile_idx = base_tile_idx + k;

        cb_wait_front(CB_SYNC, 1);
        cb_wait_front(CB_STATE_R, 1);
        cb_wait_front(CB_STATE_I, 1);

        const uint32_t state_r_l1 = get_read_ptr(CB_STATE_R);
        const uint32_t state_i_l1 = get_read_ptr(CB_STATE_I);

        // ── Optional post-twiddle scalar cmul (on this BRISC) ────────────
        if constexpr (APPLY_POST_TWIDDLE) {
            cb_wait_front(CB_PT_R, 1);
            cb_wait_front(CB_PT_I, 1);

            volatile tt_l1_ptr float* const sr =
                reinterpret_cast<volatile tt_l1_ptr float*>(state_r_l1);
            volatile tt_l1_ptr float* const si =
                reinterpret_cast<volatile tt_l1_ptr float*>(state_i_l1);
            volatile tt_l1_ptr float* const pr =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(CB_PT_R));
            volatile tt_l1_ptr float* const pi =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(CB_PT_I));
            for (uint32_t i = 0; i < SUB_N; ++i) {
                const float a = sr[i];
                const float b = si[i];
                const float c = pr[i];
                const float d = pi[i];
                sr[i] = a * c - b * d;
                si[i] = a * d + b * c;
            }

            cb_pop_front(CB_PT_R, 1);
            cb_pop_front(CB_PT_I, 1);
        }

        // ── Optional output scale (IFFT 1/N fold, commit 6c) ────────────
        //   Applied AFTER any post-twiddle (so it commutes with the cmul
        //   above — scaling a complex number doesn't change the order of
        //   operations) and BEFORE the bf16 truncation (so we don't lose
        //   precision in the scale itself).  Runs in fp32 on BRISC1 just
        //   like the post-twiddle loop above; total cost ≈ SUB_N extra
        //   fp32 muls per row.
        //
        //   The runtime arg fetch + bit-cast LIVE INSIDE this constexpr
        //   block so that the no-scale path's BRISC1 instruction stream
        //   is bit-identical to commit 5 — protects against any subtle
        //   timing / L1-stack-layout regression on the unchanged FFT
        //   path.
        if constexpr (APPLY_SCALE) {
            // Bit-cast uint32_t → float via union (strict-aliasing safe).
            union { uint32_t u; float f; } scale_u;
            scale_u.u = get_arg_val<uint32_t>(5);
            const float output_scale = scale_u.f;
            volatile tt_l1_ptr float* const sr =
                reinterpret_cast<volatile tt_l1_ptr float*>(state_r_l1);
            volatile tt_l1_ptr float* const si =
                reinterpret_cast<volatile tt_l1_ptr float*>(state_i_l1);
            for (uint32_t i = 0; i < SUB_N; ++i) {
                sr[i] = sr[i] * output_scale;
                si[i] = si[i] * output_scale;
            }
        }

        if constexpr (OUTPUT_BF16) {
            // fp32 STATE → bf16 in CB_OUT_*_BF16, then DMA bf16 tile.
            cb_reserve_back(CB_OUT_R_BF16, 1);
            cb_reserve_back(CB_OUT_I_BF16, 1);
            const uint32_t out_r_bf16_l1 = get_write_ptr(CB_OUT_R_BF16);
            const uint32_t out_i_bf16_l1 = get_write_ptr(CB_OUT_I_BF16);

            volatile tt_l1_ptr uint32_t* const src_r =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_r_l1);
            volatile tt_l1_ptr uint32_t* const src_i =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_i_l1);
            volatile tt_l1_ptr uint16_t* const dst_r =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const dst_i =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_i_bf16_l1);
            for (uint32_t i = 0; i < SUB_N; ++i) {
                dst_r[i] = static_cast<uint16_t>(src_r[i] >> 16);
                dst_i[i] = static_cast<uint16_t>(src_i[i] >> 16);
            }
            cb_push_back(CB_OUT_R_BF16, 1);
            cb_push_back(CB_OUT_I_BF16, 1);

            noc_async_write_tile(tile_idx, out_r_gen, out_r_bf16_l1);
            noc_async_write_tile(tile_idx, out_i_gen, out_i_bf16_l1);
            noc_async_write_barrier();

            cb_pop_front(CB_OUT_R_BF16, 1);
            cb_pop_front(CB_OUT_I_BF16, 1);
        } else {
            noc_async_write_tile(tile_idx, out_r_gen, state_r_l1);
            noc_async_write_tile(tile_idx, out_i_gen, state_i_l1);
            noc_async_write_barrier();
        }

        cb_pop_front(CB_SYNC, 1);
        cb_pop_front(CB_STATE_R, 1);
        cb_pop_front(CB_STATE_I, 1);
    }
}
