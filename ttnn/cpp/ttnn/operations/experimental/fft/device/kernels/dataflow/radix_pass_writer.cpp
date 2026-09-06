// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// radix_pass_writer.cpp — BRISC1 / writer for ttnn::prim::fft_radix_pass.
//
// Identical to batch_fft_writer.cpp PLUS, when APPLY_POST_TWIDDLE=1,
// an in-place scalar fp32 complex-multiply of STATE_R/I against the
// post-twiddle tile pre-loaded by the reader into CB_PT_R/I.  We do
// the cmul HERE (on BRISC1, the same RISC that issues the subsequent
// noc_async_write_page) instead of on the reader because:
//   * BRISC1's scalar L1 stores are trivially visible to BRISC1's
//     subsequent NoC reads (single-thread, single-pipeline).
//   * Doing the cmul on BRISC0 (reader) and then having BRISC1 read
//     STATE turns out to be flaky for some core placements — the
//     late scalar stores aren't always visible to the writer's NoC
//     read in time (observed: row 12 / grid (4,1) on WH 8×8).
//
// Named compile-time arg: sub_n. OUTPUT_BF16, APPLY_POST_TWIDDLE and
// APPLY_SCALE are defines. Named runtime args carry the row range and the
// uint32_t bit-pattern of output_scale.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "radix_pass_common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t base_tile_idx = get_arg(args::base_tile_idx);
    const uint32_t batch_per_core = get_arg(args::batch_per_core);

    constexpr uint32_t SUB_N = get_arg(args::sub_n);

    const auto out_r_gen = TensorAccessor(tensor::out_r);
    const auto out_i_gen = TensorAccessor(tensor::out_i);

    Noc noc;
    DataflowBuffer cb_sync(dfb::sync);
    DataflowBuffer cb_state_r(dfb::state_r);
    DataflowBuffer cb_state_i(dfb::state_i);
#ifdef APPLY_POST_TWIDDLE
    DataflowBuffer cb_pt_r(dfb::post_twiddle_r);
    DataflowBuffer cb_pt_i(dfb::post_twiddle_i);
#endif
#ifdef OUTPUT_BF16
    DataflowBuffer cb_out_r_bf16(dfb::out_r_bf16);
    DataflowBuffer cb_out_i_bf16(dfb::out_i_bf16);
#endif

    for (uint32_t k = 0; k < batch_per_core; ++k) {
        const uint32_t tile_idx = base_tile_idx + k;

        cb_sync.wait_front(1);
        cb_state_r.wait_front(1);
        cb_state_i.wait_front(1);

        const uint32_t state_r_l1 = cb_state_r.get_read_ptr();
        const uint32_t state_i_l1 = cb_state_i.get_read_ptr();

        // ── Optional post-twiddle scalar cmul (on this BRISC) ────────────
#ifdef APPLY_POST_TWIDDLE
        {
            cb_pt_r.wait_front(1);
            cb_pt_i.wait_front(1);

            volatile tt_l1_ptr float* const sr = reinterpret_cast<volatile tt_l1_ptr float*>(state_r_l1);
            volatile tt_l1_ptr float* const si = reinterpret_cast<volatile tt_l1_ptr float*>(state_i_l1);
            volatile tt_l1_ptr float* const pr = reinterpret_cast<volatile tt_l1_ptr float*>(cb_pt_r.get_read_ptr());
            volatile tt_l1_ptr float* const pi = reinterpret_cast<volatile tt_l1_ptr float*>(cb_pt_i.get_read_ptr());
            for (uint32_t i = 0; i < SUB_N; ++i) {
                const float a = sr[i];
                const float b = si[i];
                const float c = pr[i];
                const float d = pi[i];
                sr[i] = a * c - b * d;
                si[i] = a * d + b * c;
            }

            cb_pt_r.pop_front(1);
            cb_pt_i.pop_front(1);
        }
#endif

        // ── Optional output scale (IFFT 1/N fold, an earlier iteration) ────────────
        //   Applied AFTER any post-twiddle (so it commutes with the cmul
        //   above — scaling a complex number doesn't change the order of
        //   operations) and BEFORE the bf16 truncation (so we don't lose
        //   precision in the scale itself).  Runs in fp32 on BRISC1 just
        //   like the post-twiddle loop above; total cost ≈ SUB_N extra
        //   fp32 muls per row.
        //
        //   The runtime arg fetch + bit-cast LIVE INSIDE this constexpr
        //   block so that the no-scale path's BRISC1 instruction stream
        //   is bit-identical to an earlier iteration — protects against any subtle
        //   timing / L1-stack-layout regression on the unchanged FFT
        //   path.
#ifdef APPLY_SCALE
        {
            // Bit-cast uint32_t → float via union (strict-aliasing safe).
            union {
                uint32_t u;
                float f;
            } scale_u;
            scale_u.u = get_arg(args::output_scale_bits);

            const float output_scale = scale_u.f;
            volatile tt_l1_ptr float* const sr = reinterpret_cast<volatile tt_l1_ptr float*>(state_r_l1);
            volatile tt_l1_ptr float* const si = reinterpret_cast<volatile tt_l1_ptr float*>(state_i_l1);
            for (uint32_t i = 0; i < SUB_N; ++i) {
                sr[i] = sr[i] * output_scale;
                si[i] = si[i] * output_scale;
            }
        }
#endif

#ifdef OUTPUT_BF16
        {
            // fp32 STATE → bf16 in CB_OUT_*_BF16, then DMA bf16 tile.
            cb_out_r_bf16.reserve_back(1);
            cb_out_i_bf16.reserve_back(1);
            const uint32_t out_r_bf16_l1 = cb_out_r_bf16.get_write_ptr();
            const uint32_t out_i_bf16_l1 = cb_out_i_bf16.get_write_ptr();

            volatile tt_l1_ptr uint32_t* const src_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_r_l1);
            volatile tt_l1_ptr uint32_t* const src_i = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_i_l1);
            volatile tt_l1_ptr uint16_t* const dst_r = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const dst_i = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_i_bf16_l1);
            for (uint32_t i = 0; i < SUB_N; ++i) {
                dst_r[i] = static_cast<uint16_t>(src_r[i] >> 16);
                dst_i[i] = static_cast<uint16_t>(src_i[i] >> 16);
            }
            cb_out_r_bf16.push_back(1);
            cb_out_i_bf16.push_back(1);

            noc.async_write(cb_out_r_bf16, out_r_gen, out_r_gen.get_aligned_page_size(), {}, {.page_id = tile_idx});
            noc.async_write(cb_out_i_bf16, out_i_gen, out_i_gen.get_aligned_page_size(), {}, {.page_id = tile_idx});
            noc.async_write_barrier();

            cb_out_r_bf16.pop_front(1);
            cb_out_i_bf16.pop_front(1);
        }
#else
        {
            noc.async_write(cb_state_r, out_r_gen, out_r_gen.get_aligned_page_size(), {}, {.page_id = tile_idx});
            noc.async_write(cb_state_i, out_i_gen, out_i_gen.get_aligned_page_size(), {}, {.page_id = tile_idx});
            noc.async_write_barrier();
        }
#endif

        cb_sync.pop_front(1);
        cb_state_r.pop_front(1);
        cb_state_i.pop_front(1);
    }
}
