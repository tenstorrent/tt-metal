// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// radix_pass_reader.cpp — BRISC0 / reader for ttnn::prim::fft_radix_pass.
//
// Functionally identical to batch_fft_reader.cpp (same bit-reversal,
// same per-stage scatter/gather, same bf16 expansion at the DRAM
// boundary) PLUS — when APPLY_POST_TWIDDLE=1 — it ALSO loads the
// post-twiddle tile T[tile_idx % pt_modulus, :] into CB_PT_R/I so the
// writer (BRISC1) can perform the in-place complex-multiply RIGHT
// BEFORE issuing the noc_async_write_page out to DRAM.
//
// Why on the WRITER instead of here?  Doing the cmul here would write
// to STATE_R/I after we have already pushed them; cross-BRISC L1
// visibility for those late scalar stores is sensitive to the per-core
// L1 layout and observably flaky for some (row, core) combinations
// (e.g. row 12 on grid (4,1)).  Letting BRISC1 read, modify, and write
// STATE within a single thread closes that gap entirely — BRISC1's
// own scalar stores are guaranteed visible to its subsequent
// noc_async_write_page read of L1.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "radix_pass_common.h"
#include "experimental/kernel_args.h"

constexpr uint32_t kScalarCutoff = 64;

FORCE_INLINE void async_local_memcpy(
    Noc& noc, uint32_t src_l1, uint32_t dst_l1, uint32_t n_bytes, uint32_t my_noc_x, uint32_t my_noc_y) {
    CoreLocalMem<uint32_t> src(src_l1);
    UnicastEndpoint dst;
    noc.async_write(src, dst, n_bytes, {}, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = dst_l1});
}

void kernel_main() {
    const uint32_t base_tile_idx = get_arg(args::base_tile_idx);
    const uint32_t batch_per_core = get_arg(args::batch_per_core);
    const uint32_t my_noc_x = get_arg(args::noc_x);
    const uint32_t my_noc_y = get_arg(args::noc_y);
    // Post-twiddle tensors are tile-sized fp32 tables with layout
    //     identical to apply_twiddles_host::build_twiddle_table(P, N2):
    //     N2 tiles total, tile n2 holds T[n2, 0..P-1] in slots [0, P).
    //   pt_modulus = N2 (twiddle modulus on row index).
    //   pt_stride  = row-index stride before the modulus (default 1).
    //                Three-pass Pass-2 uses pt_stride=N3 so that the row
    //                enumeration (b, n1, k3) — laid out at stride N3 —
    //                picks the right n1 twiddle without an extra
    //                transpose.
    const uint32_t pt_modulus = get_arg(args::pt_modulus);
    const uint32_t pt_stride = get_arg(args::pt_stride);

    constexpr uint32_t SUB_N = get_arg(args::sub_n);
    constexpr uint32_t LOG2_SUB_N = get_arg(args::log2_sub_n);
    constexpr uint32_t LOCAL_PAIRS = SUB_N / 2;
    constexpr uint32_t BIT_REVERSE_ON_LOAD = get_arg(args::bit_reverse_on_load);

    const auto in_r_gen = TensorAccessor(tensor::in_r);
    const auto in_i_gen = TensorAccessor(tensor::in_i);

    const auto tw_r_gen = TensorAccessor(tensor::tw_r);
    const auto tw_i_gen = TensorAccessor(tensor::tw_i);

    Noc noc;
    DataflowBuffer cb_even_r(dfb::even_r);
    DataflowBuffer cb_even_i(dfb::even_i);
    DataflowBuffer cb_odd_r(dfb::odd_r);
    DataflowBuffer cb_odd_i(dfb::odd_i);
    DataflowBuffer cb_tw_r(dfb::twiddle_r);
    DataflowBuffer cb_tw_i(dfb::twiddle_i);
    DataflowBuffer cb_out0_r(dfb::out0_r);
    DataflowBuffer cb_out0_i(dfb::out0_i);
    DataflowBuffer cb_out1_r(dfb::out1_r);
    DataflowBuffer cb_out1_i(dfb::out1_i);
    DataflowBuffer cb_state_r(dfb::state_r);
    DataflowBuffer cb_state_i(dfb::state_i);
    DataflowBuffer cb_sync(dfb::sync);
#ifdef INPUT_BF16
    DataflowBuffer cb_in_r_bf16(dfb::in_r_bf16);
    DataflowBuffer cb_in_i_bf16(dfb::in_i_bf16);
#endif
#ifdef APPLY_POST_TWIDDLE
    DataflowBuffer cb_pt_r(dfb::post_twiddle_r);
    DataflowBuffer cb_pt_i(dfb::post_twiddle_i);
#endif

    for (uint32_t k = 0; k < batch_per_core; ++k) {
        const uint32_t tile_idx = base_tile_idx + k;

        cb_state_r.reserve_back(1);
        cb_state_i.reserve_back(1);
        const uint32_t state_r_l1 = cb_state_r.get_write_ptr();
        const uint32_t state_i_l1 = cb_state_i.get_write_ptr();

#ifdef INPUT_BF16
        {
            cb_in_r_bf16.reserve_back(1);
            cb_in_i_bf16.reserve_back(1);
            const uint32_t in_r_bf16_l1 = cb_in_r_bf16.get_write_ptr();
            const uint32_t in_i_bf16_l1 = cb_in_i_bf16.get_write_ptr();
            noc.async_read(in_r_gen, cb_in_r_bf16, in_r_gen.get_aligned_page_size(), {.page_id = tile_idx}, {});
            noc.async_read(in_i_gen, cb_in_i_bf16, in_i_gen.get_aligned_page_size(), {.page_id = tile_idx}, {});
            noc.async_read_barrier();
            cb_in_r_bf16.push_back(1);
            cb_in_i_bf16.push_back(1);

            volatile tt_l1_ptr uint16_t* const sb_r = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const sb_i = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_i_bf16_l1);
            volatile tt_l1_ptr uint32_t* const dst_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_r_l1);
            volatile tt_l1_ptr uint32_t* const dst_i = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_i_l1);
            for (uint32_t kk = 0; kk < SUB_N; ++kk) {
                dst_r[kk] = static_cast<uint32_t>(sb_r[kk]) << 16;
                dst_i[kk] = static_cast<uint32_t>(sb_i[kk]) << 16;
            }

            cb_in_r_bf16.pop_front(1);
            cb_in_i_bf16.pop_front(1);
        }
#else
        {
            noc.async_read(in_r_gen, cb_state_r, in_r_gen.get_aligned_page_size(), {.page_id = tile_idx}, {});
            noc.async_read(in_i_gen, cb_state_i, in_i_gen.get_aligned_page_size(), {.page_id = tile_idx}, {});
            noc.async_read_barrier();
        }
#endif

        cb_state_r.push_back(1);
        cb_state_i.push_back(1);

        volatile tt_l1_ptr float* const state_r = reinterpret_cast<volatile tt_l1_ptr float*>(state_r_l1);
        volatile tt_l1_ptr float* const state_i = reinterpret_cast<volatile tt_l1_ptr float*>(state_i_l1);

        if constexpr (BIT_REVERSE_ON_LOAD) {
            for (uint32_t kk = 0; kk < SUB_N; ++kk) {
                uint32_t br = 0;
                for (uint32_t b = 0; b < LOG2_SUB_N; ++b) {
                    br = (br << 1) | ((kk >> b) & 1u);
                }
                if (kk < br) {
                    float tr = state_r[kk];
                    state_r[kk] = state_r[br];
                    state_r[br] = tr;
                    float ti = state_i[kk];
                    state_i[kk] = state_i[br];
                    state_i[br] = ti;
                }
            }
        }

        // ── LOCAL Stockham stages 0..LOG2_SUB_N-1 ──────────────────────
        for (uint32_t s = 0; s < LOG2_SUB_N; ++s) {
            const uint32_t stride = 1u << s;
            const uint32_t group_size = stride << 1;
            const uint32_t mask = stride - 1;
            const uint32_t num_groups = LOCAL_PAIRS / stride;
            const uint32_t block_bytes = stride * 4u;
            const bool use_dma = block_bytes >= kScalarCutoff;

            cb_tw_r.reserve_back(1);
            cb_tw_i.reserve_back(1);
            noc.async_read(tw_r_gen, cb_tw_r, tw_r_gen.get_aligned_page_size(), {.page_id = s}, {});
            noc.async_read(tw_i_gen, cb_tw_i, tw_i_gen.get_aligned_page_size(), {.page_id = s}, {});
            noc.async_read_barrier();
            cb_tw_r.push_back(1);
            cb_tw_i.push_back(1);

            cb_even_r.reserve_back(1);
            cb_even_i.reserve_back(1);
            cb_odd_r.reserve_back(1);
            cb_odd_i.reserve_back(1);

            const uint32_t even_r_l1 = cb_even_r.get_write_ptr();
            const uint32_t even_i_l1 = cb_even_i.get_write_ptr();
            const uint32_t odd_r_l1 = cb_odd_r.get_write_ptr();
            const uint32_t odd_i_l1 = cb_odd_i.get_write_ptr();

            if (use_dma) {
                for (uint32_t g = 0; g < num_groups; ++g) {
                    const uint32_t src_even = state_r_l1 + (g * group_size) * 4u;
                    const uint32_t src_odd = state_r_l1 + (g * group_size + stride) * 4u;
                    const uint32_t src_evi = state_i_l1 + (g * group_size) * 4u;
                    const uint32_t src_odi = state_i_l1 + (g * group_size + stride) * 4u;
                    const uint32_t dst_even = even_r_l1 + (g * stride) * 4u;
                    const uint32_t dst_odd = odd_r_l1 + (g * stride) * 4u;
                    const uint32_t dst_evi = even_i_l1 + (g * stride) * 4u;
                    const uint32_t dst_odi = odd_i_l1 + (g * stride) * 4u;
                    async_local_memcpy(noc, src_even, dst_even, block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(noc, src_odd, dst_odd, block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(noc, src_evi, dst_evi, block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(noc, src_odi, dst_odi, block_bytes, my_noc_x, my_noc_y);
                }
                noc.async_write_barrier();
            } else {
                volatile tt_l1_ptr float* const even_r = reinterpret_cast<volatile tt_l1_ptr float*>(even_r_l1);
                volatile tt_l1_ptr float* const even_i = reinterpret_cast<volatile tt_l1_ptr float*>(even_i_l1);
                volatile tt_l1_ptr float* const odd_r = reinterpret_cast<volatile tt_l1_ptr float*>(odd_r_l1);
                volatile tt_l1_ptr float* const odd_i = reinterpret_cast<volatile tt_l1_ptr float*>(odd_i_l1);
                for (uint32_t i = 0; i < LOCAL_PAIRS; ++i) {
                    const uint32_t group = i >> s;
                    const uint32_t pos = i & mask;
                    const uint32_t lo = group * group_size + pos;
                    const uint32_t hi = lo + stride;
                    even_r[i] = state_r[lo];
                    even_i[i] = state_i[lo];
                    odd_r[i] = state_r[hi];
                    odd_i[i] = state_i[hi];
                }
            }

            cb_even_r.push_back(1);
            cb_even_i.push_back(1);
            cb_odd_r.push_back(1);
            cb_odd_i.push_back(1);

            cb_out0_r.wait_front(1);
            cb_out0_i.wait_front(1);
            cb_out1_r.wait_front(1);
            cb_out1_i.wait_front(1);

            const uint32_t o0r_l1 = cb_out0_r.get_read_ptr();
            const uint32_t o0i_l1 = cb_out0_i.get_read_ptr();
            const uint32_t o1r_l1 = cb_out1_r.get_read_ptr();
            const uint32_t o1i_l1 = cb_out1_i.get_read_ptr();

            if (use_dma) {
                for (uint32_t g = 0; g < num_groups; ++g) {
                    const uint32_t dst_lo_r = state_r_l1 + (g * group_size) * 4u;
                    const uint32_t dst_hi_r = state_r_l1 + (g * group_size + stride) * 4u;
                    const uint32_t dst_lo_i = state_i_l1 + (g * group_size) * 4u;
                    const uint32_t dst_hi_i = state_i_l1 + (g * group_size + stride) * 4u;
                    const uint32_t src_o0r = o0r_l1 + (g * stride) * 4u;
                    const uint32_t src_o0i = o0i_l1 + (g * stride) * 4u;
                    const uint32_t src_o1r = o1r_l1 + (g * stride) * 4u;
                    const uint32_t src_o1i = o1i_l1 + (g * stride) * 4u;
                    async_local_memcpy(noc, src_o0r, dst_lo_r, block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(noc, src_o1r, dst_hi_r, block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(noc, src_o0i, dst_lo_i, block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(noc, src_o1i, dst_hi_i, block_bytes, my_noc_x, my_noc_y);
                }
                noc.async_write_barrier();
            } else {
                volatile tt_l1_ptr float* const o0r = reinterpret_cast<volatile tt_l1_ptr float*>(o0r_l1);
                volatile tt_l1_ptr float* const o0i = reinterpret_cast<volatile tt_l1_ptr float*>(o0i_l1);
                volatile tt_l1_ptr float* const o1r = reinterpret_cast<volatile tt_l1_ptr float*>(o1r_l1);
                volatile tt_l1_ptr float* const o1i = reinterpret_cast<volatile tt_l1_ptr float*>(o1i_l1);
                for (uint32_t i = 0; i < LOCAL_PAIRS; ++i) {
                    const uint32_t group = i >> s;
                    const uint32_t pos = i & mask;
                    const uint32_t lo = group * group_size + pos;
                    const uint32_t hi = lo + stride;
                    state_r[lo] = o0r[i];
                    state_i[lo] = o0i[i];
                    state_r[hi] = o1r[i];
                    state_i[hi] = o1i[i];
                }
            }

            cb_out0_r.pop_front(1);
            cb_out0_i.pop_front(1);
            cb_out1_r.pop_front(1);
            cb_out1_i.pop_front(1);
        }

        // ── Load post-twiddle tile (consumed by the writer) ─────────────
        // STATE_R/I now hold the FFT output for this row.  We DON'T
        // touch them here — instead, drop the broadcast twiddle row
        // T[tile_idx % pt_modulus, :] into CB_PT_R/I so the writer can
        // do the scalar complex-multiply right before its
        // noc_async_write_page (see header comment for the rationale).
#ifdef APPLY_POST_TWIDDLE
        {
            const auto pt_r_gen = TensorAccessor(tensor::post_tw_r);
            const auto pt_i_gen = TensorAccessor(tensor::post_tw_i);

            const uint32_t pt_tile_idx = (tile_idx / pt_stride) % pt_modulus;
            cb_pt_r.reserve_back(1);
            cb_pt_i.reserve_back(1);
            noc.async_read(pt_r_gen, cb_pt_r, pt_r_gen.get_aligned_page_size(), {.page_id = pt_tile_idx}, {});
            noc.async_read(pt_i_gen, cb_pt_i, pt_i_gen.get_aligned_page_size(), {.page_id = pt_tile_idx}, {});
            noc.async_read_barrier();
            cb_pt_r.push_back(1);
            cb_pt_i.push_back(1);
        }
#endif

        cb_sync.reserve_back(1);
        cb_sync.push_back(1);
    }
}
