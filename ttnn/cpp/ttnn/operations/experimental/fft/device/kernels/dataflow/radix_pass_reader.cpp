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
// BEFORE issuing the noc_async_write_tile out to DRAM.
//
// Why on the WRITER instead of here?  Doing the cmul here would write
// to STATE_R/I after we have already pushed them; cross-BRISC L1
// visibility for those late scalar stores is sensitive to the per-core
// L1 layout and observably flaky for some (row, core) combinations
// (e.g. row 12 on grid (4,1)).  Letting BRISC1 read, modify, and write
// STATE within a single thread closes that gap entirely — BRISC1's
// own scalar stores are guaranteed visible to its subsequent
// noc_async_write_tile read of L1.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "radix_pass_common.h"

constexpr uint32_t kScalarCutoff = 64;

FORCE_INLINE void async_local_memcpy(
    uint32_t src_l1, uint32_t dst_l1, uint32_t n_bytes,
    uint32_t my_noc_x, uint32_t my_noc_y)
{
    const uint64_t dst_noc = get_noc_addr(my_noc_x, my_noc_y, dst_l1);
    noc_async_write(src_l1, dst_noc, n_bytes);
}

void kernel_main() {
    const uint32_t in_r_addr       = get_arg_val<uint32_t>(0);
    const uint32_t in_i_addr       = get_arg_val<uint32_t>(1);
    const uint32_t tw_r_addr       = get_arg_val<uint32_t>(2);
    const uint32_t tw_i_addr       = get_arg_val<uint32_t>(3);
    const uint32_t base_tile_idx   = get_arg_val<uint32_t>(4);
    const uint32_t batch_per_core  = get_arg_val<uint32_t>(5);
    const uint32_t my_noc_x        = get_arg_val<uint32_t>(6);
    const uint32_t my_noc_y        = get_arg_val<uint32_t>(7);
    const uint32_t in_page_size_override      = get_arg_val<uint32_t>(8);
    const uint32_t in_imag_page_size_override = get_arg_val<uint32_t>(9);
    // Post-twiddle args.  When APPLY_POST_TWIDDLE=0, the factory passes
    // zeros for 10..13 and they are never dereferenced.  When =1:
    //   pt_r_addr / pt_i_addr — tile-sized fp32 twiddle table, layout
    //     identical to apply_twiddles_host::build_twiddle_table(P, N2):
    //     N2 tiles total, tile n2 holds T[n2, 0..P-1] in slots [0, P).
    //   pt_modulus = N2 (twiddle modulus on row index).
    //   pt_stride  = row-index stride before the modulus (default 1).
    //                Three-pass Pass-2 uses pt_stride=N3 so that the row
    //                enumeration (b, n1, k3) — laid out at stride N3 —
    //                picks the right n1 twiddle without an extra
    //                transpose.
    const uint32_t pt_r_addr       = get_arg_val<uint32_t>(10);
    const uint32_t pt_i_addr       = get_arg_val<uint32_t>(11);
    const uint32_t pt_modulus      = get_arg_val<uint32_t>(12);
    const uint32_t pt_stride       = get_arg_val<uint32_t>(13);

    constexpr uint32_t SUB_N        = get_compile_time_arg_val(0);
    constexpr uint32_t LOG2_SUB_N   = get_compile_time_arg_val(1);
    constexpr uint32_t LOCAL_PAIRS  = SUB_N / 2;
    constexpr uint32_t BIT_REVERSE_ON_LOAD = get_compile_time_arg_val(2);
    constexpr uint32_t INPUT_BF16          = get_compile_time_arg_val(3);
    constexpr uint32_t APPLY_POST_TWIDDLE  = get_compile_time_arg_val(4);

    const DataFormat df = get_dataformat(CB_EVEN_R);
    const uint32_t   ts = get_tile_size(CB_EVEN_R);

    uint32_t fallback_ts;
    if constexpr (INPUT_BF16) {
        fallback_ts = get_tile_size(CB_IN_R_BF16);
    } else {
        fallback_ts = ts;
    }
    const uint32_t in_r_ps = in_page_size_override      ? in_page_size_override      : fallback_ts;
    const uint32_t in_i_ps = in_imag_page_size_override ? in_imag_page_size_override : fallback_ts;
    const InterleavedAddrGen<true> in_r_gen = {
            .bank_base_address = in_r_addr, .page_size = in_r_ps};
    const InterleavedAddrGen<true> in_i_gen = {
            .bank_base_address = in_i_addr, .page_size = in_i_ps};

    InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr, .page_size = ts, .data_format = df};

    // Post-twiddle generators — only used when APPLY_POST_TWIDDLE=1.
    // The twiddle MeshBuffer is tile-sized, so *Fast is correct here.
    InterleavedAddrGenFast<true> pt_r_gen = {
        .bank_base_address = pt_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> pt_i_gen = {
        .bank_base_address = pt_i_addr, .page_size = ts, .data_format = df};

    for (uint32_t k = 0; k < batch_per_core; ++k) {
        const uint32_t tile_idx = base_tile_idx + k;

        cb_reserve_back(CB_STATE_R, 1);
        cb_reserve_back(CB_STATE_I, 1);
        const uint32_t state_r_l1 = get_write_ptr(CB_STATE_R);
        const uint32_t state_i_l1 = get_write_ptr(CB_STATE_I);

        if constexpr (INPUT_BF16) {
            cb_reserve_back(CB_IN_R_BF16, 1);
            cb_reserve_back(CB_IN_I_BF16, 1);
            const uint32_t in_r_bf16_l1 = get_write_ptr(CB_IN_R_BF16);
            const uint32_t in_i_bf16_l1 = get_write_ptr(CB_IN_I_BF16);
            noc_async_read_tile(tile_idx, in_r_gen, in_r_bf16_l1);
            noc_async_read_tile(tile_idx, in_i_gen, in_i_bf16_l1);
            noc_async_read_barrier();
            cb_push_back(CB_IN_R_BF16, 1);
            cb_push_back(CB_IN_I_BF16, 1);

            volatile tt_l1_ptr uint16_t* const sb_r =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_r_bf16_l1);
            volatile tt_l1_ptr uint16_t* const sb_i =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_i_bf16_l1);
            volatile tt_l1_ptr uint32_t* const dst_r =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_r_l1);
            volatile tt_l1_ptr uint32_t* const dst_i =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_i_l1);
            for (uint32_t kk = 0; kk < SUB_N; ++kk) {
                dst_r[kk] = static_cast<uint32_t>(sb_r[kk]) << 16;
                dst_i[kk] = static_cast<uint32_t>(sb_i[kk]) << 16;
            }

            cb_pop_front(CB_IN_R_BF16, 1);
            cb_pop_front(CB_IN_I_BF16, 1);
        } else {
            noc_async_read_tile(tile_idx, in_r_gen, state_r_l1);
            noc_async_read_tile(tile_idx, in_i_gen, state_i_l1);
            noc_async_read_barrier();
        }

        cb_push_back(CB_STATE_R, 1);
        cb_push_back(CB_STATE_I, 1);

        volatile tt_l1_ptr float* const state_r =
            reinterpret_cast<volatile tt_l1_ptr float*>(state_r_l1);
        volatile tt_l1_ptr float* const state_i =
            reinterpret_cast<volatile tt_l1_ptr float*>(state_i_l1);

        if constexpr (BIT_REVERSE_ON_LOAD) {
            for (uint32_t kk = 0; kk < SUB_N; ++kk) {
                uint32_t br = 0;
                for (uint32_t b = 0; b < LOG2_SUB_N; ++b) {
                    br = (br << 1) | ((kk >> b) & 1u);
                }
                if (kk < br) {
                    float tr = state_r[kk]; state_r[kk] = state_r[br]; state_r[br] = tr;
                    float ti = state_i[kk]; state_i[kk] = state_i[br]; state_i[br] = ti;
                }
            }
        }

        // ── LOCAL Stockham stages 0..LOG2_SUB_N-1 ──────────────────────
        for (uint32_t s = 0; s < LOG2_SUB_N; ++s) {
            const uint32_t stride       = 1u << s;
            const uint32_t group_size   = stride << 1;
            const uint32_t mask         = stride - 1;
            const uint32_t num_groups   = LOCAL_PAIRS / stride;
            const uint32_t block_bytes  = stride * 4u;
            const bool     use_dma      = block_bytes >= kScalarCutoff;

            cb_reserve_back(CB_TW_R, 1);
            cb_reserve_back(CB_TW_I, 1);
            noc_async_read_tile(s, tw_r_gen, get_write_ptr(CB_TW_R));
            noc_async_read_tile(s, tw_i_gen, get_write_ptr(CB_TW_I));
            noc_async_read_barrier();
            cb_push_back(CB_TW_R, 1);
            cb_push_back(CB_TW_I, 1);

            cb_reserve_back(CB_EVEN_R, 1);
            cb_reserve_back(CB_EVEN_I, 1);
            cb_reserve_back(CB_ODD_R,  1);
            cb_reserve_back(CB_ODD_I,  1);

            const uint32_t even_r_l1 = get_write_ptr(CB_EVEN_R);
            const uint32_t even_i_l1 = get_write_ptr(CB_EVEN_I);
            const uint32_t odd_r_l1  = get_write_ptr(CB_ODD_R);
            const uint32_t odd_i_l1  = get_write_ptr(CB_ODD_I);

            if (use_dma) {
                for (uint32_t g = 0; g < num_groups; ++g) {
                    const uint32_t src_even = state_r_l1 + (g * group_size)          * 4u;
                    const uint32_t src_odd  = state_r_l1 + (g * group_size + stride) * 4u;
                    const uint32_t src_evi  = state_i_l1 + (g * group_size)          * 4u;
                    const uint32_t src_odi  = state_i_l1 + (g * group_size + stride) * 4u;
                    const uint32_t dst_even = even_r_l1  + (g * stride)              * 4u;
                    const uint32_t dst_odd  = odd_r_l1   + (g * stride)              * 4u;
                    const uint32_t dst_evi  = even_i_l1  + (g * stride)              * 4u;
                    const uint32_t dst_odi  = odd_i_l1   + (g * stride)              * 4u;
                    async_local_memcpy(src_even, dst_even, block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(src_odd,  dst_odd,  block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(src_evi,  dst_evi,  block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(src_odi,  dst_odi,  block_bytes, my_noc_x, my_noc_y);
                }
                noc_async_write_barrier();
            } else {
                volatile tt_l1_ptr float* const even_r =
                    reinterpret_cast<volatile tt_l1_ptr float*>(even_r_l1);
                volatile tt_l1_ptr float* const even_i =
                    reinterpret_cast<volatile tt_l1_ptr float*>(even_i_l1);
                volatile tt_l1_ptr float* const odd_r =
                    reinterpret_cast<volatile tt_l1_ptr float*>(odd_r_l1);
                volatile tt_l1_ptr float* const odd_i =
                    reinterpret_cast<volatile tt_l1_ptr float*>(odd_i_l1);
                for (uint32_t i = 0; i < LOCAL_PAIRS; ++i) {
                    const uint32_t group = i >> s;
                    const uint32_t pos   = i & mask;
                    const uint32_t lo    = group * group_size + pos;
                    const uint32_t hi    = lo + stride;
                    even_r[i] = state_r[lo];
                    even_i[i] = state_i[lo];
                    odd_r[i]  = state_r[hi];
                    odd_i[i]  = state_i[hi];
                }
            }

            cb_push_back(CB_EVEN_R, 1);
            cb_push_back(CB_EVEN_I, 1);
            cb_push_back(CB_ODD_R,  1);
            cb_push_back(CB_ODD_I,  1);

            cb_wait_front(CB_OUT0_R, 1);
            cb_wait_front(CB_OUT0_I, 1);
            cb_wait_front(CB_OUT1_R, 1);
            cb_wait_front(CB_OUT1_I, 1);

            const uint32_t o0r_l1 = get_read_ptr(CB_OUT0_R);
            const uint32_t o0i_l1 = get_read_ptr(CB_OUT0_I);
            const uint32_t o1r_l1 = get_read_ptr(CB_OUT1_R);
            const uint32_t o1i_l1 = get_read_ptr(CB_OUT1_I);

            if (use_dma) {
                for (uint32_t g = 0; g < num_groups; ++g) {
                    const uint32_t dst_lo_r = state_r_l1 + (g * group_size)          * 4u;
                    const uint32_t dst_hi_r = state_r_l1 + (g * group_size + stride) * 4u;
                    const uint32_t dst_lo_i = state_i_l1 + (g * group_size)          * 4u;
                    const uint32_t dst_hi_i = state_i_l1 + (g * group_size + stride) * 4u;
                    const uint32_t src_o0r  = o0r_l1 + (g * stride) * 4u;
                    const uint32_t src_o0i  = o0i_l1 + (g * stride) * 4u;
                    const uint32_t src_o1r  = o1r_l1 + (g * stride) * 4u;
                    const uint32_t src_o1i  = o1i_l1 + (g * stride) * 4u;
                    async_local_memcpy(src_o0r, dst_lo_r, block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(src_o1r, dst_hi_r, block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(src_o0i, dst_lo_i, block_bytes, my_noc_x, my_noc_y);
                    async_local_memcpy(src_o1i, dst_hi_i, block_bytes, my_noc_x, my_noc_y);
                }
                noc_async_write_barrier();
            } else {
                volatile tt_l1_ptr float* const o0r =
                    reinterpret_cast<volatile tt_l1_ptr float*>(o0r_l1);
                volatile tt_l1_ptr float* const o0i =
                    reinterpret_cast<volatile tt_l1_ptr float*>(o0i_l1);
                volatile tt_l1_ptr float* const o1r =
                    reinterpret_cast<volatile tt_l1_ptr float*>(o1r_l1);
                volatile tt_l1_ptr float* const o1i =
                    reinterpret_cast<volatile tt_l1_ptr float*>(o1i_l1);
                for (uint32_t i = 0; i < LOCAL_PAIRS; ++i) {
                    const uint32_t group = i >> s;
                    const uint32_t pos   = i & mask;
                    const uint32_t lo    = group * group_size + pos;
                    const uint32_t hi    = lo + stride;
                    state_r[lo] = o0r[i]; state_i[lo] = o0i[i];
                    state_r[hi] = o1r[i]; state_i[hi] = o1i[i];
                }
            }

            cb_pop_front(CB_OUT0_R, 1);
            cb_pop_front(CB_OUT0_I, 1);
            cb_pop_front(CB_OUT1_R, 1);
            cb_pop_front(CB_OUT1_I, 1);
        }

        // ── Load post-twiddle tile (consumed by the writer) ─────────────
        // STATE_R/I now hold the FFT output for this row.  We DON'T
        // touch them here — instead, drop the broadcast twiddle row
        // T[tile_idx % pt_modulus, :] into CB_PT_R/I so the writer can
        // do the scalar complex-multiply right before its
        // noc_async_write_tile (see header comment for the rationale).
        if constexpr (APPLY_POST_TWIDDLE) {
            const uint32_t pt_tile_idx = (tile_idx / pt_stride) % pt_modulus;
            cb_reserve_back(CB_PT_R, 1);
            cb_reserve_back(CB_PT_I, 1);
            noc_async_read_tile(pt_tile_idx, pt_r_gen, get_write_ptr(CB_PT_R));
            noc_async_read_tile(pt_tile_idx, pt_i_gen, get_write_ptr(CB_PT_I));
            noc_async_read_barrier();
            cb_push_back(CB_PT_R, 1);
            cb_push_back(CB_PT_I, 1);
        }

        cb_reserve_back(CB_SYNC, 1);
        cb_push_back(CB_SYNC, 1);
    }
}
