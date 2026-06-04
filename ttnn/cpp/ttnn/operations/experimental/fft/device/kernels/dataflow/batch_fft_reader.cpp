// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// batch_fft_reader.cpp — BRISC0 / reader for device-side BATCH FFT.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "batch_fft_common.h"

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

    constexpr uint32_t SUB_N        = get_compile_time_arg_val(0);
    constexpr uint32_t LOG2_SUB_N   = get_compile_time_arg_val(1);
    constexpr uint32_t LOCAL_PAIRS  = SUB_N / 2;

    const DataFormat df = get_dataformat(CB_EVEN_R);
    const uint32_t   ts = get_tile_size(CB_EVEN_R);

    InterleavedAddrGenFast<true> in_r_gen = {
        .bank_base_address = in_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> in_i_gen = {
        .bank_base_address = in_i_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr, .page_size = ts, .data_format = df};

    for (uint32_t k = 0; k < batch_per_core; ++k) {
        const uint32_t tile_idx = base_tile_idx + k;

        // ── Load bit-reversed input tile into STATE ──────────────────────
        cb_reserve_back(CB_STATE_R, 1);
        cb_reserve_back(CB_STATE_I, 1);
        const uint32_t state_r_l1 = get_write_ptr(CB_STATE_R);
        const uint32_t state_i_l1 = get_write_ptr(CB_STATE_I);
        noc_async_read_tile(tile_idx, in_r_gen, state_r_l1);
        noc_async_read_tile(tile_idx, in_i_gen, state_i_l1);
        noc_async_read_barrier();
        cb_push_back(CB_STATE_R, 1);
        cb_push_back(CB_STATE_I, 1);

        volatile tt_l1_ptr float* const state_r =
            reinterpret_cast<volatile tt_l1_ptr float*>(state_r_l1);
        volatile tt_l1_ptr float* const state_i =
            reinterpret_cast<volatile tt_l1_ptr float*>(state_i_l1);

        // ── LOCAL stages 0 .. LOG2_SUB_N-1 ──────────────────────────────
        for (uint32_t s = 0; s < LOG2_SUB_N; ++s) {
            const uint32_t stride       = 1u << s;
            const uint32_t group_size   = stride << 1;
            const uint32_t mask         = stride - 1;
            const uint32_t num_groups   = LOCAL_PAIRS / stride;
            const uint32_t block_bytes  = stride * 4u;
            const bool     use_dma      = block_bytes >= kScalarCutoff;

            // Stage twiddle tile (same across cores, depends only on s).
            cb_reserve_back(CB_TW_R, 1);
            cb_reserve_back(CB_TW_I, 1);
            noc_async_read_tile(s, tw_r_gen, get_write_ptr(CB_TW_R));
            noc_async_read_tile(s, tw_i_gen, get_write_ptr(CB_TW_I));
            noc_async_read_barrier();
            cb_push_back(CB_TW_R, 1);
            cb_push_back(CB_TW_I, 1);

            // Scatter: STATE -> EVEN/ODD.
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

            // Gather: OUT0/OUT1 -> STATE.
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

        // ── Hand STATE off to the writer for tile `tile_idx` ─────────────
        // The writer pops STATE_R/I (and SYNC) which frees the 1-slot CB so
        // the next sub-FFT's `cb_reserve_back(CB_STATE_*)` can proceed.
        cb_reserve_back(CB_SYNC, 1);
        cb_push_back(CB_SYNC, 1);
    }
}
