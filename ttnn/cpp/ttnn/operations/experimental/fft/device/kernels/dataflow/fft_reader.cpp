// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_reader.cpp — BRISC0 / reader (multi-core, DMA-optimised)
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

#ifndef FFT_DPRINT_HELLO
#define FFT_DPRINT_HELLO 0
#endif
#if FFT_DPRINT_HELLO
#include "debug/dprint.h"
#endif

// Block size (in bytes) below which scalar memcpy beats NoC DMA because
// of per-packet setup cost. Empirical; 64 B ~= 16 floats is a good knee.
constexpr uint32_t kScalarCutoff = 64;

// Async local L1->L1 copy via the core's own NoC endpoint. Multiple of these
// can be in flight simultaneously; the single barrier at the end of each
// logical group (gather / scatter) is what makes it free.
FORCE_INLINE void async_local_memcpy(
    uint32_t src_l1, uint32_t dst_l1, uint32_t n_bytes,
    uint32_t my_noc_x, uint32_t my_noc_y)
{
    const uint64_t dst_noc = get_noc_addr(my_noc_x, my_noc_y, dst_l1);
    noc_async_write(src_l1, dst_noc, n_bytes);
}

void kernel_main() {
    const uint32_t in_r_addr  = get_arg_val<uint32_t>(0);
    const uint32_t in_i_addr  = get_arg_val<uint32_t>(1);
    const uint32_t tw_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t tw_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t my_core    = get_arg_val<uint32_t>(4);
    const uint32_t sem_id     = get_arg_val<uint32_t>(5);
    // args 6..6+2P-1 : noc_x[c], noc_y[c] interleaved for c=0..P-1

#if FFT_DPRINT_HELLO
    DPRINT << "fft: my_core=" << my_core << " alive\n";
#endif

    constexpr uint32_t N             = get_compile_time_arg_val(0);
    constexpr uint32_t LOG2N         = get_compile_time_arg_val(1);
    constexpr uint32_t P             = get_compile_time_arg_val(2);
    constexpr uint32_t LOG2P         = get_compile_time_arg_val(3);
    constexpr uint32_t LOG2N_LOCAL   = LOG2N - LOG2P;
    constexpr uint32_t LOCAL_PAIRS   = (P == 1) ? (N / 2) : (TILE_ELEMS / 2);

    const DataFormat df = get_dataformat(CB_EVEN_R);
    const uint32_t   ts = get_tile_size(CB_EVEN_R);

    // My own NoC coords (for local L1->L1 DMA). They're in the noc_xy table
    // at slot my_core.
    const uint32_t my_noc_x = get_arg_val<uint32_t>(6 + my_core * 2);
    const uint32_t my_noc_y = get_arg_val<uint32_t>(6 + my_core * 2 + 1);

    InterleavedAddrGenFast<true> in_r_gen = {
        .bank_base_address = in_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> in_i_gen = {
        .bank_base_address = in_i_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr, .page_size = ts, .data_format = df};

    // ── Load this core's shard of the bit-reversed input into state ───────
    cb_reserve_back(CB_STATE_R, 1);
    cb_reserve_back(CB_STATE_I, 1);
    const uint32_t state_r_l1 = get_write_ptr(CB_STATE_R);
    const uint32_t state_i_l1 = get_write_ptr(CB_STATE_I);
    noc_async_read_tile(my_core, in_r_gen, state_r_l1);
    noc_async_read_tile(my_core, in_i_gen, state_i_l1);
    noc_async_read_barrier();
    cb_push_back(CB_STATE_R, 1);
    cb_push_back(CB_STATE_I, 1);

    volatile tt_l1_ptr float* const state_r =
        reinterpret_cast<volatile tt_l1_ptr float*>(state_r_l1);
    volatile tt_l1_ptr float* const state_i =
        reinterpret_cast<volatile tt_l1_ptr float*>(state_i_l1);

    // Cross-core bookkeeping (unused if P == 1).
    const uint32_t sem_l1 = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* const sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);

    const uint32_t recv_r_l1 = get_write_ptr(CB_RECV_R);
    const uint32_t recv_i_l1 = get_write_ptr(CB_RECV_I);

    // ── LOCAL stages (0 .. LOG2N_LOCAL-1) ─────────────────────────────────
    for (uint32_t s = 0; s < LOG2N_LOCAL; ++s) {
        const uint32_t stride       = 1u << s;
        const uint32_t group_size   = stride << 1;      // 2*stride elements
        const uint32_t mask         = stride - 1;
        const uint32_t num_groups   = LOCAL_PAIRS / stride;  // N/(2*stride)
        const uint32_t block_bytes  = stride * 4u;
        const bool     use_dma      = block_bytes >= kScalarCutoff;

        // --- stage twiddle tile ----------------------------------------
        cb_reserve_back(CB_TW_R, 1);
        cb_reserve_back(CB_TW_I, 1);
        noc_async_read_tile(s * P + my_core, tw_r_gen, get_write_ptr(CB_TW_R));
        noc_async_read_tile(s * P + my_core, tw_i_gen, get_write_ptr(CB_TW_I));
        noc_async_read_barrier();
        cb_push_back(CB_TW_R, 1);
        cb_push_back(CB_TW_I, 1);

        // --- scatter: state -> EVEN/ODD --------------------------------
        cb_reserve_back(CB_EVEN_R, 1);
        cb_reserve_back(CB_EVEN_I, 1);
        cb_reserve_back(CB_ODD_R,  1);
        cb_reserve_back(CB_ODD_I,  1);

        const uint32_t even_r_l1 = get_write_ptr(CB_EVEN_R);
        const uint32_t even_i_l1 = get_write_ptr(CB_EVEN_I);
        const uint32_t odd_r_l1  = get_write_ptr(CB_ODD_R);
        const uint32_t odd_i_l1  = get_write_ptr(CB_ODD_I);

        if (use_dma) {
            // For each butterfly group, the first `stride` elements of state
            // are the "even" block and the next `stride` are "odd". 4 async
            // writes per group (R/I x even/odd). One barrier at the end.
            for (uint32_t g = 0; g < num_groups; ++g) {
                const uint32_t src_even = state_r_l1 + (g * group_size)           * 4u;
                const uint32_t src_odd  = state_r_l1 + (g * group_size + stride)  * 4u;
                const uint32_t src_evi  = state_i_l1 + (g * group_size)           * 4u;
                const uint32_t src_odi  = state_i_l1 + (g * group_size + stride)  * 4u;
                const uint32_t dst_even = even_r_l1  + (g * stride)               * 4u;
                const uint32_t dst_odd  = odd_r_l1   + (g * stride)               * 4u;
                const uint32_t dst_evi  = even_i_l1  + (g * stride)               * 4u;
                const uint32_t dst_odi  = odd_i_l1   + (g * stride)               * 4u;
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

        // --- gather: OUT0/OUT1 -> state --------------------------------
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

    if constexpr (P > 1) {
        // One-time initial send: prime partner_0's recv buffer.
        {
            const uint32_t p0    = my_core ^ 1u;
            const uint32_t p0_x  = get_arg_val<uint32_t>(6 + p0 * 2);
            const uint32_t p0_y  = get_arg_val<uint32_t>(6 + p0 * 2 + 1);
            const uint64_t p0_rr = get_noc_addr(p0_x, p0_y, recv_r_l1);
            const uint64_t p0_ri = get_noc_addr(p0_x, p0_y, recv_i_l1);
            const uint64_t p0_sm = get_noc_addr(p0_x, p0_y, sem_l1);
            noc_async_write(state_r_l1, p0_rr, TILE_SIZE_FP32);
            noc_async_write(state_i_l1, p0_ri, TILE_SIZE_FP32);
#if defined(ARCH_BLACKHOLE)
            noc_async_full_barrier();
#else
            noc_async_write_barrier();
#endif
            noc_semaphore_inc(p0_sm, 1);
        }

        for (uint32_t k = 0; k < LOG2P; ++k) {
            const uint32_t s         = LOG2N_LOCAL + k;
            const uint32_t bit       = 1u << k;
            const bool     is_c_even = (my_core & bit) == 0;

            // Kick off twiddle DRAM read early — this overlaps the NoC
            // semaphore wait below.
            cb_reserve_back(CB_TW_R, 1);
            cb_reserve_back(CB_TW_I, 1);
            noc_async_read_tile(s * P + my_core, tw_r_gen, get_write_ptr(CB_TW_R));
            noc_async_read_tile(s * P + my_core, tw_i_gen, get_write_ptr(CB_TW_I));

            // Wait for partner_k's tile (sent at end of their stage k-1, or
            // by the initial prime for k=0). Monotonic count so missed/late
            // increments can't race.
#if defined(ARCH_BLACKHOLE)
            noc_semaphore_wait_min(sem_ptr, k + 1);
#else
            noc_semaphore_wait(sem_ptr, k + 1);
#endif

            // Finish the twiddle read.
            noc_async_read_barrier();
            cb_push_back(CB_TW_R, 1);
            cb_push_back(CB_TW_I, 1);

            // Fill EVEN/ODD via async full-tile DMA (4096 B per copy).
            cb_reserve_back(CB_EVEN_R, 1);
            cb_reserve_back(CB_EVEN_I, 1);
            cb_reserve_back(CB_ODD_R,  1);
            cb_reserve_back(CB_ODD_I,  1);

            const uint32_t even_r_l1 = get_write_ptr(CB_EVEN_R);
            const uint32_t even_i_l1 = get_write_ptr(CB_EVEN_I);
            const uint32_t odd_r_l1  = get_write_ptr(CB_ODD_R);
            const uint32_t odd_i_l1  = get_write_ptr(CB_ODD_I);

            if (is_c_even) {
                async_local_memcpy(state_r_l1, even_r_l1, TILE_SIZE_FP32, my_noc_x, my_noc_y);
                async_local_memcpy(state_i_l1, even_i_l1, TILE_SIZE_FP32, my_noc_x, my_noc_y);
                async_local_memcpy(recv_r_l1,  odd_r_l1,  TILE_SIZE_FP32, my_noc_x, my_noc_y);
                async_local_memcpy(recv_i_l1,  odd_i_l1,  TILE_SIZE_FP32, my_noc_x, my_noc_y);
            } else {
                async_local_memcpy(recv_r_l1,  even_r_l1, TILE_SIZE_FP32, my_noc_x, my_noc_y);
                async_local_memcpy(recv_i_l1,  even_i_l1, TILE_SIZE_FP32, my_noc_x, my_noc_y);
                async_local_memcpy(state_r_l1, odd_r_l1,  TILE_SIZE_FP32, my_noc_x, my_noc_y);
                async_local_memcpy(state_i_l1, odd_i_l1,  TILE_SIZE_FP32, my_noc_x, my_noc_y);
            }
            noc_async_write_barrier();

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

            // The tile we keep as our new state: OUT0 if c_even, else OUT1.
            const uint32_t src_r = is_c_even ? o0r_l1 : o1r_l1;
            const uint32_t src_i = is_c_even ? o0i_l1 : o1i_l1;

            // Local scatter (always).
            async_local_memcpy(src_r, state_r_l1, TILE_SIZE_FP32, my_noc_x, my_noc_y);
            async_local_memcpy(src_i, state_i_l1, TILE_SIZE_FP32, my_noc_x, my_noc_y);

            // Fused send to next stage's partner (in parallel with local
            // scatter). Last stage has no next partner.
            if (k + 1 < LOG2P) {
                const uint32_t np    = my_core ^ (1u << (k + 1));
                const uint32_t np_x  = get_arg_val<uint32_t>(6 + np * 2);
                const uint32_t np_y  = get_arg_val<uint32_t>(6 + np * 2 + 1);
                const uint64_t np_rr = get_noc_addr(np_x, np_y, recv_r_l1);
                const uint64_t np_ri = get_noc_addr(np_x, np_y, recv_i_l1);
                const uint64_t np_sm = get_noc_addr(np_x, np_y, sem_l1);
                noc_async_write(src_r, np_rr, TILE_SIZE_FP32);
                noc_async_write(src_i, np_ri, TILE_SIZE_FP32);
#if defined(ARCH_BLACKHOLE)
                noc_async_full_barrier();
#else
                noc_async_write_barrier();
#endif
                noc_semaphore_inc(np_sm, 1);
            } else {
                noc_async_write_barrier();
            }

            cb_pop_front(CB_OUT0_R, 1);
            cb_pop_front(CB_OUT0_I, 1);
            cb_pop_front(CB_OUT1_R, 1);
            cb_pop_front(CB_OUT1_I, 1);
        }
    }

    cb_reserve_back(CB_SYNC, 1);
    cb_push_back(CB_SYNC, 1);
}
