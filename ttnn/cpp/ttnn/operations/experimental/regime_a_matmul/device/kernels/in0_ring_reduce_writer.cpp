// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Regime-A in0 ring all-gather + split-K reduction + output write (runs on the core's non-in1 NoC/RISC).
//
// PHASE 1 — in0 ring all-gather. The G=8 cores sharing a (n-slice, m-block) group across the 8 banks form
// a ring. in0 is small and DRAM-interleaved: every core reads only its OWN shard (W blocks of in0[:,k-slice])
// in parallel, then the shards rotate cyclically so each core ends up holding the full k-slice. cb0 holds G
// slots of W blocks; slot s of core c ends up holding shard (c-s), matched by the in1 reader's rotated read.
//
// PHASE 2 — split-K reduction (only when Pk>1). The Pk k-slices of a fixed (bank, n-slice, m-block) form a
// linear chain. Each non-bottom band receives the running sum from the band below into cb_reduce; compute
// (REDUCE_K) adds it; the top band writes the final [M,N] block to DRAM. When Pk==1 every core is bottom AND
// top, so it writes its own block directly with no reduction traffic (and cb_reduce is never touched).
//
// Cleaned unified-only port of the prototype in0_ring_writer.cpp (ring all-gather + reduction chain). No
// mshard / in0_direct / scatter / in0-share / skip / noreduce ablation paths.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M_block = get_compile_time_arg_val(0);
    constexpr uint32_t K_block = get_compile_time_arg_val(1);       // kb
    constexpr uint32_t N_block = get_compile_time_arg_val(2);       // N_sub
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(3);  // G*W (full k-slice)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t Kt = get_compile_time_arg_val(5);               // in0 physical row stride (= logical Kt)
    constexpr uint32_t Nt = get_compile_time_arg_val(6);               // out physical row stride (= logical Nt)
    constexpr uint32_t W = get_compile_time_arg_val(7);                // blocks per shard
    constexpr uint32_t G = get_compile_time_arg_val(8);                // ring size (8)
    constexpr uint32_t fwd_sem_id = get_compile_time_arg_val(9);       // in0 ring recv semaphore
    constexpr uint32_t red_sem_id = get_compile_time_arg_val(10);      // reduction recv semaphore
    constexpr uint32_t N_bpc = get_compile_time_arg_val(11);           // N-sub-blocks per core
    constexpr uint32_t redfree_sem_id = get_compile_time_arg_val(12);  // cb_reduce reverse credit
    constexpr uint32_t use_reduce = get_compile_time_arg_val(13);      // 1 when Pk>1 (reduction chain active)
    constexpr auto in0_args = TensorAccessorArgs<14>();
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    const uint32_t in0_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t m_start = get_arg_val<uint32_t>(2);  // first logical M tile (balanced)
    const uint32_t n_start = get_arg_val<uint32_t>(3);  // first logical (global) N tile (output addressing)
    const uint32_t k_start = get_arg_val<uint32_t>(4);  // first logical K tile (balanced)
    const uint32_t ring_pos = get_arg_val<uint32_t>(5);
    const uint32_t fwd_next_x = get_arg_val<uint32_t>(6);
    const uint32_t fwd_next_y = get_arg_val<uint32_t>(7);
    const uint32_t red_next_x = get_arg_val<uint32_t>(8);
    const uint32_t red_next_y = get_arg_val<uint32_t>(9);
    const uint32_t is_bottom = get_arg_val<uint32_t>(10);
    const uint32_t is_top = get_arg_val<uint32_t>(11);
    const uint32_t red_prev_x = get_arg_val<uint32_t>(12);
    const uint32_t red_prev_y = get_arg_val<uint32_t>(13);
    const uint32_t valid_k = get_arg_val<uint32_t>(14);  // valid K tiles (rest of capacity zero-filled)
    const uint32_t valid_m = get_arg_val<uint32_t>(15);  // valid M tiles (rest zero / not written)
    const uint32_t valid_n = get_arg_val<uint32_t>(16);  // valid N tiles (rest zero / not written)

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    constexpr uint32_t in0_cb = 0, out_cb = 2, cb_reduce = 7;
    constexpr uint32_t in0_blk = M_block * K_block;
    constexpr uint32_t in0_blk_bytes = in0_blk * tile_bytes;
    constexpr uint32_t shard_bytes = W * in0_blk_bytes;
    constexpr uint32_t out_blk = M_block * N_block;
    const uint32_t fwd_addr = get_semaphore(fwd_sem_id);
    volatile tt_l1_ptr uint32_t* fwd_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_addr);
    constexpr uint32_t words_per_tile = tile_bytes / 4u;
    auto zero_tile = [](uint32_t addr) {
        volatile tt_l1_ptr uint32_t* q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        for (uint32_t i = 0; i < words_per_tile; ++i) {
            q[i] = 0;
        }
    };

    // ---- PHASE 1: in0 ring all-gather (balanced tails: read only valid M rows / valid K, else zero) ----
    cb_reserve_back(in0_cb, K_num_blocks * in0_blk);
    const uint32_t base0 = get_write_ptr(in0_cb);
    for (uint32_t step = 0; step < G; ++step) {
        uint32_t slot = base0 + step * shard_bytes;
        if (step == 0) {
            // read our OWN shard (shard index = ring_pos) into slot 0
            uint32_t p = slot;
            for (uint32_t wb = 0; wb < W; ++wb) {
                uint32_t sb = ring_pos * W + wb;  // capacity-local block index of own shard
                for (uint32_t m = 0; m < M_block; ++m) {
                    for (uint32_t k = 0; k < K_block; ++k) {
                        const uint32_t l = sb * K_block + k;  // capacity-local K index within the slice
                        if (m < valid_m && l < valid_k) {
                            noc_async_read_page((m_start + m) * Kt + (k_start + l), in0, p);
                        } else {
                            zero_tile(p);  // pad M row or K tail -> local zero (no DRAM read)
                        }
                        p += tile_bytes;
                    }
                }
            }
            noc_async_read_barrier();
        } else {
            noc_semaphore_wait_min(fwd_ptr, step);  // prev forwarded a shard into our slot `step`
        }
        if (step + 1 < G) {  // forward this slot to the next core's slot (step+1)
            uint64_t dst = get_noc_addr(fwd_next_x, fwd_next_y, base0 + (step + 1) * shard_bytes);
            noc_async_write(slot, dst, shard_bytes);
            noc_semaphore_inc(get_noc_addr(fwd_next_x, fwd_next_y, fwd_addr), 1);
        }
        cb_push_back(in0_cb, W * in0_blk);  // compute consumes this shard (W blocks)
    }
    noc_async_write_barrier();  // all ring forwards landed

    // ---- PHASE 2: output / split-K reduction over the N_bpc output blocks ----
    constexpr uint32_t out_blk_bytes = out_blk * tile_bytes;

    if constexpr (!use_reduce) {
        // Pk == 1: every core is bottom AND top; compute produced its full block into out_cb -> write DRAM.
        for (uint32_t nb = 0; nb < N_bpc; ++nb) {
            cb_wait_front(out_cb, out_blk);
            uint32_t r = get_read_ptr(out_cb);
            const uint32_t n_off = n_start + nb * N_block;  // global N tile of this subblock
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t n = 0; n < N_block; ++n) {
                    if (m < valid_m && (nb * N_block + n) < valid_n) {  // write only valid_m x valid_n
                        noc_async_write_page((m_start + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(out_cb, out_blk);
        }
        return;
    }

    // Pk > 1: linear reduction chain. cb_reduce holds 2 blocks (double-buffered). reduce_base captured ONCE
    // BEFORE any cb_reduce use (the write ptr drifts after receives).
    const uint32_t reduce_base = get_write_ptr(cb_reduce);
    const uint32_t red_addr = get_semaphore(red_sem_id);
    volatile tt_l1_ptr uint32_t* red_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_addr);
    const uint32_t redfree_addr = get_semaphore(redfree_sem_id);
    volatile tt_l1_ptr uint32_t* redfree_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(redfree_addr);
    const uint64_t prev_redfree = get_noc_addr(red_prev_x, red_prev_y, redfree_addr);
    const uint64_t next_recv = get_noc_addr(red_next_x, red_next_y, red_addr);

    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
        if (!is_bottom) {
            cb_reserve_back(cb_reduce, out_blk);      // wait our compute freed slot (nb-2)
            noc_semaphore_inc(prev_redfree, 1);       // tell prev: our slot (nb%2) is free for block nb
            noc_semaphore_wait_min(red_ptr, nb + 1);  // prev forwarded block nb into it
            cb_push_back(cb_reduce, out_blk);         // compute reduce_add's it -> out_cb, pops cb_reduce
        }
        cb_wait_front(out_cb, out_blk);  // compute produced reduced block nb
        uint32_t r = get_read_ptr(out_cb);
        if (!is_top) {
            noc_semaphore_wait_min(redfree_ptr, nb + 1);  // next signalled its slot (nb%2) is free
            uint64_t dst = get_noc_addr(red_next_x, red_next_y, reduce_base + (nb % 2) * out_blk_bytes);
            noc_async_write(r, dst, out_blk_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(next_recv, 1);  // block nb delivered
        } else {
            const uint32_t n_off = n_start + nb * N_block;  // global N tile of this subblock
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t n = 0; n < N_block; ++n) {
                    if (m < valid_m && (nb * N_block + n) < valid_n) {  // write only valid_m x valid_n
                        noc_async_write_page((m_start + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
                    }
                }
            }
            noc_async_write_barrier();
        }
        cb_pop_front(out_cb, out_blk);
    }
}
