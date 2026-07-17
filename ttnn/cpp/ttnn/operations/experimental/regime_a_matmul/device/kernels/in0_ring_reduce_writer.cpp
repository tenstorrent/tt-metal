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
#ifdef DIAG_IN0_SCATTER
    // DIAG_IN0_SCATTER (correct variant): read our OWN shard into slot 0, then in ONE scatter round write it
    // to the G-1 cores ahead (peer d gets it in its slot d) and receive the G-1 shards from the cores behind
    // (they fill our slots 1..G-1). base0 is the same L1 address on every ring core, so slot d ends up
    // holding shard (ring_pos-d) exactly as the ring rotation would -> compute + the in1 rotated read are
    // unchanged. Replaces G-1 SERIAL rotations (each gated on the previous) with 1 scatter + 1 wait.
    {
        uint32_t p = base0;
        for (uint32_t wb = 0; wb < W; ++wb) {
            uint32_t sb = ring_pos * W + wb;
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t k = 0; k < K_block; ++k) {
                    const uint32_t l = sb * K_block + k;
                    if (m < valid_m && l < valid_k) {
                        noc_async_read_page((m_start + m) * Kt + (k_start + l), in0, p);
                    } else {
                        zero_tile(p);  // pad M row / K tail -> exact 0.0
                    }
                    p += tile_bytes;
                }
            }
        }
        noc_async_read_barrier();
    }
    for (uint32_t d = 1; d < G; ++d) {  // scatter own shard (slot 0) to peer d's slot d
        uint32_t px = get_arg_val<uint32_t>(17 + (d - 1) * 2), py = get_arg_val<uint32_t>(18 + (d - 1) * 2);
        noc_async_write(base0, get_noc_addr(px, py, base0 + d * shard_bytes), shard_bytes);
    }
    noc_async_writes_flushed();  // payloads landed before we signal
    for (uint32_t d = 1; d < G; ++d) {
        uint32_t px = get_arg_val<uint32_t>(17 + (d - 1) * 2), py = get_arg_val<uint32_t>(18 + (d - 1) * 2);
        noc_semaphore_inc(get_noc_addr(px, py, fwd_addr), 1);
    }
    noc_semaphore_wait_min(fwd_ptr, G - 1);  // the G-1 cores behind filled our slots 1..G-1
    cb_push_back(in0_cb, K_num_blocks * in0_blk);
    noc_async_write_barrier();
#elif defined(IN0_REPL)
    // Replicated shorter ring (R = IN0_REPL): read R seed shards (stride G/R), then rotate the R-shard bundle
    // for G/R rounds (nearest-neighbor forward + incremental per-round push, exactly like the R=1 ring). Slot
    // (r*R+i) ends up holding shard (ring_pos - r - i*(G/R)); the in1 reader uses the same formula, so compute
    // is unchanged. Depth = G/R-1 forwards (vs G-1); cost = R x in0 DRAM seed reads.
    constexpr uint32_t R = IN0_REPL;
    constexpr uint32_t RR = G / R;      // rounds
    constexpr uint32_t stride = G / R;  // seed stride
    for (uint32_t r = 0; r < RR; ++r) {
        uint32_t bslot = base0 + r * R * shard_bytes;
        if (r == 0) {
            uint32_t p = bslot;
            for (uint32_t i = 0; i < R; ++i) {
                const uint32_t shard = (ring_pos + 2u * G - i * stride) % G;  // i-th seed (rp - i*stride)
                for (uint32_t wb = 0; wb < W; ++wb) {
                    const uint32_t sb = shard * W + wb;  // capacity-local block index of this shard
                    for (uint32_t m = 0; m < M_block; ++m) {
                        for (uint32_t k = 0; k < K_block; ++k) {
                            const uint32_t l = sb * K_block + k;
                            if (m < valid_m && l < valid_k) {
                                noc_async_read_page((m_start + m) * Kt + (k_start + l), in0, p);
                            } else {
                                zero_tile(p);
                            }
                            p += tile_bytes;
                        }
                    }
                }
            }
            noc_async_read_barrier();
        } else {
            noc_semaphore_wait_min(fwd_ptr, r);  // prev forwarded round r's R shards into our slots r*R..
        }
        if (r + 1 < RR) {  // forward our R-shard bundle to the next core's round-(r+1) slots
            uint64_t dst = get_noc_addr(fwd_next_x, fwd_next_y, base0 + (r + 1) * R * shard_bytes);
            noc_async_write(bslot, dst, R * shard_bytes);
            noc_semaphore_inc(get_noc_addr(fwd_next_x, fwd_next_y, fwd_addr), 1);
        }
        cb_push_back(in0_cb, R * W * in0_blk);  // incremental: compute consumes this round's R shards
    }
    noc_async_write_barrier();
#elif defined(DIAG_IN0_XCHG) || defined(DIAG_IN0_XCHGRR)
    // Direct-exchange all-gather. Read own shard into slot 0 and push it (compute starts). Then send our
    // shard to the G-1 ahead peers with the SAME write-then-signal ordering the ring uses (payload write
    // then semaphore-inc to the SAME peer on the SAME NoC -> ordered, NO writes_flushed), so each peer's
    // slot is exposed as ITS OWN write lands (true per-write producer/consumer overlap). We push our
    // received slots in compute order (slot d from the core d-behind). Ring cb0 layout (slot d = shard
    // rp-d), so in1/compute unchanged. Runtime args: 17..17+G-2 = per-slot sem ids; 17+G-1.. = ahead peers.
    // DIAG_IN0_XCHG    = eager: issue all G-1 (write+signal) up front, then consume -> depth 1, G-1 in flight.
    // DIAG_IN0_XCHGRR  = round-robin: per round d, (write+signal peer d) then wait+push OUR slot d before
    //                    advancing -> 1 transfer/core/round (less burst congestion), still incremental.
    {
        uint32_t p = base0;
        for (uint32_t wb = 0; wb < W; ++wb) {
            const uint32_t sb = ring_pos * W + wb;
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t k = 0; k < K_block; ++k) {
                    const uint32_t l = sb * K_block + k;
                    if (m < valid_m && l < valid_k) {
                        noc_async_read_page((m_start + m) * Kt + (k_start + l), in0, p);
                    } else {
                        zero_tile(p);
                    }
                    p += tile_bytes;
                }
            }
        }
        noc_async_read_barrier();
    }
    cb_push_back(in0_cb, W * in0_blk);          // slot 0 ready -> compute starts
    constexpr uint32_t XPEER = 17u + (G - 1u);  // first peer-coord arg
    auto send_d = [&](uint32_t d) {             // write own shard to peer d's slot d, then signal (ring order)
        uint32_t px = get_arg_val<uint32_t>(XPEER + (d - 1) * 2), py = get_arg_val<uint32_t>(XPEER + 1 + (d - 1) * 2);
        uint32_t sem = get_semaphore(get_arg_val<uint32_t>(17 + (d - 1)));
        noc_async_write(base0, get_noc_addr(px, py, base0 + d * shard_bytes), shard_bytes);
        noc_semaphore_inc(get_noc_addr(px, py, sem), 1);  // NO flush: payload-then-inc to same peer is ordered
    };
    auto recv_d = [&](uint32_t d) {  // wait our slot d (written by the core d-behind) then push it
        volatile tt_l1_ptr uint32_t* sp =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(17 + (d - 1))));
        noc_semaphore_wait_min(sp, 1);
        cb_push_back(in0_cb, W * in0_blk);
    };
#ifdef DIAG_IN0_XCHG
    for (uint32_t d = 1; d < G; ++d) {
        send_d(d);
    }
    for (uint32_t d = 1; d < G; ++d) {
        recv_d(d);
    }
#else  // DIAG_IN0_XCHGRR
    for (uint32_t d = 1; d < G; ++d) {
        send_d(d);
        recv_d(d);
    }
#endif
    noc_async_write_barrier();
#elif defined(IN0_CHUNK)
    // Finer-grained ring streaming (diagnostic): publish C-block bundles instead of the whole W-block shard
    // (C=1 is K-block-granular). Preserves ring order, physical placement, PARETO order, CB0 layout (each
    // chunk lives at its existing shard offset), K traversal, and the in0/in1 pairing -- only the read/
    // forward/publish GRANULARITY changes. For local step 0 each chunk is read + barriered + published
    // SEPARATELY (to actually expose the read/compute overlap). Forward writes the chunk to the next core's
    // slot (step+1) at the same chunk offset, payload-then-signal (same-NoC ordered, like the whole-shard
    // ring; no flush). Cumulative per-chunk fwd credit = (step-1)*chunks + chunk + 1. Final write barrier
    // retained (CB0 stays resident until the end). C=W would be byte-identical to the baseline below.
    constexpr uint32_t Cw = (IN0_CHUNK < W) ? IN0_CHUNK : W;  // min(C, W); partial final bundle handled by cnt
    constexpr uint32_t chunks = (W + Cw - 1u) / Cw;           // ceil(W / C)
    for (uint32_t step = 0; step < G; ++step) {
        for (uint32_t ch = 0; ch < chunks; ++ch) {
            const uint32_t cstart = ch * Cw;
            const uint32_t cnt = ((W - cstart) < Cw) ? (W - cstart) : Cw;  // min(C, W - cstart)
            const uint32_t coff = step * shard_bytes + cstart * in0_blk_bytes;
            const uint32_t cslot = base0 + coff;
            if (step == 0) {
                uint32_t p = cslot;
                for (uint32_t wb = cstart; wb < cstart + cnt; ++wb) {
                    const uint32_t sb = ring_pos * W + wb;  // capacity-local block index of own shard
                    for (uint32_t m = 0; m < M_block; ++m) {
                        for (uint32_t k = 0; k < K_block; ++k) {
                            const uint32_t l = sb * K_block + k;
                            if (m < valid_m && l < valid_k) {
                                noc_async_read_page((m_start + m) * Kt + (k_start + l), in0, p);
                            } else {
                                zero_tile(p);  // pad M row / K tail -> exact 0.0
                            }
                            p += tile_bytes;
                        }
                    }
                }
                noc_async_read_barrier();  // per-chunk barrier: publish this bundle before reading the next
            } else {
                noc_semaphore_wait_min(fwd_ptr, (step - 1u) * chunks + ch + 1u);  // cumulative chunk arrivals
            }
            if (step + 1u < G) {  // forward this chunk to next core's slot (step+1), same chunk offset
                noc_async_write(
                    cslot, get_noc_addr(fwd_next_x, fwd_next_y, base0 + coff + shard_bytes), cnt * in0_blk_bytes);
                noc_semaphore_inc(get_noc_addr(fwd_next_x, fwd_next_y, fwd_addr), 1);  // payload-then-signal
            }
            cb_push_back(in0_cb, cnt * in0_blk);  // compute consumes this bundle (cnt blocks) in K order
        }
    }
    noc_async_write_barrier();  // all ring forwards landed (CB0 resident until end)
#else
    for (uint32_t step = 0; step < G; ++step) {
        uint32_t slot = base0 + step * shard_bytes;
        if (step == 0) {
            // read our OWN shard (shard index = ring_pos) into slot 0
            uint32_t p = slot;
            for (uint32_t wb = 0; wb < W; ++wb) {
                [[maybe_unused]] uint32_t sb = ring_pos * W + wb;  // capacity-local block index of own shard
                for (uint32_t m = 0; m < M_block; ++m) {
                    for (uint32_t k = 0; k < K_block; ++k) {
                        // DIAG_SKIP_IN0_READ: suppress this core's step-0 in0 DRAM reads + read barrier. Pointer
                        // advancement, CB production, ring forwarding, ring semaphores, compute consumption,
                        // reduction, and output are all preserved. NOT replaced with zero-fill (that would
                        // measure L1 init instead of removing the read); the slot holds garbage.
#ifndef DIAG_SKIP_IN0_READ
                        const uint32_t l = sb * K_block + k;  // capacity-local K index within the slice
                        if (m < valid_m && l < valid_k) {
                            noc_async_read_page((m_start + m) * Kt + (k_start + l), in0, p);
                        } else {
                            zero_tile(p);  // pad M row or K tail -> local zero (no DRAM read)
                        }
#endif
                        p += tile_bytes;
                    }
                }
            }
#ifndef DIAG_SKIP_IN0_READ
            noc_async_read_barrier();
#endif
        } else {
            noc_semaphore_wait_min(fwd_ptr, step);  // prev forwarded a shard into our slot `step`
        }
        if (step + 1 < G) {  // forward this slot to the next core's slot (step+1)
            // DIAG_SKIP_IN0_FORWARD: suppress the ring payload write but STILL signal the next core, so every
            // ring step progresses (no consumer deadlock) and the semaphore-chain latency + CB schedule are
            // preserved. Isolates payload-forwarding cost, not ring synchronization.
#ifndef DIAG_SKIP_IN0_FORWARD
            uint64_t dst = get_noc_addr(fwd_next_x, fwd_next_y, base0 + (step + 1) * shard_bytes);
            noc_async_write(slot, dst, shard_bytes);
#endif
            noc_semaphore_inc(get_noc_addr(fwd_next_x, fwd_next_y, fwd_addr), 1);
        }
        cb_push_back(in0_cb, W * in0_blk);  // compute consumes this shard (W blocks)
    }
    noc_async_write_barrier();  // all ring forwards landed
#endif  // DIAG_IN0_SCATTER

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
#ifdef DIAG_BARRIER_DRAIN
            noc_async_write_barrier();  // A/B baseline: wait remote completion before reusing the slot
#else
            noc_async_writes_flushed();  // DEFAULT (pipelined): pages departed L1 -> out_cb slot safe to reuse
#endif
            cb_pop_front(out_cb, out_blk);
        }
#ifndef DIAG_BARRIER_DRAIN
        noc_async_write_barrier();  // pipelined: single deferred completion barrier before return (no atomics)
#endif
        return;
    }

    // Pk > 1: linear reduction chain.
#ifdef DIAG_NO_REDUCE
    // NO_REDUCE: every compute core took the bottom-band copy path (compute.cpp forces copy_block), so each
    // produced its OWN matmul partial into out_cb. Bypass ALL reduction traffic (credits, receives, partial
    // -sum forwards) and never touch cb_reduce. Non-top bands consume and DISCARD their partial; only the
    // original top band writes its partial to DRAM (exactly one top-band output write, unchanged core count
    // / CB alloc / output production). This removes reduction communication AND the reduction-add compute
    // together — a COMBINED counterfactual, not a pure reduction-comm isolation.
    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
        cb_wait_front(out_cb, out_blk);
        uint32_t r = get_read_ptr(out_cb);
        if (is_top) {
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
#else
    // cb_reduce holds 2 blocks (double-buffered). reduce_base captured ONCE BEFORE any cb_reduce use (the
    // write ptr drifts after receives).
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
#ifdef DIAG_BARRIER_DRAIN
            noc_async_write_barrier();        // A/B baseline: wait remote completion, then signal
            noc_semaphore_inc(next_recv, 1);  // block nb delivered
#else
            // DEFAULT (pipelined): payload THEN signal to the SAME peer on the SAME NoC (ordered, like the in0
            // ring) so the receiver never observes readiness before its partial-sum has landed. Flush (not a
            // full barrier) so the out_cb source slot is reusable; completion is deferred to the final barrier.
            noc_semaphore_inc(next_recv, 1);  // block nb delivered (ordered after the payload write)
            noc_async_writes_flushed();       // payload departed L1 -> out_cb slot safe to reuse
#endif
        } else {
            const uint32_t n_off = n_start + nb * N_block;  // global N tile of this subblock
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t n = 0; n < N_block; ++n) {
                    if (m < valid_m && (nb * N_block + n) < valid_n) {  // write only valid_m x valid_n
                        noc_async_write_page((m_start + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
                    }
                }
            }
#ifdef DIAG_BARRIER_DRAIN
            noc_async_write_barrier();
#else
            noc_async_writes_flushed();  // output pages departed L1 -> out_cb slot safe to reuse
#endif
        }
        cb_pop_front(out_cb, out_blk);
    }
#ifndef DIAG_BARRIER_DRAIN
    // Pipelined (default): single deferred completion before return — drain this core's forwarded partial-sums
    // / DRAM output writes AND the non-posted reduction-readiness semaphore atomics (noc_semaphore_inc), so no
    // in-flight NoC transaction outlives the program (writes_flushed above only guarantees source-L1 reuse).
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
#endif
}
