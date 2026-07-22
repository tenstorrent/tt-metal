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
// Cleaned unified-only port of the prototype in0_ring_writer.cpp (ring all-gather + reduction chain). The
// rejected in0-delivery variants (scatter / replicated-ring / direct-exchange) and the skip-read/forward
// ablations have been removed; the only test-only diagnostics that remain here are DIAG_NO_REDUCE and the
// DIAG_BARRIER_DRAIN write-sync A/B (both #ifdef, absent from the mask-0 public compile).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Test-only causal timing zones (compile-gated; profiler injects DeviceZoneScopedN). Mask 0 => no-op =>
// byte-identical. See RegimeADiag::DIAG_ZONES.
#ifdef DIAG_ZONES
#define RA_ZONE(n) DeviceZoneScopedN(n)
#else
#define RA_ZONE(n)
#endif

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
    // Optional fused-epilogue operands (appended by the factory in this order: bias, then ternary_a/_b).
    // Present only when the matching define is set, so the no-fusion compile is unchanged.
#if defined(FUSE_BIAS)
    constexpr auto bias_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
#if defined(FUSE_TERNARY)
    constexpr auto ta_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto tb_args = TensorAccessorArgs<ta_args.next_compile_time_args_offset()>();
#endif
#elif defined(FUSE_TERNARY)
    constexpr auto ta_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto tb_args = TensorAccessorArgs<ta_args.next_compile_time_args_offset()>();
#endif

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
#ifdef REDTREE
    // Fan-in-2 reduction-tree runtime args (index 17+; never combined with fusion/chunks — see the factory).
    const uint32_t tree_num_recv = get_arg_val<uint32_t>(17);      // incoming partials this core sums (0/1/2)
    const uint32_t tree_parent_nrecv = get_arg_val<uint32_t>(18);  // parent's num_recv (sender slot cadence)
    const uint32_t tree_channel = get_arg_val<uint32_t>(19);       // channel this core writes at its parent
    const uint32_t red_sem2_id = get_arg_val<uint32_t>(20);        // channel-1 receive semaphore id
    const uint32_t red_src0_x = get_arg_val<uint32_t>(21);         // channel-0 source (for reverse credit)
    const uint32_t red_src0_y = get_arg_val<uint32_t>(22);
    const uint32_t red_src1_x = get_arg_val<uint32_t>(23);  // channel-1 source
    const uint32_t red_src1_y = get_arg_val<uint32_t>(24);
#endif
#ifdef RSCATTER
    // Ring reduce-scatter runtime args (index 17+; unfused only — never combined with fusion/chunks/other diag).
    const uint32_t rs_next_x = get_arg_val<uint32_t>(17);  // next core in the Pk ring (I send to it)
    const uint32_t rs_next_y = get_arg_val<uint32_t>(18);
    const uint32_t rs_prev_x = get_arg_val<uint32_t>(19);  // prev core (it sends to me)
    const uint32_t rs_prev_y = get_arg_val<uint32_t>(20);
    const uint32_t rs_owned_row = get_arg_val<uint32_t>(21);  // block M-row this core owns + writes to DRAM
#endif

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

    // ---- Optional fused-epilogue reads. This core produces the fused output (and therefore reads the
    // bias/residual/gate operands into CBs c_4/c_5/c_6 for compute) only when it is the reduction ROOT:
    //   Pk==1 (use_reduce==0): every core is its own root; Pk>1: only the top band (is_top).
    // The compute kernel consumes these CBs and applies the epilogue exactly once. All gated by defines so
    // the no-fusion compile is byte-identical. ----
#if defined(FUSE_BIAS) || defined(FUSE_TERNARY) || defined(OUT_CHUNKS)
    const bool fuse_root = (use_reduce == 0u) || (is_top != 0u);
    uint32_t fidx = 17u;  // fusion/chunk runtime args follow the base 17 (never combined with diag ablations)
#endif
#if defined(FUSE_BIAS)
    constexpr uint32_t bias_cb = 4;
    const uint32_t bias_addr = get_arg_val<uint32_t>(fidx++);
    const auto bias = TensorAccessor(bias_args, bias_addr, tile_bytes);
#endif
#if defined(FUSE_TERNARY)
    constexpr uint32_t ta_cb = 5, tb_cb = 6;
    const uint32_t ta_addr = get_arg_val<uint32_t>(fidx++);
    const uint32_t tb_addr = get_arg_val<uint32_t>(fidx++);
    const uint32_t bcast_gate = get_arg_val<uint32_t>(fidx++);
#if defined(TERNARY_B_IS_FLOAT32)
    constexpr uint32_t gate_tile_bytes = tile_bytes * 2u;  // fp32 gate tile = 2x bf16
#else
    constexpr uint32_t gate_tile_bytes = tile_bytes;
#endif
    const auto ta = TensorAccessor(ta_args, ta_addr, tile_bytes);
    const auto tb = TensorAccessor(tb_args, tb_addr, gate_tile_bytes);
#endif
    auto zero_bytes = [](uint32_t addr, uint32_t nbytes) {
        volatile tt_l1_ptr uint32_t* q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        const uint32_t nw = nbytes / 4u;
        for (uint32_t i = 0; i < nw; ++i) {
            q[i] = 0;
        }
    };
    (void)zero_bytes;

    // Feed the fused-epilogue operands for output sub-block `nb` into c_4/c_5/c_6, matching the consumption
    // order/shape of compute's add_bias_block / add_bias_and_addcmul_block. Operands are indexed by GLOBAL
    // (m,n) on the FULL-N stride (Nt); invalid tail positions (m>=valid_m or local col>=valid_n) are zeroed
    // (never read out of range) — their fused output columns/rows are not written to DRAM.
    [[maybe_unused]] auto feed_fused = [&](uint32_t nb) {
        [[maybe_unused]] const uint32_t n_off = n_start + nb * N_block;  // global N tile base of this sub-block
#if defined(FUSE_BIAS)
        cb_reserve_back(bias_cb, N_block);
        uint32_t pb = get_write_ptr(bias_cb);
        for (uint32_t n = 0; n < N_block; ++n) {
            if ((nb * N_block + n) < valid_n) {
                noc_async_read_page(n_off + n, bias, pb);  // bias [1,N]: page = global N tile
            } else {
                zero_tile(pb);
            }
            pb += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(bias_cb, N_block);
#endif
#if defined(FUSE_TERNARY)
        if (bcast_gate) {  // gate [1,N]: one row for the whole sub-block
            cb_reserve_back(tb_cb, N_block);
            uint32_t pg = get_write_ptr(tb_cb);
            for (uint32_t n = 0; n < N_block; ++n) {
                if ((nb * N_block + n) < valid_n) {
                    noc_async_read_page(n_off + n, tb, pg);
                } else {
                    zero_bytes(pg, gate_tile_bytes);
                }
                pg += gate_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(tb_cb, N_block);
        }
        for (uint32_t m = 0; m < M_block; ++m) {  // residual [M,N] (+ full gate [M,N]) one M-row at a time
            cb_reserve_back(ta_cb, N_block);
            uint32_t pa = get_write_ptr(ta_cb);
            for (uint32_t n = 0; n < N_block; ++n) {
                if (m < valid_m && (nb * N_block + n) < valid_n) {
                    noc_async_read_page((m_start + m) * Nt + (n_off + n), ta, pa);
                } else {
                    zero_tile(pa);
                }
                pa += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(ta_cb, N_block);
            if (!bcast_gate) {  // gate [M,N]: this M-row
                cb_reserve_back(tb_cb, N_block);
                uint32_t pg = get_write_ptr(tb_cb);
                for (uint32_t n = 0; n < N_block; ++n) {
                    if (m < valid_m && (nb * N_block + n) < valid_n) {
                        noc_async_read_page((m_start + m) * Nt + (n_off + n), tb, pg);
                    } else {
                        zero_bytes(pg, gate_tile_bytes);
                    }
                    pg += gate_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(tb_cb, N_block);
            }
        }
#endif
    };

    // Chunked output support (regime_a_matmul_split): route each output tile to the chunk buffer that owns
    // its global N column. chunk = global_n / out_ntc, col = global_n % out_ntc; write page (m)*out_ntc+col
    // into chunk-buffer `chunk`. All chunk buffers share the output TensorAccessorArgs (same [M, N/chunks]
    // spec), differing only by base address. Not compiled unless OUT_CHUNKS (chunks>1); chunks==1 uses the
    // original single-buffer write below (byte-identical).
#if defined(OUT_CHUNKS)
    constexpr uint32_t kMaxChunks = 16u;
    const uint32_t n_chunks = get_arg_val<uint32_t>(fidx++);
    const uint32_t out_ntc = get_arg_val<uint32_t>(fidx++);  // per-chunk N tiles
    uint32_t chunk_addr[kMaxChunks];
    chunk_addr[0] = out_addr;  // chunk 0 == writer arg 1
    for (uint32_t c = 1; c < n_chunks; ++c) {
        chunk_addr[c] = get_arg_val<uint32_t>(fidx++);
    }
    auto write_out_tile = [&](uint32_t m_row, uint32_t gn, uint32_t l1_addr) {
        const uint32_t chunk = gn / out_ntc;
        const uint32_t col = gn - chunk * out_ntc;
        const auto oc = TensorAccessor(out_args, chunk_addr[chunk], tile_bytes);
        noc_async_write_page(m_row * out_ntc + col, oc, l1_addr);
    };
#endif

    // ---- PHASE 1: in0 ring all-gather (balanced tails: read only valid M rows / valid K, else zero) ----
    {
        RA_ZONE("Z_RING");
        cb_reserve_back(in0_cb, K_num_blocks * in0_blk);
        const uint32_t base0 = get_write_ptr(in0_cb);
    for (uint32_t step = 0; step < G; ++step) {
        uint32_t slot = base0 + step * shard_bytes;
        if (step == 0) {
            // read our OWN shard (shard index = ring_pos) into slot 0
            uint32_t p = slot;
            for (uint32_t wb = 0; wb < W; ++wb) {
                const uint32_t sb = ring_pos * W + wb;  // capacity-local block index of own shard
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
#ifdef DIAG_RINGDRAIN
        // Test-only: source-lifetime flush only; remote completion deferred to the kernel-exit barrier
        // (Pk>1 path drains write+atomics at 392-393). Remote landing is sem-synchronized (payload->sem).
        noc_async_writes_flushed();
#else
        noc_async_write_barrier();  // all ring forwards landed
#endif
    }  // end Z_RING

    // ---- PHASE 2: output / split-K reduction over the N_bpc output blocks ----
    RA_ZONE("Z_PHASE2");  // function-scope zone: destructor fires at return / kernel end (reduce + output)
    constexpr uint32_t out_blk_bytes = out_blk * tile_bytes;

    if constexpr (!use_reduce) {
        // Pk == 1: every core is bottom AND top; compute produced its full block into out_cb -> write DRAM.
        for (uint32_t nb = 0; nb < N_bpc; ++nb) {
#if defined(FUSE_BIAS) || defined(FUSE_TERNARY)
            feed_fused(nb);  // supply bias/residual/gate; compute fuses -> out_cb (before we wait on it)
#endif
            cb_wait_front(out_cb, out_blk);
            uint32_t r = get_read_ptr(out_cb);
            const uint32_t n_off = n_start + nb * N_block;  // global N tile of this subblock
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t n = 0; n < N_block; ++n) {
                    if (m < valid_m && (nb * N_block + n) < valid_n) {  // write only valid_m x valid_n
#if defined(OUT_CHUNKS)
                        write_out_tile(m_start + m, n_off + n, r + (m * N_block + n) * tile_bytes);
#else
                        noc_async_write_page((m_start + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
#endif
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
#elif defined(REDTREE)
    // ---- Fan-in-2 reduction TREE (test-only; Pk==4). Depth 2 vs the chain's depth 3. ----
    // This core RECEIVES tree_num_recv partials (0 = leaf, 1 = inner, 2 = root) on disjoint channels, then
    // (if not the root) FORWARDS its summed block up to its parent on channel `tree_channel`. The reduce-CB
    // (cb_reduce, 2 slots) is addressed exactly like the chain: the receiver reserves one slot per round and
    // the sender writes reduce_base + slot*out_blk_bytes where slot = (nb*parent_nrecv + channel) % 2 — this
    // matches the receiver's per-round FIFO reservation for both parent kinds (inner parent nrecv=1 =>
    // double-buffered nb%2 like the chain; root parent nrecv=2 => channel-fixed slots 0/1). Channel 0 uses
    // red_sem, channel 1 uses red_sem2 (disjoint counters => no cross-channel fungibility).
    const uint32_t reduce_base = get_write_ptr(cb_reduce);
    const uint32_t red_addr = get_semaphore(red_sem_id);    // my channel-0 receive counter
    const uint32_t red2_addr = get_semaphore(red_sem2_id);  // my channel-1 receive counter
    volatile tt_l1_ptr uint32_t* red_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_addr);
    volatile tt_l1_ptr uint32_t* red2_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red2_addr);
    const uint32_t redfree_addr = get_semaphore(redfree_sem_id);
    volatile tt_l1_ptr uint32_t* redfree_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(redfree_addr);
    // Reverse-credit targets: my channel-0/1 sources' redfree semaphores (I free their reduce slots).
    const uint64_t src0_redfree = get_noc_addr(red_src0_x, red_src0_y, redfree_addr);
    const uint64_t src1_redfree = get_noc_addr(red_src1_x, red_src1_y, redfree_addr);
    // Forward target: my parent's channel-`tree_channel` receive counter (same L1 offset on every core).
    const uint32_t parent_recv_addr = (tree_channel == 0u) ? red_addr : red2_addr;
    const uint64_t next_recv = get_noc_addr(red_next_x, red_next_y, parent_recv_addr);

    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
        if (tree_num_recv >= 1u) {  // channel 0 (from red_src0), reduced first
            cb_reserve_back(cb_reduce, out_blk);
            noc_semaphore_inc(src0_redfree, 1);
            {
                RA_ZONE("Z_P2_RECVWAIT");
                noc_semaphore_wait_min(red_ptr, nb + 1);
            }
            cb_push_back(cb_reduce, out_blk);  // compute reduce-adds it (round 0)
        }
        if (tree_num_recv >= 2u) {  // channel 1 (from red_src1), reduced second -> out_cb
            cb_reserve_back(cb_reduce, out_blk);
            noc_semaphore_inc(src1_redfree, 1);
            {
                RA_ZONE("Z_P2_RECVWAIT");
                noc_semaphore_wait_min(red2_ptr, nb + 1);
            }
            cb_push_back(cb_reduce, out_blk);  // compute reduce-adds it (round 1)
        }
        {
            RA_ZONE("Z_P2_OUTWAIT");
            cb_wait_front(out_cb, out_blk);  // compute produced this core's (summed) block nb
        }
        uint32_t r = get_read_ptr(out_cb);
        if (tree_num_recv < 2u) {                         // not the root: forward my block up to my parent
            noc_semaphore_wait_min(redfree_ptr, nb + 1);  // parent freed my target slot
            const uint32_t slot = (nb * tree_parent_nrecv + tree_channel) & 1u;
            uint64_t dst = get_noc_addr(red_next_x, red_next_y, reduce_base + slot * out_blk_bytes);
            noc_async_write(r, dst, out_blk_bytes);
            noc_semaphore_inc(next_recv, 1);  // ordered after payload (same peer + NoC, like the in0 ring)
            noc_async_writes_flushed();       // payload departed L1 -> out_cb slot reusable
        } else {                              // root (is_top): write the final block to DRAM
            RA_ZONE("Z_P2_OUTWRITE");
            const uint32_t n_off = n_start + nb * N_block;
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t n = 0; n < N_block; ++n) {
                    if (m < valid_m && (nb * N_block + n) < valid_n) {
#if defined(OUT_CHUNKS)
                        write_out_tile(m_start + m, n_off + n, r + (m * N_block + n) * tile_bytes);
#else
                        noc_async_write_page((m_start + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
#endif
                    }
                }
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(out_cb, out_blk);
    }
    // Single deferred completion (like the chain default): drain forwarded partial-sums / DRAM writes AND the
    // non-posted reduction-readiness / reverse-credit atomics so no NoC transaction outlives the program.
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#elif defined(RSCATTER)
    // ---- Ring REDUCE-SCATTER (test-only; Pk==4, one output block M_block x N_block, N_bpc==1). ----
    // P = Pk = 4 cores in an optimized cyclic order. Chunk = one block M-row (N_block tiles). Every round each
    // core sends one chunk to `next` and receives one from `prev` into cb_recv (=cb_reduce); compute adds its
    // own resident partial row and forwards the running sum, so after P-1 rounds each core holds ONE fully-
    // reduced row-chunk (its rs_owned_row) and writes it to DRAM. Reuses the in0-ring payload->credit protocol:
    // red_sem = "prev delivered a chunk into my cb_recv", redfree_sem = "next freed my send slot".
    constexpr uint32_t P = 4u;       // Pk (RSCATTER requires Pk==4)
    constexpr uint32_t cb_send = 4;  // compute -> writer send-chunk CB (bf16), double-buffered
    constexpr uint32_t cb_recv = 5;  // incoming-chunk CB (bf16), EXACTLY 2 slots (period matches t%2)
    constexpr uint32_t chunk_tiles = N_block;
    constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
    const uint32_t recv_base = get_write_ptr(cb_recv);  // my cb_recv L1 base (== same offset on every core)
    const uint32_t rs_recv_addr = get_semaphore(red_sem_id);
    volatile tt_l1_ptr uint32_t* rs_recv_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_recv_addr);
    const uint32_t rs_free_addr = get_semaphore(redfree_sem_id);
    volatile tt_l1_ptr uint32_t* rs_free_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_free_addr);
    const uint64_t prev_rs_free = get_noc_addr(rs_prev_x, rs_prev_y, rs_free_addr);  // I credit prev
    const uint64_t next_rs_recv = get_noc_addr(rs_next_x, rs_next_y, rs_recv_addr);  // I signal next

    for (uint32_t t = 0; t < P - 1u; ++t) {
        // Post my recv-slot credit to prev FIRST (so all cores credit before any blocks on its own send credit
        // -> no ring deadlock). cb_reserve_back waits until compute consumed my previous incoming chunk.
        cb_reserve_back(cb_recv, chunk_tiles);
        noc_semaphore_inc(prev_rs_free, 1);  // tell prev: my cb_recv slot free for round t
        // Send my staged chunk to next (round 0 = my own row `ring_pos`; rounds >0 = the sum compute produced).
        cb_wait_front(cb_send, chunk_tiles);
        {
            RA_ZONE("Z_RS_SENDWAIT");
            noc_semaphore_wait_min(rs_free_ptr, t + 1);  // next freed my send slot
        }
        const uint32_t slot = t & 1u;  // double-buffered (2 slots)
        uint64_t dst = get_noc_addr(rs_next_x, rs_next_y, recv_base + slot * chunk_bytes);
        noc_async_write(get_read_ptr(cb_send), dst, chunk_bytes);
        noc_semaphore_inc(next_rs_recv, 1);  // ordered after payload (same peer+NoC, like the in0 ring)
        noc_async_writes_flushed();          // payload departed L1 -> cb_send slot reusable
        cb_pop_front(cb_send, chunk_tiles);
        // Complete my receive for round t.
        {
            RA_ZONE("Z_RS_RECVWAIT");
            noc_semaphore_wait_min(rs_recv_ptr, t + 1);  // prev delivered chunk t into my cb_recv slot t%2
        }
        cb_push_back(cb_recv, chunk_tiles);  // compute adds its own row + (forwards | writes owned)
    }
    // Final round produced my fully-reduced owned row-chunk into out_cb -> write it to DRAM (contiguous N).
    {
        RA_ZONE("Z_RS_OUTWRITE");
        cb_wait_front(out_cb, chunk_tiles);
        const uint32_t r = get_read_ptr(out_cb);
        if (rs_owned_row < valid_m) {
            for (uint32_t n = 0; n < N_block; ++n) {
                if (n < valid_n) {
                    noc_async_write_page((m_start + rs_owned_row) * Nt + (n_start + n), out, r + n * tile_bytes);
                }
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(out_cb, chunk_tiles);
    }
    // Single deferred completion: drain forwarded chunks / DRAM writes AND the non-posted deliver/credit atomics.
    noc_async_write_barrier();
    noc_async_atomic_barrier();
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
            cb_reserve_back(cb_reduce, out_blk);  // wait our compute freed slot (nb-2)
            noc_semaphore_inc(prev_redfree, 1);   // tell prev: our slot (nb%2) is free for block nb
            {
                RA_ZONE("Z_P2_RECVWAIT");                 // wait prev core's forwarded partial (chain latency)
                noc_semaphore_wait_min(red_ptr, nb + 1);  // prev forwarded block nb into it
            }
            cb_push_back(cb_reduce, out_blk);  // compute reduce_add's it -> out_cb, pops cb_reduce
        }
#if defined(FUSE_BIAS) || defined(FUSE_TERNARY)
        if (is_top) {
            feed_fused(nb);  // ROOT only: supply bias/residual/gate for compute's single fused epilogue
        }
#endif
        {
            RA_ZONE("Z_P2_OUTWAIT");         // wait compute to produce the reduced block into out_cb
            cb_wait_front(out_cb, out_blk);  // compute produced reduced (+ fused at top) block nb
        }
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
            RA_ZONE("Z_P2_OUTWRITE");  // ROOT: issue output DRAM pages + flush (the reduction tail on the wall)
            const uint32_t n_off = n_start + nb * N_block;  // global N tile of this subblock
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t n = 0; n < N_block; ++n) {
                    if (m < valid_m && (nb * N_block + n) < valid_n) {  // write only valid_m x valid_n
#if defined(OUT_CHUNKS)
                        write_out_tile(m_start + m, n_off + n, r + (m * N_block + n) * tile_bytes);
#else
                        noc_async_write_page((m_start + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
#endif
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
