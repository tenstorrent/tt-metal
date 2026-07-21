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
        noc_async_write_barrier();  // all ring forwards landed
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
#if defined(FUSE_BIAS) || defined(FUSE_TERNARY)
        if (is_top) {
            feed_fused(nb);  // ROOT only: supply bias/residual/gate for compute's single fused epilogue
        }
#endif
        cb_wait_front(out_cb, out_blk);  // compute produced reduced (+ fused at top) block nb
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
