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
// Production port of the prototype in0_ring_writer.cpp: ring all-gather + linear split-K reduction chain +
// output write. Write sync is pipelined (per-block source-lifetime flush + one deferred completion barrier).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Per-phase timing zones, compile-gated. Absent (mask 0) => no-op => production is byte-identical.
#if defined(RA_PROFILE_ZONES)
#define RA_ZONE(n) DeviceZoneScopedN(n)
#else
#define RA_ZONE(n)
#endif

// Temporary hang-diagnosis markers (compile-gated, never on in production).
#if defined(RA_DBG)
#include "api/debug/dprint.h"
#define RA_DBG_P(tag, v) DPRINT(tag "={}\n", (uint32_t)(v))
#else
#define RA_DBG_P(tag, v)
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
    constexpr uint32_t use_reduce = get_compile_time_arg_val(13);      // 0 when Pk==1; else cb_reduce DEPTH
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
    uint32_t fidx = 17u;  // fusion/chunk runtime args follow the base 17 writer args
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

    // ---- TEST-ONLY in0-read-skip flag. Appended at index 17 by the factory ONLY for the unfused/
    // single-output diagnostic build (bit0/bit1 set); never present in production, so this read + the guard
    // below compile to nothing in the mask-0 binary. 1 => skip this core's in0 DRAM read (leave stale L1,
    // preserve CB reserve/push/pop, pointer advance, barrier, ring forwarding, semaphores, compute). ----
    constexpr uint32_t kDiagBase0 = 17u;
#if defined(SKIP_ALL_IN0_READ) || defined(SKIP_REDUNDANT_IN0_READ)
    const uint32_t in0_skip = get_arg_val<uint32_t>(kDiagBase0);
    constexpr uint32_t kDiagBase1 = kDiagBase0 + 1u;
#else
    constexpr uint32_t kDiagBase1 = kDiagBase0;
#endif
#define DIAG_ARG_BASE kDiagBase1
    // ---- TEST-ONLY ring-forward PERTURBATION args (bit6 FWD_NEAR): nearest program core on this core's
    // writer NoC. Payload only; the readiness semaphore below still targets the TRUE ring successor, so the
    // ring's step count / dependency chain is unchanged and only hop distance is removed. ----
#if defined(FWD_NEAR)
    const uint32_t near_x = get_arg_val<uint32_t>(kDiagBase1);
    const uint32_t near_y = get_arg_val<uint32_t>(kDiagBase1 + 1);
    constexpr uint32_t kDiagBase2 = kDiagBase1 + 2u;
#else
    constexpr uint32_t kDiagBase2 = kDiagBase1;
#endif
    // ---- MEET-IN-THE-MIDDLE reduction args (bit20). red_nrecv: incoming partials at this core (2 only at the
    // meet root). red_channel: which of the root's two channels THIS core sends on - channel 0 uses red_sem /
    // red_prev, channel 1 uses red_sem2 / red_prev2, so the two arrivals can never be confused for each other
    // (a single shared counter would be fungible and is exactly what corrupted earlier reduction work). ----
#if defined(REDUCE_MEET)
    const uint32_t red_nrecv = get_arg_val<uint32_t>(kDiagBase2);      // my incoming partials (0, 1 or 2)
    const uint32_t red_send_ord = get_arg_val<uint32_t>(kDiagBase2 + 1);  // my ordinal at my destination
    const uint32_t red_prev2_x = get_arg_val<uint32_t>(kDiagBase2 + 2);
    const uint32_t red_prev2_y = get_arg_val<uint32_t>(kDiagBase2 + 3);
    const uint32_t red_dest_nrecv = get_arg_val<uint32_t>(kDiagBase2 + 4);  // inputs my DESTINATION has
    const uint32_t red_cb_slots = get_arg_val<uint32_t>(kDiagBase2 + 5);    // cb_reduce depth in blocks
#endif

    // ---- RING REDUCE-SCATTER args (index 17+; unfused/single-chunk/no-diag, so index 17 is free). ----
#if defined(RSCATTER)
    const uint32_t rs_next_x = get_arg_val<uint32_t>(17);  // next core in the Pk cycle (I send to it)
    const uint32_t rs_next_y = get_arg_val<uint32_t>(18);
    const uint32_t rs_prev_x = get_arg_val<uint32_t>(19);  // prev core (it sends to me)
    const uint32_t rs_prev_y = get_arg_val<uint32_t>(20);
    const uint32_t rs_owned_chunk = get_arg_val<uint32_t>(21);  // tile-chunk this core owns + writes to DRAM
    const uint32_t rs_P = get_arg_val<uint32_t>(22);            // cycle size = Pk
    const uint32_t rs_T = get_arg_val<uint32_t>(23);            // tiles per output sub-block = M_block*N_block
#endif

    // ---- PHASE 1: in0 ring all-gather (balanced tails: read only valid M rows / valid K, else zero) ----
    cb_reserve_back(in0_cb, K_num_blocks * in0_blk);
    const uint32_t base0 = get_write_ptr(in0_cb);
    for (uint32_t step = 0; step < G; ++step) {
        uint32_t slot = base0 + step * shard_bytes;
        if (step == 0) {
            // read our OWN shard (shard index = ring_pos) from DRAM into slot 0 (+ barrier).
            uint32_t p = slot;
            for (uint32_t wb = 0; wb < W; ++wb) {
                const uint32_t sb = ring_pos * W + wb;  // capacity-local block index of own shard
                for (uint32_t m = 0; m < M_block; ++m) {
                    for (uint32_t k = 0; k < K_block; ++k) {
                        const uint32_t l = sb * K_block + k;  // capacity-local K index within the slice
                        if (m < valid_m && l < valid_k) {
#if defined(SKIP_ALL_IN0_READ) || defined(SKIP_REDUNDANT_IN0_READ)
                            // Diagnostic: skip ONLY the DRAM read; leave the L1 slot's stale contents and
                            // keep the pointer advance / loop / barrier / push / ring forwarding intact so
                            // downstream work is measured unchanged (output is intentionally invalid).
                            if (!in0_skip) {
                                noc_async_read_page((m_start + m) * Kt + (k_start + l), in0, p);
                            }
#else
                            noc_async_read_page((m_start + m) * Kt + (k_start + l), in0, p);
#endif
                        } else {
                            zero_tile(p);  // pad M row or K tail -> local zero (no DRAM read)
                        }
                        p += tile_bytes;
                    }
                }
            }
            noc_async_read_barrier();
        } else {
            noc_semaphore_wait_min(fwd_ptr, step);  // wait for prev to forward a shard into our slot `step`
        }
        if (step + 1 < G) {  // forward this slot to the next core's slot (step+1) + signal
            uint64_t dst = get_noc_addr(fwd_next_x, fwd_next_y, base0 + (step + 1) * shard_bytes);
#if !defined(SKIP_IN0_RING_FORWARD)
#if defined(FWD_NEAR)
            // diagnostic: same bytes, ~1 hop. Destination is the SAME cb0 offset on a near core (identical CB
            // layout on every core, so the address is in-bounds); its slot is also written by its true
            // predecessor -> content is garbage, which is expected for a diagnostic mask.
            dst = get_noc_addr(near_x, near_y, base0 + (step + 1) * shard_bytes);
#endif
#if defined(FWD_HALF)
            noc_async_write(slot, dst, shard_bytes / 2u);  // diagnostic: half the payload, true destination
#else
            noc_async_write(slot, dst, shard_bytes);
#endif
#else
            (void)dst;  // diagnostic: drop the ring PAYLOAD write; keep the readiness/credit semaphore below
#endif
            noc_semaphore_inc(get_noc_addr(fwd_next_x, fwd_next_y, fwd_addr), 1);  // credit preserved (stale L1)
        }
        cb_push_back(in0_cb, W * in0_blk);  // compute consumes this shard (W blocks)
    }
    noc_async_write_barrier();  // all ring forwards landed

    // ---- PHASE 2: output / split-K reduction over the N_bpc output blocks ----
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
#if defined(SKIP_OUTPUT_WRITE)
                        // diagnostic: drop the DRAM payload write; keep the iteration + CB consumption below.
#elif defined(OUT_CHUNKS)
                        write_out_tile(m_start + m, n_off + n, r + (m * N_block + n) * tile_bytes);
#else
                        noc_async_write_page((m_start + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
#endif
                    }
                }
            }
            noc_async_writes_flushed();  // pipelined: pages departed L1 -> out_cb slot safe to reuse
            cb_pop_front(out_cb, out_blk);
        }
        noc_async_write_barrier();  // pipelined: single deferred completion barrier before return (no atomics)
        return;
    }

#if defined(RSCATTER) && defined(RS_STRIPED)
    // ---- Pk > 1: S-WAY STRIPED OWNER-GATHER. Only S of the group's Pk cores own output; owner j owns stripe j
    // (rs_T/S tiles). Every core writes its partial for stripe j straight to owner j, so the group sends
    // S*(Pk-1) messages instead of the full exchange's Pk*(Pk-1) -- Pk/S fewer, same total bytes. An owner keeps
    // its OWN stripe where it already is (in the fp32 intermediate CB), so there is NO loopback NoC traffic.
    //
    // Arrivals are split across TWO semaphores by sender position so the owner can reduce the first half of its
    // partials while the second half is still in flight (incremental reduction). The receive area is
    // DOUBLE-BUFFERED (generation = nb & 1), so a sender can fill generation nb while the owner still reduces
    // nb-1; a sender therefore only has to know generation nb-2 was consumed, which is what the credit counter
    // tracks (each owner credits each of its Pk-1 senders once per sub-block => S credits per sender).
    const uint32_t P = rs_P;
    const uint32_t S = get_arg_val<uint32_t>(24);
    const uint32_t my_pos = get_arg_val<uint32_t>(25);
    const uint32_t my_stripe = get_arg_val<uint32_t>(26);
    const uint32_t na = get_arg_val<uint32_t>(27);  // senders with position < na use arrival counter A
    const uint32_t b_sem_id = get_arg_val<uint32_t>(28);
    const bool is_owner = (my_stripe < S);
    const uint32_t stripe_tiles = (rs_T + S - 1u) / S;
    const uint32_t stripe_bytes = stripe_tiles * tile_bytes;
    const uint32_t gen_bytes = P * stripe_bytes;
    constexpr uint32_t cb_send = 4;
    constexpr uint32_t cb_recv = 5;
    const uint32_t recv_base = get_write_ptr(cb_recv);  // identical offset on every core
    // ARRIVAL counters are PER GENERATION (gen = nb & 1) and are RESET by the owner once consumed. A single
    // cumulative counter is WRONG: the credit wait only engages at nb>=2, so one sender may advance to nb=1
    // unimpeded and its second increment alone satisfies the owner's nb=0 threshold while a slower sender's slot
    // still holds garbage. Per-generation counters make the threshold mean "this many DISTINCT senders arrived
    // for THIS generation", because a sender increments a given generation's counter exactly once.
    const uint32_t arr_addr[2] = {get_semaphore(red_sem_id), get_semaphore(b_sem_id)};
    auto owner_pos = [&](uint32_t j) { return get_arg_val<uint32_t>(29 + j); };
    auto cred_sem = [&](uint32_t j) { return get_arg_val<uint32_t>(29 + S + j); };
    auto gcx = [&](uint32_t p) { return get_arg_val<uint32_t>(29 + 2u * S + p * 2u); };
    auto gcy = [&](uint32_t p) { return get_arg_val<uint32_t>(30 + 2u * S + p * 2u); };
    const uint32_t own_beg = is_owner ? (my_stripe * stripe_tiles) : 0u;
    const uint32_t own_end = is_owner ? ((own_beg + stripe_tiles < rs_T) ? own_beg + stripe_tiles : rs_T) : 0u;

    RA_DBG_P("RSS_start pos", my_pos);
    RA_DBG_P("  stripe", my_stripe);
    RA_DBG_P("  S", S);
    RA_DBG_P("  P", P);
    RA_DBG_P("  st_tiles", stripe_tiles);
    RA_DBG_P("  Nbpc", N_bpc);
    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
        const uint32_t n_base = nb * N_block;
        RA_DBG_P("A_top nb", nb);
        if (nb >= 2u) {
            RA_ZONE("Z_RSS_CREDITWAIT");
            // EVERY owner we write to must have consumed generation nb-2. Checking each owner's own counter is
            // required: a summed counter lets one fast owner cover for a slow one and the slow owner's live
            // buffer then gets overwritten.
            for (uint32_t j = 0; j < S; ++j) {
                if (owner_pos(j) != my_pos) {
                    volatile tt_l1_ptr uint32_t* cp =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(cred_sem(j)));
                    noc_semaphore_wait_min(cp, nb - 1u);
                }
            }
        }
        RA_DBG_P("B_waitsend nb", nb);
        cb_wait_front(cb_send, rs_T);  // compute staged the bf16 image of our whole partial
        const uint32_t img = get_read_ptr(cb_send);
        const uint32_t gen = (nb & 1u) * gen_bytes;
        RA_DBG_P("C_gotsend nb", nb);
        if (is_owner) {
            cb_reserve_back(cb_recv, P * stripe_tiles);
        }
        RA_DBG_P("D_reserved nb", nb);
        {
            RA_ZONE("Z_RSS_PAYLOAD");
            for (uint32_t j = 0; j < S; ++j) {
                const uint32_t op = owner_pos(j);
                if (op == my_pos) {
                    continue;  // no loopback: our own stripe stays in the fp32 intermediate CB
                }
                const uint32_t beg = j * stripe_tiles;
                const uint32_t nby = ((beg + stripe_tiles < rs_T) ? stripe_tiles : (rs_T - beg)) * tile_bytes;
                noc_async_write(
                    img + beg * tile_bytes,
                    get_noc_addr(gcx(op), gcy(op), recv_base + gen + my_pos * stripe_bytes),
                    nby);
            }
            // readiness AFTER payload, same peer + same NoC (ordered), so no owner sees a partial early
            for (uint32_t j = 0; j < S; ++j) {
                const uint32_t op = owner_pos(j);
                if (op != my_pos) {
                    noc_semaphore_inc(get_noc_addr(gcx(op), gcy(op), arr_addr[nb & 1u]), 1);
                }
            }
            noc_async_writes_flushed();  // payload departed cb_send -> compute may refill it
        }
        RA_DBG_P("E_sent nb", nb);
        cb_pop_front(cb_send, rs_T);

        if (is_owner) {
            volatile tt_l1_ptr uint32_t* gen_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_addr[nb & 1u]);
            {
                RA_ZONE("Z_RSS_ARRIVE");
                noc_semaphore_wait_min(gen_ptr, P - 1u);  // every peer's partial for THIS generation has landed
            }
            RA_DBG_P("F_arrived nb", nb);
            cb_push_back(cb_recv, P * stripe_tiles);

            {
                // Blocks until compute has finished reducing this stripe, so this zone IS the reduction cost as
                // seen from the data-movement core (compute-side zones are unavailable on TRISC).
                RA_ZONE("Z_RSS_REDUCEWAIT");
                cb_wait_front(out_cb, stripe_tiles);
            }
            RA_DBG_P("H_reduced nb", nb);
            {
                RA_ZONE("Z_RSS_OUTWRITE");
                const uint32_t rr = get_read_ptr(out_cb);
                for (uint32_t idx = own_beg; idx < own_end; ++idx) {
                    const uint32_t m = idx / N_block;
                    const uint32_t n = idx - m * N_block;
                    if (m < valid_m && (n_base + n) < valid_n) {
                        noc_async_write_page(
                            (m_start + m) * Nt + (n_start + n_base + n), out, rr + (idx - own_beg) * tile_bytes);
                    }
                }
                noc_async_writes_flushed();
                cb_pop_front(out_cb, stripe_tiles);
            }
            {
                RA_ZONE("Z_RSS_CREDITSEND");
                // Clear this generation's arrival counter BEFORE crediting. A sender may run two sub-blocks
                // ahead and reuse this same counter, but only after receiving the credit below, so the reset can
                // never swallow its increment.
                noc_semaphore_set(gen_ptr, 0);
                const uint32_t my_cred = get_semaphore(cred_sem(my_stripe));
                for (uint32_t p = 0; p < P; ++p) {
                    if (p != my_pos) {
                        noc_semaphore_inc(get_noc_addr(gcx(p), gcy(p), my_cred), 1);
                    }
                }
            }
        }
    }
#elif defined(RSCATTER) && defined(RS_DIRECT)
    // ---- Pk > 1: DIRECT-EXCHANGE REDUCE-SCATTER (all-to-all within the group of Pk k-slice cores). ----
    // The ring variant below needs Pk-1 SEQUENTIAL rounds, each paying a semaphore round-trip. Here every core
    // issues all Pk partial-writes back to back and then waits ONCE for all Pk arrivals, so the Pk-1 sequential
    // sync steps collapse to 1. Total bytes and total adds are identical to the ring; only serialization goes.
    //
    // Layout: compute leaves a bf16 image of this core's WHOLE sub-block partial in cb_send. Slice q of that
    // image (chunk q) belongs to the core at position q, so we write it into that core's cb_recv slot indexed by
    // OUR position -- distinct slot per source, so a single arrival counter is unambiguous (no fungibility
    // problem). The loop includes ourselves, so the loopback write fills our own slot and the reduce sees Pk
    // uniform partials with no special case.
    //
    // Flow control: each core credits every group member once after consuming a sub-block, so a sender waits for
    // nb*Pk credits before writing sub-block nb. Each member sends exactly one credit per sub-block and there
    // are Pk of them, so nb*Pk total credits implies every member has finished sub-block nb-1 (max per member is
    // nb, total is nb*Pk over Pk members => each is exactly nb).
    const uint32_t P = rs_P;
    constexpr uint32_t cb_send = 4;
    constexpr uint32_t cb_recv = 5;
    const uint32_t rs_base_tiles = rs_T / P;
    const uint32_t rs_rem = rs_T - rs_base_tiles * P;
    const uint32_t max_chunk = rs_base_tiles + (rs_rem ? 1u : 0u);
    const uint32_t max_chunk_bytes = max_chunk * tile_bytes;
    auto csize = [=](uint32_t c) { return rs_base_tiles + (c < rs_rem ? 1u : 0u); };
    auto coff = [=](uint32_t c) { return c * rs_base_tiles + (c < rs_rem ? c : rs_rem); };
    const uint32_t my_pos = rs_owned_chunk;             // direct mode: position == owned chunk index
    const uint32_t recv_base = get_write_ptr(cb_recv);  // identical offset on every core (same CB config)
    const uint32_t arrive_addr = get_semaphore(red_sem_id);
    volatile tt_l1_ptr uint32_t* arrive_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_addr);
    const uint32_t credit_addr = get_semaphore(redfree_sem_id);
    volatile tt_l1_ptr uint32_t* credit_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_addr);
    const uint32_t own_tiles = csize(my_pos);

    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
        const uint32_t n_base = nb * N_block;
        // Wait until every group member has drained its receive buffer for the previous sub-block.
        if (nb) {
            noc_semaphore_wait_min(credit_ptr, nb * P);
        }
        cb_reserve_back(cb_recv, P * max_chunk);  // claim the receive slots BEFORE any peer writes into them
        cb_wait_front(cb_send, rs_T);             // compute staged the bf16 image of our whole partial
        const uint32_t img = get_read_ptr(cb_send);
        // Scatter: slice q -> the core at position q, into OUR slot there.
        for (uint32_t q = 0; q < P; ++q) {
            const uint32_t px = get_arg_val<uint32_t>(24 + q * 2), py = get_arg_val<uint32_t>(25 + q * 2);
            const uint32_t nbytes = csize(q) * tile_bytes;
            noc_async_write(
                img + coff(q) * tile_bytes, get_noc_addr(px, py, recv_base + my_pos * max_chunk_bytes), nbytes);
        }
        // Payload THEN readiness to the SAME peer on the SAME NoC (ordered, as in the in0 ring), so no core can
        // observe an arrival before its data has landed.
        for (uint32_t q = 0; q < P; ++q) {
            const uint32_t px = get_arg_val<uint32_t>(24 + q * 2), py = get_arg_val<uint32_t>(25 + q * 2);
            noc_semaphore_inc(get_noc_addr(px, py, arrive_addr), 1);
        }
        noc_async_writes_flushed();  // payloads departed cb_send -> the slot can be refilled by compute
        cb_pop_front(cb_send, rs_T);
        // ONE wait for the whole exchange (this is the point of the scheme).
        noc_semaphore_wait_min(arrive_ptr, (nb + 1) * P);
        cb_push_back(cb_recv, P * max_chunk);  // compute reduces all P slots -> out_cb

        // Our chunk is now fully reduced: write its tiles to DRAM.
        cb_wait_front(out_cb, max_chunk);
        const uint32_t rr = get_read_ptr(out_cb);
        const uint32_t own_off = coff(my_pos);
        for (uint32_t j = 0; j < own_tiles; ++j) {
            const uint32_t idx = own_off + j;
            const uint32_t m = idx / N_block;
            const uint32_t n = idx - m * N_block;
            if (m < valid_m && (n_base + n) < valid_n) {
                noc_async_write_page((m_start + m) * Nt + (n_start + n_base + n), out, rr + j * tile_bytes);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(out_cb, max_chunk);
        // Receive buffer is free again: credit every group member (including ourselves, keeping the count
        // uniform at P per sub-block).
        for (uint32_t q = 0; q < P; ++q) {
            const uint32_t px = get_arg_val<uint32_t>(24 + q * 2), py = get_arg_val<uint32_t>(25 + q * 2);
            noc_semaphore_inc(get_noc_addr(px, py, credit_addr), 1);
        }
    }
#elif defined(RSCATTER)
    // ---- Pk > 1: RING REDUCE-SCATTER (one independent reduce-scatter per output SUB-block). ----
    // P = Pk cores in the factory's optimized cyclic order. Each of the N_bpc output sub-blocks (M_block x
    // N_block tiles = rs_T tiles) is tile-partitioned into P contiguous chunks (row-major). The partition does
    // NOT need to be even: chunk sizes differ by at most one tile (the first rs_T%P chunks take one extra), so
    // any rs_T >= P works. Every round each core sends one chunk to `next` and receives one from `prev` into
    // cb_recv; compute adds its own resident partial tiles and forwards the running sum, so after P-1 rounds
    // each core holds ONE fully-reduced chunk (rs_owned_chunk) and writes its tiles.
    // Semaphore EPOCHS increase monotonically across (nb, round): global g = nb*(P-1)+t, so a wait targets g+1
    // and can never alias across sub-blocks. Reuses the in0-ring payload->credit protocol (red_sem/redfree_sem).
    // cb_send/cb_recv are EXACTLY 2 slots of the MAXIMUM chunk size and every CB operation moves a full
    // max-size slot (only the useful prefix is ever written or read), so the FIFO period stays 2 and the remote
    // write offset is a constant stride even when the chunks are uneven.
    const uint32_t P = rs_P;
    constexpr uint32_t cb_send = 4;
    constexpr uint32_t cb_recv = 5;
    const uint32_t rs_base_tiles = rs_T / P;  // floor size; the first rs_rem chunks carry one more
    const uint32_t rs_rem = rs_T - rs_base_tiles * P;
    const uint32_t max_chunk = rs_base_tiles + (rs_rem ? 1u : 0u);
    const uint32_t max_chunk_bytes = max_chunk * tile_bytes;
    // size/offset of chunk c within the sub-block
    auto csize = [=](uint32_t c) { return rs_base_tiles + (c < rs_rem ? 1u : 0u); };
    auto coff = [=](uint32_t c) { return c * rs_base_tiles + (c < rs_rem ? c : rs_rem); };
    // My cycle position, derived from the chunk I own (the factory sets rs_own_chunk = (rs_pos+1) % P).
    const uint32_t rs_pos_local = (rs_owned_chunk + P - 1u) % P;
    const uint32_t recv_base = get_write_ptr(cb_recv);  // my cb_recv L1 base (identical offset on every core)
    const uint32_t rs_recv_addr = get_semaphore(red_sem_id);
    volatile tt_l1_ptr uint32_t* rs_recv_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_recv_addr);
    const uint32_t rs_free_addr = get_semaphore(redfree_sem_id);
    volatile tt_l1_ptr uint32_t* rs_free_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs_free_addr);
    const uint64_t prev_rs_free = get_noc_addr(rs_prev_x, rs_prev_y, rs_free_addr);  // I credit prev
    const uint64_t next_rs_recv = get_noc_addr(rs_next_x, rs_next_y, rs_recv_addr);  // I signal next

    uint32_t g = 0;  // monotonically increasing (nb, round) epoch, shared by rs_recv and rs_free
    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
        const uint32_t n_base = nb * N_block;  // this sub-block's N-column base within the core's ownership
        for (uint32_t t = 0; t + 1u < P; ++t, ++g) {
            // Post my recv-slot credit to prev FIRST (every core credits before any core blocks on its own send
            // credit -> no cyclic deadlock). cb_reserve_back waits until compute consumed my previous chunk.
            cb_reserve_back(cb_recv, max_chunk);
            noc_semaphore_inc(prev_rs_free, 1);          // tell prev: my cb_recv slot is free for epoch g
            cb_wait_front(cb_send, max_chunk);           // compute staged this epoch's send chunk
            noc_semaphore_wait_min(rs_free_ptr, g + 1);  // next freed its slot for epoch g
            const uint32_t slot = g & 1u;                // double-buffered (2 slots), period matches the global epoch
            // The chunk I send at round t is the one I reduced at round t-1 (at t==0, my own seed chunk).
            // `next` derives the SAME size for its round-t receive, since its round-t chunk is exactly this one.
            const uint32_t send_bytes = csize((rs_pos_local + P - t) % P) * tile_bytes;
            uint64_t dst = get_noc_addr(rs_next_x, rs_next_y, recv_base + slot * max_chunk_bytes);
            noc_async_write(get_read_ptr(cb_send), dst, send_bytes);
            noc_semaphore_inc(next_rs_recv, 1);  // ordered after the payload (same peer + NoC, like the in0 ring)
            noc_async_writes_flushed();          // payload departed L1 -> cb_send slot reusable
            cb_pop_front(cb_send, max_chunk);
            noc_semaphore_wait_min(rs_recv_ptr, g + 1);  // prev delivered epoch g into my cb_recv slot g%2
            cb_push_back(cb_recv, max_chunk);            // compute adds its own tiles + (forwards | finishes owned)
        }
        // The last round produced my fully-reduced OWNED chunk into out_cb -> write its tiles to DRAM.
        // Chunk c owns sub-block tiles [coff(c), coff(c)+csize(c)); tile i -> (m = i/N_block, n = i%N_block) at
        // global (m_start+m, n_start + n_base + n); valid iff m < valid_m and (n_base+n) < valid_n.
        const uint32_t own_tiles = csize(rs_owned_chunk);
        const uint32_t own_off = coff(rs_owned_chunk);
        cb_wait_front(out_cb, max_chunk);
        const uint32_t rr = get_read_ptr(out_cb);
        for (uint32_t j = 0; j < own_tiles; ++j) {
            const uint32_t idx = own_off + j;
            const uint32_t m = idx / N_block;
            const uint32_t n = idx - m * N_block;
            if (m < valid_m && (n_base + n) < valid_n) {
                noc_async_write_page((m_start + m) * Nt + (n_start + n_base + n), out, rr + j * tile_bytes);
            }
        }
        noc_async_writes_flushed();  // output pages departed L1 -> out_cb slot safe to reuse
        cb_pop_front(out_cb, max_chunk);
    }
#else
    // Pk > 1: linear reduction chain. `use_reduce` carries the cb_reduce DEPTH (>=2); guard the modulus so the
    // Pk==1 compile (use_reduce == 0) does not instantiate a division by zero in this unreachable branch.
    constexpr uint32_t red_depth = use_reduce ? use_reduce : 1u;
    // cb_reduce holds 2 blocks (double-buffered). reduce_base captured ONCE BEFORE any cb_reduce use (the
    // write ptr drifts after receives).
    const uint32_t reduce_base = get_write_ptr(cb_reduce);
    const uint32_t red_addr = get_semaphore(red_sem_id);
#if defined(REDUCE_MEET)
    // ONE shared receive counter is sufficient: a root waits for the TOTAL number of arrivals for this
    // sub-block (nrecv per nb), not for a specific one, and the two senders write to DIFFERENT slots decided
    // by their ordinal. So there is no fungibility problem and no second semaphore.
    const uint64_t prev2_redfree = get_noc_addr(red_prev2_x, red_prev2_y, get_semaphore(redfree_sem_id));
#endif
    volatile tt_l1_ptr uint32_t* red_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_addr);
    const uint32_t redfree_addr = get_semaphore(redfree_sem_id);
    volatile tt_l1_ptr uint32_t* redfree_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(redfree_addr);
    const uint64_t prev_redfree = get_noc_addr(red_prev_x, red_prev_y, redfree_addr);
    const uint64_t next_recv = get_noc_addr(red_next_x, red_next_y, red_addr);

    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
#if defined(SKIP_REDUCTION)
        // Diagnostic: remove the split-K chain (non-root sends, root receives/accumulation). compute copied
        // this band's LOCAL partial into out_cb; every band writes it to DRAM directly (unless output skipped).
        (void)reduce_base;
        (void)red_ptr;
        (void)redfree_ptr;
        (void)prev_redfree;
        (void)next_recv;
        cb_wait_front(out_cb, out_blk);
        uint32_t r = get_read_ptr(out_cb);
        const uint32_t n_off = n_start + nb * N_block;
        for (uint32_t m = 0; m < M_block; ++m) {
            for (uint32_t n = 0; n < N_block; ++n) {
                if (m < valid_m && (nb * N_block + n) < valid_n) {
#if !defined(SKIP_OUTPUT_WRITE)
                    noc_async_write_page((m_start + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
#endif
                }
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(out_cb, out_blk);
#else
#if defined(REDUCE_MEET)
        // Return a credit to each predecessor, then wait for ALL of this sub-block's partials to land, then
        // push them in ordinal order. Pushing only after every arrival is what lets one counter serve both.
        if (red_nrecv > 0u) {
            cb_reserve_back(cb_reduce, red_nrecv * out_blk);
            noc_semaphore_inc(prev_redfree, 1);
            if (red_nrecv > 1u) {
                noc_semaphore_inc(prev2_redfree, 1);
            }
            noc_semaphore_wait_min(red_ptr, (nb + 1) * red_nrecv);
            cb_push_back(cb_reduce, red_nrecv * out_blk);
        }
#else
        if (!is_bottom) {
            cb_reserve_back(cb_reduce, out_blk);  // wait our compute freed slot (nb-2)
            noc_semaphore_inc(prev_redfree, 1);   // tell prev: our slot (nb%2) is free for block nb
            noc_semaphore_wait_min(red_ptr, nb + 1);  // prev forwarded block nb into it (chain latency)
            cb_push_back(cb_reduce, out_blk);  // compute reduce_add's it -> out_cb, pops cb_reduce
        }
#endif
#if defined(FUSE_BIAS) || defined(FUSE_TERNARY)
        if (is_top) {
            feed_fused(nb);  // ROOT only: supply bias/residual/gate for compute's single fused epilogue
        }
#endif
        cb_wait_front(out_cb, out_blk);  // compute produced reduced (+ fused at top) block nb
        uint32_t r = get_read_ptr(out_cb);
        if (!is_top) {
            noc_semaphore_wait_min(redfree_ptr, nb + 1);  // next signalled its slot is free
#if defined(REDUCE_MEET)
            // My destination pushes red_dest_nrecv blocks per sub-block in ordinal order, so the FIFO slot for
            // (nb, my ordinal) is exactly this. Both senders to a root therefore target distinct slots, and a
            // single-input receiver gets the plain nb-indexed slot.
            const uint32_t red_slot = (nb * red_dest_nrecv + red_send_ord) % red_cb_slots;
            uint64_t dst = get_noc_addr(red_next_x, red_next_y, reduce_base + red_slot * out_blk_bytes);
#else
            uint64_t dst = get_noc_addr(red_next_x, red_next_y, reduce_base + (nb % red_depth) * out_blk_bytes);
#endif
            noc_async_write(r, dst, out_blk_bytes);
            // Pipelined: payload THEN signal to the SAME peer on the SAME NoC (ordered, like the in0 ring) so
            // the receiver never observes readiness before its partial-sum has landed. Flush (not a full
            // barrier) so the out_cb source slot is reusable; completion is deferred to the final barrier.
#if defined(REDUCE_MEET)
            noc_semaphore_inc(get_noc_addr(red_next_x, red_next_y, red_addr), 1);
#else
            noc_semaphore_inc(next_recv, 1);  // block nb delivered (ordered after the payload write)
#endif
            noc_async_writes_flushed();       // payload departed L1 -> out_cb slot safe to reuse
        } else {
            // ROOT: issue output DRAM pages + flush (the reduction tail on the wall).
            const uint32_t n_off = n_start + nb * N_block;  // global N tile of this subblock
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t n = 0; n < N_block; ++n) {
                    if (m < valid_m && (nb * N_block + n) < valid_n) {  // write only valid_m x valid_n
#if defined(SKIP_OUTPUT_WRITE)
                        // diagnostic: drop the DRAM payload write; keep iteration + CB consumption.
#elif defined(OUT_CHUNKS)
                        write_out_tile(m_start + m, n_off + n, r + (m * N_block + n) * tile_bytes);
#else
                        noc_async_write_page((m_start + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
#endif
                    }
                }
            }
            noc_async_writes_flushed();  // output pages departed L1 -> out_cb slot safe to reuse
        }
        cb_pop_front(out_cb, out_blk);
#endif  // SKIP_REDUCTION
    }
#endif  // RSCATTER vs chain
    // Pipelined: single deferred completion before return — drain this core's forwarded partial-sums / DRAM
    // output writes AND the non-posted reduction-readiness semaphore atomics (noc_semaphore_inc), so no
    // in-flight NoC transaction outlives the program (writes_flushed above only guarantees source-L1 reuse).
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
