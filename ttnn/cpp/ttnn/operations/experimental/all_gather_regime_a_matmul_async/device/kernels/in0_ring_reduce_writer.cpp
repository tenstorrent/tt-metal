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
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

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
    // ---- Fused fabric all-gather (Phase 1). 0 => the whole prologue compiles out (tp == 1). ----
    constexpr uint32_t fused_gather_enabled = get_compile_time_arg_val(14);
    constexpr uint32_t fused_rt_base = get_compile_time_arg_val(15);  // first runtime arg of the fused block
    // Accessor args start after the 16 scalar compile-time args above (0..15). This index is COUPLED to
    // the host's wct push order -- adding a scalar CT arg without bumping it silently misparses every
    // accessor and corrupts the output rather than failing to build.
    constexpr auto in0_args = TensorAccessorArgs<16>();
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

    // Blocked-cyclic global-K mapping, filled in by the fused prologue below (0 => contiguous, the
    // single-chip layout). Declared out here because the in0 ring load further down needs them.
    uint32_t k_run_len = 0u, k_stripe_base = 0u, k_shard_stride = 0u;
    // Capacity-local slots PER SOURCE RANK. Equals k_run_len on the DRAM-staged path (rank stripes are packed
    // back to back); larger under direct-L1, which pads each rank up to a whole number of ring slots so that
    // no slot straddles two ranks. Only the first k_run_len slots of each rank hold real data.
    uint32_t k_rank_span = 1u;
    // Progressive consumption: filled in by the fused prologue, read by the in0 ring below so it can gate
    // each shard on that source rank having actually landed rather than on the whole gather being done.
    uint32_t wave_fwd_sem_id = 0u, wave_bwd_sem_id = 0u, ready_sem_id_c = 0u;
    uint32_t fwd_recv_total = 0u, bwd_recv_total = 0u, my_rank = 0u, my_tp = 1u;

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    constexpr uint32_t in0_cb = 0, out_cb = 2, cb_reduce = 7;
    constexpr uint32_t in0_blk = M_block * K_block;
    constexpr uint32_t in0_blk_bytes = in0_blk * tile_bytes;
    constexpr uint32_t shard_bytes = W * in0_blk_bytes;
    constexpr uint32_t out_blk = M_block * N_block;

    // Fused-gather transfer batch: how many tiles are kept in flight between synchronization points.
    // Bounded by the per-RISC packet-header pool (12 on Blackhole; we need batch + 1 for the credit) and
    // by cb0's capacity, whose first tiles double as the scratch ring during the prologue.
    // Must match kGatherScratchTiles on the host: the scratch is its own CB (c_11), NOT cb0's head.
    // Sharing cb0 aliased the on-chip ring's slots; see the comment at the c_11 allocation.
    constexpr uint32_t kGatherBatch = 8;
    constexpr uint32_t gather_scratch_cb = 11;

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
    // ONE running index over the optional runtime args, consumed in the exact order the factory pushes them:
    // bias, ternary, chunk info, then reduce-scatter. A single counter is what lets fusion / chunked output /
    // reduce-scatter be active simultaneously (they used to be mutually exclusive at index 17).
    // ---- On-chip ring SLOT SCHEDULE (args 17..32; RingSlotArg on the host) ----
    // A cb0 slot index is a CONSUMPTION-ORDER index: compute waits cumulatively and addresses each block by an
    // explicit ascending offset, so "consumed s-th" and "in slot s" are the same statement. The host therefore
    // controls consumption order by choosing which stripe lands in which slot, and the kernel just follows.
    const uint32_t ring_own_slot = get_arg_val<uint32_t>(17);  // where MY chunk goes
    // Where MY chunk goes on each neighbour device. Not the same as ring_own_slot once consumption is ordered
    // by availability: hop counts differ per device, so each neighbour sorts its chunks differently.
    const uint32_t ring_peer_slot_fwd = get_arg_val<uint32_t>(18);
    const uint32_t ring_peer_slot_bwd = get_arg_val<uint32_t>(19);
    constexpr uint32_t kRingFwdBase = 20;             // G pairs {src_slot, dst_slot}, one per consume step
    constexpr uint32_t kRingNoForward = 0xFFFFFFFFu;  // src_slot sentinel: this step forwards nothing
    [[maybe_unused]] uint32_t fidx = 20u + 2u * G;    // optional args start after the schedule
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

    // Feed the fused-epilogue operands for the SLICE [beg, beg+nt) of output sub-block `nb`, in SLICE ORDER, so
    // compute indexes every operand 0..nt-1 with no modular arithmetic. Used by reduce-scatter, where each owner
    // applies the epilogue exactly once to its own fully reduced slice. Slice tile j is sub-block tile beg+j,
    // i.e. (m, n) = ((beg+j)/N_block, (beg+j)%N_block); bias/broadcast-gate depend only on n, residual and a
    // full gate on (m, n). Invalid tail positions are zero-filled and never DRAM-read, exactly as feed_fused.
    [[maybe_unused]] auto feed_fused_slice = [&](uint32_t nb, uint32_t beg, uint32_t nt) {
        [[maybe_unused]] const uint32_t n_off = n_start + nb * N_block;
#if defined(FUSE_BIAS)
        cb_reserve_back(bias_cb, nt);
        uint32_t pb = get_write_ptr(bias_cb);
        for (uint32_t j = 0; j < nt; ++j) {
            const uint32_t idx = beg + j;
            const uint32_t n = idx - (idx / N_block) * N_block;
            if ((nb * N_block + n) < valid_n) {
                noc_async_read_page(n_off + n, bias, pb);
            } else {
                zero_tile(pb);
            }
            pb += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(bias_cb, nt);
#endif
#if defined(FUSE_TERNARY)
        cb_reserve_back(ta_cb, nt);
        cb_reserve_back(tb_cb, nt);
        uint32_t pa = get_write_ptr(ta_cb);
        uint32_t pg = get_write_ptr(tb_cb);
        for (uint32_t j = 0; j < nt; ++j) {
            const uint32_t idx = beg + j;
            const uint32_t m = idx / N_block;
            const uint32_t n = idx - m * N_block;
            const bool ok = (m < valid_m) && ((nb * N_block + n) < valid_n);
            if (ok) {
                noc_async_read_page((m_start + m) * Nt + (n_off + n), ta, pa);
            } else {
                zero_tile(pa);
            }
            // gate: [1,N] broadcast reads by column only; [M,N] reads the same (m,n) as the residual
            if (ok) {
                noc_async_read_page(bcast_gate ? (n_off + n) : ((m_start + m) * Nt + (n_off + n)), tb, pg);
            } else {
                zero_bytes(pg, gate_tile_bytes);
            }
            pa += tile_bytes;
            pg += gate_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(ta_cb, nt);
        cb_push_back(tb_cb, nt);
#endif
    };

    // Chunked output support (all_gather_regime_a_matmul_async_split): route each output tile to the chunk buffer that
    // owns its global N column. chunk = global_n / out_ntc, col = global_n % out_ntc; write page (m)*out_ntc+col into
    // chunk-buffer `chunk`. All chunk buffers share the output TensorAccessorArgs (same [M, N/chunks] spec), differing
    // only by base address. Not compiled unless OUT_CHUNKS (chunks>1); chunks==1 uses the original single-buffer write
    // below (byte-identical).
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

    // ---- RING REDUCE-SCATTER args, read AFTER any fusion/chunk args via the shared running index. ----
#if defined(RSCATTER)
    const uint32_t rs_next_x = get_arg_val<uint32_t>(fidx++);  // next core in the Pk cycle (I send to it)
    const uint32_t rs_next_y = get_arg_val<uint32_t>(fidx++);
    const uint32_t rs_prev_x = get_arg_val<uint32_t>(fidx++);  // prev core (it sends to me)
    const uint32_t rs_prev_y = get_arg_val<uint32_t>(fidx++);
    const uint32_t rs_owned_chunk = get_arg_val<uint32_t>(fidx++);  // slice this core owns + writes to DRAM
    const uint32_t rs_P = get_arg_val<uint32_t>(fidx++);            // cycle size = Pk
    const uint32_t rs_T = get_arg_val<uint32_t>(fidx++);            // tiles per sub-block = M_block*N_block
#endif

    // ---- cb0 reservation, hoisted ABOVE the gather ----
    // cb0 holds the worker's COMPLETE gathered k-slice and is written exactly once, so reserving all of it up
    // front is free (the CB is empty at kernel entry, so this never blocks) and it is what gives the gather a
    // stable slot-0 address to write into. Direct-L1 needs that address before the gather runs, since a peer
    // writes our slot 0 straight into L1; the staged path is unaffected by the earlier reservation.
    cb_reserve_back(in0_cb, K_num_blocks * in0_blk);
    const uint32_t base0 = get_write_ptr(in0_cb);

#if defined(DIRECT_L1)
    // LEAF DEFERRAL. A core that receives its chunk over the fabric but relays nothing onward (hop distance
    // tp-1) has no downstream dependant, so its wait for its own chunk can move OUT of the prologue and into
    // the ring step where that chunk is actually consumed. Those are exactly the cores whose chunk lands in
    // the LAST fabric wave -- the ones that set the makespan. Appendix B's worked example: core 2 idles
    // through the entire gather and then owes all 8 of its steps, while two of its chunks have been sitting
    // in its L1 since t~1.
    //
    // RELAY cores deliberately keep the eager prologue wait. Deferring theirs would push the next device's
    // arrival out by however long this core computes first, which is strictly worse than the stall it saves.
    uint32_t dl1_own_pending = 0;  // 1 => we still owe our own chunk (wait if remote, then relay it onward)
    uint32_t dl1_recv_sem = 0;     // arrival semaphore address
    // The fabric relay moved out of the prologue and into the ring loop, so the values it needs come with it.
    // dl1_mux_fa is where this core's mux client block starts in the runtime args.
    std::size_t dl1_mux_fa = 0;
    uint32_t dl1_h_dist = 0, dl1_h_send_fwd = 0, dl1_h_send_bwd = 0, dl1_h_packet_bytes = 0;
#endif

    // ---- PHASE 0: fused fabric all-gather (Phase 1 of the design spec; DRAM-staged) ----
    //
    // Runs BEFORE the on-chip ring below. Two different rings, easy to confuse:
    //   * the fabric gather here spans the TP group ACROSS DEVICES and fills the staging buffer;
    //   * the loop below is the existing 8-core ON-CHIP rotation, and is untouched.
    //
    // Once staging holds the full [M, K_global] activation, the on-chip ring reads it with global-K stride
    // exactly as it reads a single-chip in0 -- the host has already pointed the in0 accessor and in0_addr
    // at the staging buffer, so nothing below needs to change.
    //
    // Order of business:
    //   1. copy this rank's own shard into staging at K offset rank * k_shard_tiles (local, no fabric);
    //   2. master-ring cores push that shard to the neighbours' staging over mux v2, forward/backward,
    //      re-injecting received shards until every rank has all tp of them (store-and-forward);
    //   3. every core blocks on the readiness semaphore until all tp shards have landed.
    // Payload must precede readiness: flush the write, THEN atomic-inc, or a peer can consume a
    // half-written shard.
#if defined(FUSED_GATHER)
    {
        std::size_t fa = fused_rt_base;
        const uint32_t is_fabric_client = get_arg_val<uint32_t>(fa++);
        const uint32_t rank = get_arg_val<uint32_t>(fa++);
        my_rank = rank;
        const uint32_t tp = get_arg_val<uint32_t>(fa++);
        my_tp = tp;
        const uint32_t k_shard_tiles = get_arg_val<uint32_t>(fa++);
        const uint32_t stage_addr = get_arg_val<uint32_t>(fa++);
        const uint32_t ready_sem_id = get_arg_val<uint32_t>(fa++);
        ready_sem_id_c = ready_sem_id;
        const uint32_t has_fwd = get_arg_val<uint32_t>(fa++);
        const uint32_t has_bwd = get_arg_val<uint32_t>(fa++);
        const uint32_t Mt_total = get_arg_val<uint32_t>(fa++);
        const uint32_t Kt_global = get_arg_val<uint32_t>(fa++);
        const uint32_t shard_addr = get_arg_val<uint32_t>(fa++);
        const uint32_t bank_id = get_arg_val<uint32_t>(fa++);
        // GLOBAL semaphore ADDRESS (not a program semaphore id): a peer chip's atomic-inc can land before
        // this program launches, so the credit has to live in memory the caller allocated up front.
        const uint32_t my_recv_sem_addr = get_arg_val<uint32_t>(fa++);
        const uint32_t my_dir = get_arg_val<uint32_t>(fa++);  // 0 = forward, 1 = backward
        const uint32_t my_send_rounds = get_arg_val<uint32_t>(fa++);
        const uint32_t my_recv_rounds = get_arg_val<uint32_t>(fa++);
        const uint32_t m_groups = get_arg_val<uint32_t>(fa++);
        // On-chip barrier args (see the FusedGatherArg enum on the host).
        const uint32_t is_master0 = get_arg_val<uint32_t>(fa++);
        const uint32_t gather_count_sem_id = get_arg_val<uint32_t>(fa++);
        const uint32_t num_masters = get_arg_val<uint32_t>(fa++);
        const uint32_t master0_x = get_arg_val<uint32_t>(fa++);
        const uint32_t master0_y = get_arg_val<uint32_t>(fa++);
        const uint32_t dir_count_sem_id = get_arg_val<uint32_t>(fa++);
        wave_fwd_sem_id = get_arg_val<uint32_t>(fa++);
        wave_bwd_sem_id = get_arg_val<uint32_t>(fa++);
        const uint32_t fwd_coord_x = get_arg_val<uint32_t>(fa++);
        const uint32_t fwd_coord_y = get_arg_val<uint32_t>(fa++);
        const uint32_t bwd_coord_x = get_arg_val<uint32_t>(fa++);
        const uint32_t bwd_coord_y = get_arg_val<uint32_t>(fa++);
        fwd_recv_total = get_arg_val<uint32_t>(fa++);
        bwd_recv_total = get_arg_val<uint32_t>(fa++);
        const uint32_t local_done_sem_id = get_arg_val<uint32_t>(fa++);
        k_run_len = get_arg_val<uint32_t>(fa++);
        k_stripe_base = get_arg_val<uint32_t>(fa++);
        k_shard_stride = get_arg_val<uint32_t>(fa++);
        const uint32_t fwd_coord_swaps = get_arg_val<uint32_t>(fa++);
        const uint32_t bwd_coord_swaps = get_arg_val<uint32_t>(fa++);
        k_rank_span = get_arg_val<uint32_t>(fa++);
        // ---- PHASE 2 direct-L1 stream plan for THIS core (all zero on the staged path) ----
        const uint32_t dl1_active = get_arg_val<uint32_t>(fa++);
        const uint32_t dl1_dist = get_arg_val<uint32_t>(fa++);
        const uint32_t dl1_send_fwd = get_arg_val<uint32_t>(fa++);
        const uint32_t dl1_send_bwd = get_arg_val<uint32_t>(fa++);
        const uint32_t dl1_recv_sem_addr = get_arg_val<uint32_t>(fa++);
        const uint32_t dl1_packet_bytes = get_arg_val<uint32_t>(fa++);
        const uint32_t dl1_defer = get_arg_val<uint32_t>(fa++);
        const uint32_t num_release_ranges = get_arg_val<uint32_t>(fa++);
        const std::size_t release_base = fa;  // 6 words per range: sx, sy, ex, ey, dests_fwd, dests_bwd
        fa += 6u * num_release_ranges;
        const std::size_t master_base = fa;  // num_masters (x, y) virtual coord pairs
        fa += 2u * num_masters;
        // has_fwd / has_bwd are mutually exclusive: a core is a client of exactly the one mux it drives
        // (or neither, at a line end). Which one it is comes from my_dir; the mux client block that
        // follows is whichever the host appended.

        // LOCAL-SHARD accessor: the factory appends it AFTER every other accessor, and ONLY when tp > 1.
        // It must therefore be declared in here -- declaring it unconditionally reads past the end of the
        // compile-time args on the tp == 1 build and fails JIT template deduction.
#if defined(FUSE_TERNARY)
        constexpr auto shard_args = TensorAccessorArgs<tb_args.next_compile_time_args_offset()>();
#elif defined(FUSE_BIAS)
        constexpr auto shard_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
#else
        constexpr auto shard_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
#endif
        // in0_args is ALREADY the staging accessor on the fused path (the host bound it there), so reuse
        // it rather than declaring a second view of the same buffer.
        const auto stage_acc = TensorAccessor(in0_args, stage_addr, tile_bytes);
        const auto shard_acc = TensorAccessor(shard_args, shard_addr, tile_bytes);

        // Multicast an already-set semaphore word to every core running this kernel. NON-loopback, so the
        // caller sets its own copy first; each record's num_dests already excludes the sender. The
        // rectangles come from the host because the bank-adjacent placement is not one filled box.
        auto publish_to_grid = [&](uint32_t sem_addr, bool as_bwd_coord) {
            for (uint32_t r = 0; r < num_release_ranges; ++r) {
                const std::size_t b = release_base + 6u * r;
                // Records hold NOC_0 corner order; swap for a coordinator whose writer runs on NOC_1,
                // which traverses the grid the other way. Decided per-sender: the two coordinators are
                // different cores and need not share a NOC.
                const bool swap_corners = as_bwd_coord ? (bwd_coord_swaps != 0u) : (fwd_coord_swaps != 0u);
                const uint32_t c0 = get_arg_val<uint32_t>(b + (swap_corners ? 2u : 0u));
                const uint32_t c1 = get_arg_val<uint32_t>(b + (swap_corners ? 3u : 1u));
                const uint32_t c2 = get_arg_val<uint32_t>(b + (swap_corners ? 0u : 2u));
                const uint32_t c3 = get_arg_val<uint32_t>(b + (swap_corners ? 1u : 3u));
                const uint64_t box = get_noc_multicast_addr(c0, c1, c2, c3, sem_addr);
                // num_dests must exclude ME specifically -- the host precomputed a count for each of the
                // two possible senders, because a non-loopback multicast that counts the sender waits on
                // an ack that never comes.
                const uint32_t dests = get_arg_val<uint32_t>(b + (as_bwd_coord ? 5u : 4u));
                if (dests != 0u) {
                    noc_semaphore_set_multicast(sem_addr, box, dests);
                }
            }
            // The multicast SOURCES from that word; on Blackhole NoC latency exceeds RISC->L1 latency, so
            // without this flush the RISC can overwrite it before the NoC has read it out.
            noc_async_writes_flushed();
        };

        // Report that MY M-slice of arrival `i` has landed, and -- if I am this direction's coordinator --
        // publish arrival `i` to the whole grid once every master of this direction has reported it.
        //
        // A single master's credit only covers its own M slice, so one master's arrival is not the shard.
        // Routing through a per-direction coordinator means the published value has a single writer and is
        // monotone, which is what lets consumers wait_min on it. Arrivals within a direction are strictly
        // ordered (one upstream neighbour), so a plain count is unambiguous here.
        const bool is_dir_coord = (bank_id == (my_dir == 0u ? 0u : m_groups));
        // Which reporter slot I own within my direction: banks 0..m_groups-1 forward, the rest backward.
        const uint32_t my_reporter_slot = (my_dir == 0u) ? bank_id : (bank_id - m_groups);
        const uint32_t coord_x = (my_dir == 0u) ? fwd_coord_x : bwd_coord_x;
        const uint32_t coord_y = (my_dir == 0u) ? fwd_coord_y : bwd_coord_y;
        const uint32_t my_wave_sem_id = (my_dir == 0u) ? wave_fwd_sem_id : wave_bwd_sem_id;
        auto report_arrival = [&](uint32_t i) {
            volatile tt_l1_ptr uint32_t* my_recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_recv_sem_addr);
            noc_semaphore_wait_min(my_recv, i);
            // Increment MY OWN slot on the coordinator. Slot b therefore holds master b's arrival count:
            // one writer per slot, monotone, and a fixed number of slots regardless of tp.
            noc_semaphore_inc(get_noc_addr(coord_x, coord_y, get_semaphore(dir_count_sem_id + my_reporter_slot)), 1);
            if (is_dir_coord) {
                // Arrival i is complete chip-wide only once EVERY reporter has reached i.
                for (uint32_t b = 0; b < m_groups; ++b) {
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(dir_count_sem_id + b)), i);
                }
                volatile tt_l1_ptr uint32_t* wv =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(my_wave_sem_id));
                noc_semaphore_set(wv, i);
                publish_to_grid(get_semaphore(my_wave_sem_id), my_dir != 0u);
            }
        };

#if defined(DIRECT_L1)
        // Direct-L1 needs none of the staged path's machinery: no staging accessor, no progressive-arrival
        // publication, no per-rank wave counters. The args are still READ above (the sequential `fa` walk is
        // what lands on the mux client block, so it cannot be skipped) and the on-chip ring below has no
        // arrival gate to evaluate, so these are genuinely dead here. Named explicitly rather than left to
        // -Wunused: the writer's arg block is the one place where a silently-dropped read shifts everything
        // after it.
        (void)stage_acc;
        (void)report_arrival;
        (void)in0;
        (void)k_shard_stride;
        (void)ready_sem_id_c;
        (void)fwd_recv_total;
        (void)bwd_recv_total;
        (void)my_rank;
        (void)my_tp;
        // ================= PHASE 2: DIRECT-L1 STREAMING =================
        // No DRAM staging, no store-and-forward relay buffer, and no credit/window protocol.
        //
        // cb0 slot 0 is the ONLY externally-sourced data this core has (slots 1..G-1 arrive over the on-chip
        // ring below), it is contiguous [base0, base0 + shard_bytes), and under the rank-aligned K mapping it
        // holds tiles from exactly ONE source rank. So this core's entire share of the all-gather is: get
        // slot 0 filled once, then hand those same bytes to the next device.
        //
        //   dist == 0  -> this device owns that rank: read the stripe out of the LOCAL in0 shard.
        //   dist  > 0  -> it arrives over the fabric, written by the upstream device into this same address.
        //
        // Relay source == consume destination, so a relaying core forwards the very slot it consumes: no
        // relay buffer, no extra L1. Nothing in the program ever rewrites slot 0 (slot 0 is written once,
        // slots s>0 only by the ring forward), so there is no window to flow-control -- one arrival
        // semaphore per core is the whole synchronisation.
        //
        // The destination address is OUR OWN base0. The peer core has the same core index and runs the same
        // program with the same CB config, so its cb0 base is ours. That is the same construction the staged
        // path already relies on for DRAM ("mesh tensors share an address across the mesh"); a host-side
        // check is not available because CB L1 addresses are only assigned when the program is finalized at
        // enqueue, after create_at has returned. If direct-L1 ever produces per-device-varying corruption,
        // divergent L1 allocator state across devices is the first thing to suspect.
        {
            constexpr uint32_t slot0_tiles = W * in0_blk;
            constexpr uint32_t slot0_bytes = slot0_tiles * tile_bytes;
            // OUR slot, and the slot the same stripe occupies on the peer. Identical while every device shares
            // one schedule; an arrival-ordered schedule is per-device, so the sender is TOLD the receiver's
            // slot rather than assuming its own -- otherwise a relay lands the stripe where the peer is not
            // expecting to consume it.
            const uint32_t dl1_my_base = base0 + ring_own_slot * shard_bytes;
            volatile tt_l1_ptr uint32_t* dl1_recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dl1_recv_sem_addr);
            if (!dl1_active) {
                // This ring position lies entirely past the last source rank (tp * rank_span < 8 * W * kb):
                // its slot is pure zero padding. Nothing to fetch, nothing to forward -- but it must still be
                // ZEROED, because those tiles are summed into every valid output column.
                uint32_t q = dl1_my_base;
                for (uint32_t t = 0; t < slot0_tiles; ++t) {
                    zero_tile(q);
                    q += tile_bytes;
                }
            } else if (dl1_dist == 0u) {
                // ORIGIN. Read our own shard's stripe straight into slot 0. Capacity-local index l sits at
                // shard column stripe_base + (l % rank_span); slots at or past run_len are this rank's zero
                // padding and are never read (the in1 reader zeroes the same positions, so the product is
                // 0*0, never 0*NaN).
                uint32_t q = dl1_my_base;
                for (uint32_t wb = 0; wb < W; ++wb) {
                    const uint32_t sb = ring_pos * W + wb;  // capacity-local block index of our own slot
                    for (uint32_t m = 0; m < M_block; ++m) {
                        for (uint32_t k = 0; k < K_block; ++k) {
                            const uint32_t l = sb * K_block + k;
                            const uint32_t j = l - (l / k_rank_span) * k_rank_span;  // offset within the rank
                            if (m < valid_m && j < k_run_len) {
                                noc_async_read_page((m_start + m) * k_shard_tiles + k_stripe_base + j, shard_acc, q);
                            } else {
                                zero_tile(q);
                            }
                            q += tile_bytes;
                        }
                    }
                }
                noc_async_read_barrier();
            } else {
                // RELAY / LEAF. Our chunk is written by the upstream device. We do NOT wait for it here:
                // waiting in the prologue is what froze the on-chip ring, because a core stuck here forwards
                // nothing and every core behind it starves. The wait -- and the relay that follows it -- move
                // to the ring step that first needs this chunk (see ensure_own below).
            }
            dl1_recv_sem = dl1_recv_sem_addr;
            if (dl1_active) {
                dl1_own_pending = 1u;
            }
            // Publish what the deferred relay will need; the send itself happens in ensure_own().
            dl1_mux_fa = fa;
            dl1_h_dist = dl1_dist;
            dl1_h_send_fwd = dl1_send_fwd;
            dl1_h_send_bwd = dl1_send_bwd;
            dl1_h_packet_bytes = dl1_packet_bytes;
        }
#else
        // The 8 master-ring cores split the shard by M tiles; bank_id picks the slice. Ceil-divide so the
        // last core takes the short tail rather than dropping rows.
        const uint32_t m_per_core = (Mt_total + 7u) / 8u;
        const uint32_t m_lo = bank_id * m_per_core;
        const uint32_t m_hi = (m_lo + m_per_core < Mt_total) ? (m_lo + m_per_core) : Mt_total;

        // NOTE: deliberately NOT gated on m_lo < m_hi. When Mt_total < 8 the tail cores get an empty M
        // range, but they are still registered mux clients and their peers still expect tp-1 credits from
        // them. Gating here would (a) skip sender.close(), which the v2 mux waits on before self-
        // terminating, and (b) leave every peer's counter one short forever. The tile loops below are
        // naturally empty for those cores, so they open, send zero payload, still credit, and close.
        if (is_fabric_client) {
            // ---- 1. stage OUR OWN shard: in0[m, k] -> staging[m, rank*k_shard_tiles + k] ----
            // Local DRAM->DRAM through an L1 scratch RING. cb0 is not yet in use, so its first
            // kGatherBatch tiles are free staging slots.
            //
            // BATCHED, not per-tile. A read-barrier plus a writes-flush around every single 2 KiB tile
            // makes this a strict serial chain of DRAM-read latency + write latency, with no DMA
            // concurrency at all -- measured as the fused gather gaining NOTHING from a second fabric
            // link, which is what a latency-bound (rather than bandwidth-bound) loop looks like.
            // Issuing kGatherBatch reads, then one barrier, then kGatherBatch writes, then one flush
            // keeps that many transfers in flight and cuts the serialization points by the batch factor.
            const uint32_t scratch = get_write_ptr(gather_scratch_cb);
            // Packet headers come from the per-RISC PacketHeaderPool (12 per RISC on Blackhole),
            // allocated ONCE outside the loops. One per in-flight payload, because the header stays live
            // until the send drains; plus a separate one for the credit, since to_noc_unicast_write and
            // to_noc_unicast_atomic_inc overwrite the same command_fields union.
            volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_hdr_write[kGatherBatch];
            for (uint32_t j = 0; j < kGatherBatch; ++j) {
                pkt_hdr_write[j] = PacketHeaderPool::allocate_header();
            }
            auto* pkt_hdr_seminc = PacketHeaderPool::allocate_header();
            const uint32_t own_k0 = rank * k_shard_tiles;
            const uint32_t local_total = (m_hi > m_lo) ? (m_hi - m_lo) * k_shard_tiles : 0u;
            for (uint32_t t0 = 0; t0 < local_total; t0 += kGatherBatch) {
                const uint32_t nb = ((local_total - t0) < kGatherBatch) ? (local_total - t0) : kGatherBatch;
                for (uint32_t j = 0; j < nb; ++j) {
                    const uint32_t t = t0 + j;
                    noc_async_read_page(
                        (m_lo + t / k_shard_tiles) * k_shard_tiles + (t % k_shard_tiles),
                        shard_acc,
                        scratch + j * tile_bytes);
                }
                noc_async_read_barrier();
                for (uint32_t j = 0; j < nb; ++j) {
                    const uint32_t t = t0 + j;
                    noc_async_write_page(
                        (m_lo + t / k_shard_tiles) * Kt_global + own_k0 + (t % k_shard_tiles),
                        stage_acc,
                        scratch + j * tile_bytes);
                }
                // The writes SOURCE from the scratch slots, which the next batch's reads overwrite.
                // read_barrier orders reads only, so this flush is what makes slot reuse safe.
                noc_async_writes_flushed();
            }
            noc_async_write_barrier();

            // ---- 1b. barrier across the masters: local staging complete everywhere ----
            // The local copy splits M 8 ways by bank_id, but the fabric send below splits it by
            // (bank_id mod m_groups) -- a DIFFERENT partition. So the pages a core is about to forward
            // were, in general, written by some OTHER core. At Mt=1 that is stark: bank 0 stages the only
            // tile row and bank 4 immediately forwards it. Without this barrier the backward stream ships
            // a half-written shard: measured as PCC 0.79 with rank d+1 partly zero on the small shapes.
            //
            // Every master credits every master (including itself) and then waits for the full count, so
            // there is no coordinator and no second flag. 8x8 atomic incs is nothing next to the transfer.
            // noc_async_write_barrier() above is what makes our staging writes visible before we credit.
            {
                volatile tt_l1_ptr uint32_t* local_done =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(local_done_sem_id));
                for (uint32_t mi = 0; mi < num_masters; ++mi) {
                    noc_semaphore_inc(
                        get_noc_addr(
                            get_arg_val<uint32_t>(master_base + 2u * mi),
                            get_arg_val<uint32_t>(master_base + 2u * mi + 1u),
                            get_semaphore(local_done_sem_id)),
                        1);
                }
                noc_async_atomic_barrier();
                noc_semaphore_wait(local_done, num_masters);
            }

            // Wave 0 -- the local shard -- is staged chip-wide now. Publish it immediately: the entire
            // point of progressive consumption is that consumers start on it while the remote shards are
            // still crossing the fabric.
            if (is_master0) {
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem_id)), VALID);
                publish_to_grid(get_semaphore(ready_sem_id), false);
            }

            // ---- 2. bidirectional store-and-forward ring ----
            // This core drives ONE direction (my_dir). On a RING fwd runs tp/2 rounds and bwd the
            // remainder, so the two sum to exactly tp-1 and at even tp the antipode rides the forward
            // stream alone -- delivered exactly once, never twice and never dropped. On a LINE the counts
            // are per-rank and send != recv (node d forwards the d+1 shards from 0..d but receives only
            // d), which is why the two arrive as separate runtime args.
            //
            // Round r forwards the shard that ORIGINATED at rank -/+ (r-1) (forward/backward). It already
            // sits at the same staging offset here as on the neighbour, so source and destination offsets
            // are identical -- the hop is a straight staging->staging copy. Round r (r >= 2) may only
            // forward once round r-1 has landed, tracked by this direction's own credit counter.
            //
            // The FABRIC M split is by (bank_id mod m_groups), not bank_id: each direction has half the
            // masters and they must cover all of M between them. The local copy above stays 8-way.
            const uint32_t fab_group = bank_id % m_groups;
            const uint32_t fab_per = (Mt_total + m_groups - 1u) / m_groups;
            const uint32_t fab_lo = fab_group * fab_per;
            const uint32_t fab_hi = (fab_lo + fab_per < Mt_total) ? (fab_lo + fab_per) : Mt_total;

            if (has_fwd || has_bwd) {
                auto sender = tt::tt_fabric::FabricMuxV2Sender<>::build_from_args(fa);
                sender.open();
                volatile tt_l1_ptr uint32_t* my_recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_recv_sem_addr);

#if defined(ABLATE_NOGATHER)
                for (uint32_t r = 1; r < 1u; ++r) {  // ABLATION: no payload, no credits
#else
                for (uint32_t r = 1; r <= my_send_rounds; ++r) {
#endif
                    if (r >= 2) {
                        // This round has to wait for arrival r-1 before it can forward it, so publish that
                        // arrival here rather than in a separate pass -- consumers get it as early as it
                        // can possibly be known.
                        report_arrival(r - 1);
                    }
                    // Forward carries ranks rank-1, rank-2, ...; backward carries rank+1, rank+2, ...
                    const uint32_t src_rank = (my_dir == 0u) ? ((rank + tp - (r - 1)) % tp) : ((rank + (r - 1)) % tp);
                    const uint32_t k0 = src_rank * k_shard_tiles;
                    // Batched egress: kGatherBatch tiles are read, then injected, then flushed ONCE.
                    // Per-tile barriers made this a serial chain -- see the batching note on the local
                    // staging loop above. Each in-flight tile needs its own packet header and its own
                    // scratch slot, since both stay live until the send drains.
                    //
                    // Destination is the SAME page of the SAME staging buffer on the neighbour. Mesh
                    // tensors share an address across the mesh, so the local address of a page is also
                    // its address on the peer -- the fabric supplies only the chip hop. Built through the
                    // fabric addrgen helper because it emits noc0 coords and DRAM has no virtual coords on
                    // some archs; signature is (accessor, page_id, offset), NOT (page_id, accessor).
                    // ABLATE_NOPAYLOAD: no payload bytes, but the round still credits and the sender still
                    // opens/closes, so the schedule and the arrival dependency are untouched. Implemented on
                    // this path too so the flag cannot silently no-op when direct-L1 is off.
#if defined(ABLATE_NOPAYLOAD)
                    const uint32_t fab_total = 0u;
#else
                    const uint32_t fab_total = (fab_hi > fab_lo) ? (fab_hi - fab_lo) * k_shard_tiles : 0u;
#endif
                    for (uint32_t t0 = 0; t0 < fab_total; t0 += kGatherBatch) {
                        const uint32_t nb = ((fab_total - t0) < kGatherBatch) ? (fab_total - t0) : kGatherBatch;
                        for (uint32_t j = 0; j < nb; ++j) {
                            const uint32_t t = t0 + j;
                            const uint32_t page = (fab_lo + t / k_shard_tiles) * Kt_global + k0 + (t % k_shard_tiles);
                            noc_async_read_page(page, stage_acc, scratch + j * tile_bytes);
                        }
                        noc_async_read_barrier();
                        for (uint32_t j = 0; j < nb; ++j) {
                            const uint32_t t = t0 + j;
                            const uint32_t page = (fab_lo + t / k_shard_tiles) * Kt_global + k0 + (t % k_shard_tiles);
                            const uint64_t dst_noc =
                                tt::tt_fabric::linear::addrgen_detail::get_noc_address(stage_acc, page, 0);
                            tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_write(
                                &sender,
                                pkt_hdr_write[j],
                                scratch + j * tile_bytes,
                                tile_bytes,
                                tt::tt_fabric::NocUnicastCommandHeader{dst_noc},
                                /*num_hops=*/1);
                        }
                        // One flush per batch: headers and scratch slots are reusable once the sends have
                        // been injected.
                        noc_async_writes_flushed();
                    }
                    // Credit AFTER the payload. Ordering is enforced on the RECEIVING chip: flush=true makes
                    // the peer drain every prior write on this channel before applying the increment.
                    // sender.flush() would NOT do this -- it is a no-op unless EAGER_STAGING is on.
                    // Same core index on the peer chip: core i owns a disjoint M slice on every device, so
                    // core i credits core i and each core's counter is its own private arrival count.
                    //
                    // NOC0 coords (my_x[0], not my_x[noc_index]): the packet-header setter re-encodes the
                    // address with its own mirroring, so handing it an already-noc1-mirrored coordinate
                    // mirrors twice and credits the mirror-image core. Invisible on Blackhole, where
                    // virtual coords make my_x[0] == my_x[1]; a hang on Wormhole. This writer runs on
                    // RISCV_1 / noc_index == 1, so the distinction is live.
                    const uint64_t peer_sem_noc = safe_get_noc_addr(my_x[0], my_y[0], my_recv_sem_addr, 0);
                    tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
                        &sender,
                        pkt_hdr_seminc,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{peer_sem_noc, 1, /*flush=*/true},
                        /*num_hops=*/1);
                    noc_async_writes_flushed();
                }
                // Drain writes AND non-posted atomics before closing, or the kernel exits with pending NOC
                // transactions. close() is mandatory: the v2 mux self-terminates only once every client has
                // closed, so skipping it hangs the mux kernels.
                noc_async_write_barrier();
                noc_async_atomic_barrier();
                sender.close();
            }
        }

        // ---- 3. publish each remaining arrival as it lands ----
        // Arrivals 1..my_send_rounds-1 were already published inside the send loop (the loop has to wait
        // for them anyway before it can forward). Anything past the last send round has no forwarding
        // work attached to it, so it is picked up here.
#if defined(ABLATE_NOGATHER)
        if (false) {  // ABLATION: nothing to report, nobody is waiting
#else
        if (is_fabric_client) {
            // Start at 1, not my_send_rounds: at a LINE end there is no forwarding work in this
            // direction (send_rounds == 0) and i == 0 would be a spurious report that inflates the
            // coordinator's count and publishes a wave before it has landed.
            const uint32_t tail_lo = (my_send_rounds > 1u) ? my_send_rounds : 1u;
            for (uint32_t i = tail_lo; i <= my_recv_rounds; ++i) {
                report_arrival(i);
            }
#endif
        }

        // ---- 4. (the all-or-nothing barrier is gone) ----
        // Consumers no longer wait for the whole gather. Wave 0 was published right after the local
        // staging barrier above, and each remote arrival is published by its direction coordinator as it
        // lands; the in0 ring below gates each shard on its own source rank. All that remains here is to
        // re-arm this direction's credit counter for the next invocation.
        if (is_fabric_client) {
            volatile tt_l1_ptr uint32_t* my_recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_recv_sem_addr);
            noc_semaphore_wait_min(my_recv, my_recv_rounds);
            noc_semaphore_set(my_recv, 0);
        }
#endif  // DIRECT_L1 vs DRAM-staged
    }
#endif  // FUSED_GATHER

    // ---- PHASE 1: in0 ring all-gather (balanced tails: read only valid M rows / valid K, else zero) ----
    // Step `step` consumes SLOT `step` -- slot index is the consumption-order index, see the note at
    // ring_own_slot. What varies is which stripe the host put there, and that is the whole schedule.
#if defined(DIRECT_L1)
    // Ordered by AVAILABILITY, not by ring position -- the host sorted each core's chunks by
    // fabric_hops*WAVE + on_chip_hops*HOP and emitted that as the slot schedule. Two things follow, and both
    // matter:
    //
    //  * ONE GATE PER STEP. Step s forwards the very chunk step s consumes, so there is a single wait covering
    //    both and forwarding can never block on a chunk that lands later than the one we are about to use.
    //    Compressing the G-1 forwards into the first G-1 steps caused exactly that: every step at or after the
    //    successor's own chunk forwarded C[s+1] while consuming C[s], so a ready chunk queued behind a later
    //    one. The step consuming the successor's own chunk forwards nothing (it already has it).
    //  * The own-chunk wait AND the fabric relay land here (dl1_own_pending), at the step that first needs that
    //    chunk, instead of in the prologue -- a core stuck in the prologue forwards nothing and starves every
    //    core behind it.
    for (uint32_t step = 0; step < G; ++step) {
        // Our own chunk: wait for it if it comes over the fabric, then IMMEDIATELY relay it onward. Runs at the
        // first ring step that needs this chunk, and exactly once (dl1_own_pending guards it), so the mux still
        // sees exactly one open and one close -- which is what mux v2 counts to self-terminate.
        //
        // Fires at step 0 for an ORIGIN, because an origin's own chunk has 0 fabric hops and 0 on-chip hops, so
        // availability order always sorts it first. Origins therefore still relay as eagerly as before, while
        // every other core gets to forward and push the chunks that are already in hand first.
        auto ensure_own = [&]() {
            if (dl1_own_pending == 0u) {
                return;
            }
            dl1_own_pending = 0u;
            volatile tt_l1_ptr uint32_t* dl1_recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dl1_recv_sem);
            if (dl1_h_dist != 0u) {
#if !defined(ABLATE_NOWAIT)
                noc_semaphore_wait_min(dl1_recv, 1);
#endif
                // Re-arm for the next invocation: a GLOBAL semaphore is not zeroed by program launch, which is
                // exactly why the cross-chip credit lives in one.
                noc_semaphore_set(dl1_recv, 0);
            }
            const uint32_t dl1_dist = dl1_h_dist;
            const uint32_t dl1_send_fwd = dl1_h_send_fwd;
            const uint32_t dl1_send_bwd = dl1_h_send_bwd;
            const uint32_t dl1_packet_bytes = dl1_h_packet_bytes;
            const uint32_t dl1_my_base = base0 + ring_own_slot * shard_bytes;
            constexpr uint32_t slot0_tiles = W * in0_blk;
            constexpr uint32_t slot0_bytes = slot0_tiles * tile_bytes;
            const uint32_t dl1_recv_sem_addr = dl1_recv_sem;
            std::size_t fa = dl1_mux_fa;
            (void)dl1_dist;
            if (dl1_send_fwd || dl1_send_bwd) {
                // Packet headers from the per-RISC PacketHeaderPool (12 on Blackhole), allocated ONCE: one
                // per in-flight payload (the header stays live until the send drains) plus a separate one
                // for the credit, since to_noc_unicast_write and to_noc_unicast_atomic_inc overwrite the
                // same command_fields union.
                constexpr uint32_t kDl1Batch = 8;
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[kDl1Batch];
                for (uint32_t j = 0; j < kDl1Batch; ++j) {
                    hdr[j] = PacketHeaderPool::allocate_header();
                }
                auto* hdr_sem = PacketHeaderPool::allocate_header();
                // Same core index on the peer: core i holds the same slot of the same Pk group on every
                // device, so core i credits core i and each core's counter is its own private arrival flag.
                // NOC0 coords (my_x[0], not my_x[noc_index]): the packet-header setter re-encodes the
                // address with its own mirroring, so an already-noc1-mirrored coordinate mirrors twice and
                // aims at the mirror-image core. Invisible on Blackhole, a hang on Wormhole.
                const uint64_t peer_sem = safe_get_noc_addr(my_x[0], my_y[0], dl1_recv_sem_addr, 0);
                // A LINE origin drives BOTH muxes (the stripe has to fan out either way from its owner);
                // every other core drives exactly one. The host appends the client blocks in this same
                // fwd-then-bwd order, and only for directions this core actually sends in, so reading them
                // in order off `fa` lands on the right one.
                for (uint32_t dir = 0; dir < 2u; ++dir) {
                    if ((dir == 0u) ? (dl1_send_fwd == 0u) : (dl1_send_bwd == 0u)) {
                        continue;
                    }
                    // Each neighbour has its own schedule, so its slot for this chunk differs.
                    const uint32_t dl1_peer_base =
                        base0 + (dir == 0u ? ring_peer_slot_fwd : ring_peer_slot_bwd) * shard_bytes;
                    auto sender = tt::tt_fabric::FabricMuxV2Sender<>::build_from_args(fa);
                    sender.open();
                    // ABLATE_NOPAYLOAD sends no bytes but still opens, credits and closes -- so the mux
                    // plumbing, the credit traffic and the arrival dependency all stay, and only the payload
                    // disappears. Timing only: consumers get a credit for data that never arrived.
#if defined(ABLATE_NOPAYLOAD)
                    constexpr uint32_t dl1_payload_bytes = 0u;
#else
                    constexpr uint32_t dl1_payload_bytes = slot0_bytes;
#endif
                    // Slot 0 is CONTIGUOUS in L1, so the transfer is bounded only by the fabric packet, not
                    // by the tile. dl1_packet_bytes is as many whole bf16 tiles as one packet holds (the host
                    // derives it from the fabric's own max payload), which halves the packet count -- and with
                    // it the headers, mux slot handoffs and NoC transactions -- versus sending tile by tile.
                    // The last packet is short whenever slot0_bytes is not a whole multiple.
                    for (uint32_t off0 = 0; off0 < dl1_payload_bytes; off0 += kDl1Batch * dl1_packet_bytes) {
                        uint32_t j = 0;
                        for (uint32_t off = off0; off < dl1_payload_bytes && j < kDl1Batch;
                             off += dl1_packet_bytes, ++j) {
                            const uint32_t n = ((dl1_payload_bytes - off) < dl1_packet_bytes)
                                                   ? (dl1_payload_bytes - off)
                                                   : dl1_packet_bytes;
                            tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_write(
                                &sender,
                                hdr[j],
                                dl1_my_base + off,
                                n,
                                tt::tt_fabric::NocUnicastCommandHeader{
                                    safe_get_noc_addr(my_x[0], my_y[0], dl1_peer_base + off, 0)},
                                /*num_hops=*/1);
                        }
                        // One flush per batch: the headers (not the source, which is never rewritten) are
                        // what has to be free before the next batch reuses them.
                        noc_async_writes_flushed();
                    }
                    // Credit AFTER the payload. Ordering is enforced on the RECEIVING chip: flush=true makes
                    // the peer drain every prior write on this channel before applying the increment.
                    // sender.flush() would NOT do this -- it is a no-op unless EAGER_STAGING is on.
                    tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
                        &sender,
                        hdr_sem,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{peer_sem, 1, /*flush=*/true},
                        /*num_hops=*/1);
                    // Drain writes AND non-posted atomics before closing, or the kernel exits with pending
                    // NOC transactions. close() is mandatory: the v2 mux self-terminates only once every
                    // client has closed, so skipping it hangs the mux kernels.
                    noc_async_write_barrier();
                    noc_async_atomic_barrier();
                    sender.close();
                }
            }
        };
        // Number of received chunks needed before slot `sl` is filled: every slot below it except our own.
        auto recv_needed = [&](uint32_t sl) { return (ring_own_slot <= sl) ? sl : (sl + 1u); };
        // THE one wait of this step: the chunk we are about to consume. Nothing else gates here.
        if (step == ring_own_slot) {
            ensure_own();
        } else {
            noc_semaphore_wait_min(fwd_ptr, recv_needed(step));
        }
        // Relay that same chunk onward. src_slot == step by construction; the sentinel marks the single step
        // whose chunk is the successor's own, which it already has.
        const uint32_t fwd_src_slot = get_arg_val<uint32_t>(kRingFwdBase + 2u * step);
        if (fwd_src_slot != kRingNoForward) {
            const uint32_t fwd_dst_slot = get_arg_val<uint32_t>(kRingFwdBase + 2u * step + 1u);
            const uint64_t dst = get_noc_addr(fwd_next_x, fwd_next_y, base0 + fwd_dst_slot * shard_bytes);
            noc_async_write(base0 + fwd_src_slot * shard_bytes, dst, shard_bytes);
            // payload THEN readiness, same peer + same NoC, so the successor cannot observe the credit early
            noc_semaphore_inc(get_noc_addr(fwd_next_x, fwd_next_y, fwd_addr), 1);
        }
        cb_push_back(in0_cb, W * in0_blk);  // compute consumes this chunk (W blocks)
    }
#else
    for (uint32_t step = 0; step < G; ++step) {
        uint32_t slot = base0 + step * shard_bytes;
        if (step == ring_own_slot) {
            // read our OWN shard (shard index = ring_pos) from DRAM into our slot (+ barrier).
            uint32_t p = slot;
            for (uint32_t wb = 0; wb < W; ++wb) {
                const uint32_t sb = ring_pos * W + wb;  // capacity-local block index of own shard
#if defined(FUSED_GATHER)
                // PROGRESSIVE CONSUMPTION. Gate this block on the SOURCE RANK it comes from rather than on
                // the whole gather. Under the blocked-cyclic mapping l / run_len IS the source rank, so a
                // block spans at most two ranks and waiting on the later one covers it.
                //
                // Each of the 8 ring cores owns a different contiguous stretch of K, hence different source
                // ranks -- so cores whose data has already landed read, forward and start feeding compute
                // while the remaining shards are still in flight. That is where the overlap comes from.
                if (k_run_len != 0u && valid_k != 0u) {
#if defined(ABLATE_NOWAIT)
                }
                if (false) {  // ABLATION: consumers do not gate on shard arrival
#endif
                    // Wait for EVERY source rank this block touches, not just the last one. l / run_len is
                    // the rank in NUMERIC order, but arrival order is local-first then outward, so a higher
                    // numeric rank can land before a lower one -- waiting only on the block's last rank
                    // would let an earlier, later-arriving rank be read too soon. Measured as PCC 0.93-0.97.
                    //
                    // The upper bound is clamped into the valid range: tiles past valid_k are zero-filled,
                    // never read, and would otherwise index a rank that does not exist.
                    const uint32_t l_lo = sb * K_block;
                    uint32_t l_hi = l_lo + K_block - 1u;
                    if (l_hi >= valid_k) {
                        l_hi = valid_k - 1u;
                    }
                    if (l_lo <= l_hi) {
                        for (uint32_t src = l_lo / k_run_len; src <= l_hi / k_run_len; ++src) {
                            if (src == my_rank) {
                                // Local shard: published right after the on-chip staging barrier.
                                noc_semaphore_wait(
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem_id_c)),
                                    VALID);
                            } else {
                                // Forward carries my_rank-1, my_rank-2, ...; backward my_rank+1, my_rank+2.
                                const uint32_t a = (my_rank + my_tp - src) % my_tp;
                                if (a <= fwd_recv_total) {
                                    noc_semaphore_wait_min(
                                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(wave_fwd_sem_id)),
                                        a);
                                } else {
                                    const uint32_t b = (src + my_tp - my_rank) % my_tp;
                                    noc_semaphore_wait_min(
                                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(wave_bwd_sem_id)),
                                        b);
                                }
                            }
                        }
                    }
                }
#endif
                for (uint32_t m = 0; m < M_block; ++m) {
                    for (uint32_t k = 0; k < K_block; ++k) {
                        const uint32_t l = sb * K_block + k;  // capacity-local K index within the slice
                        // Written with k_rank_span rather than k_run_len so this is LITERALLY the in1
                        // reader's global_k/k_valid_at. On this (staged) path the two are equal, so it is a
                        // no-op today; keeping one formula is what stops the two sides drifting apart.
                        if (m < valid_m && l < valid_k &&
                            (k_run_len == 0u || (l - (l / k_rank_span) * k_rank_span) < k_run_len)) {
                            // Capacity-local K index -> global staging column. On the fused path every
                            // Pk group owns a stripe of EVERY source rank's shard, so consecutive l walk
                            // one stripe and then jump a whole shard; k_run_len == 0 is the single-chip
                            // contiguous case.
                            const uint32_t gk =
                                (k_run_len == 0u)
                                    ? (k_start + l)
                                    : ((l / k_rank_span) * k_shard_stride + k_stripe_base + (l % k_rank_span));
                            noc_async_read_page((m_start + m) * Kt + gk, in0, p);
                        } else {
                            zero_tile(p);  // pad M row or K tail -> local zero (no DRAM read)
                        }
                        p += tile_bytes;
                    }
                }
            }
            noc_async_read_barrier();
        } else {
            // Received stripes land in ascending slot order, so consuming slot `step` needs this many of
            // them: every slot up to `step` except our own. (own_slot == 0 makes this `step`, as before.)
            noc_semaphore_wait_min(fwd_ptr, (ring_own_slot <= step) ? step : (step + 1u));
        }
        if (step + 1 < G) {  // forward per the schedule: read one of my slots, write one of my successor's
            const uint32_t fwd_src_slot = get_arg_val<uint32_t>(kRingFwdBase + 2u * step);
            const uint32_t fwd_dst_slot = get_arg_val<uint32_t>(kRingFwdBase + 2u * step + 1u);
            const uint64_t dst = get_noc_addr(fwd_next_x, fwd_next_y, base0 + fwd_dst_slot * shard_bytes);
            noc_async_write(base0 + fwd_src_slot * shard_bytes, dst, shard_bytes);
            // payload THEN readiness, same peer + same NoC, so the successor cannot observe the credit early
            noc_semaphore_inc(get_noc_addr(fwd_next_x, fwd_next_y, fwd_addr), 1);
        }
        cb_push_back(in0_cb, W * in0_blk);  // compute consumes this shard (W blocks)
    }
#endif                          // !DIRECT_L1 ring loop
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
#if defined(OUT_CHUNKS)
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
        // Pipelined: single deferred completion before return. BOTH barriers are required even on this
        // Pk == 1 path: the in0 ring forward above issues non-posted semaphore atomics (noc_semaphore_inc),
        // and noc_async_write_barrier() drains writes ONLY -- it does not wait for atomic acks. Without the
        // atomic barrier the kernel can retire with acks still in flight, which the watcher reports as
        // "kernel completing with pending NOC transactions (missing NOC non-posted atomics flushed
        // barrier)". Costs nothing in practice here (phase 2 gives the acks ample time to land), but the
        // guarantee must not depend on that timing.
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        return;
    }

#if defined(RSCATTER)
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
    constexpr uint32_t cb_send = 8;  // NOT 4/5: those are fusion operands, which reduce-scatter now supports
    constexpr uint32_t cb_recv = 9;
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
#if defined(FUSE_BIAS) || defined(FUSE_TERNARY)
        // Supply this owner's slice operands BEFORE waiting on out_cb: compute applies the epilogue to the
        // fully reduced slice, so the operands must already be queued when the last ring round completes.
        feed_fused_slice(nb, own_off, own_tiles);
#endif
        cb_wait_front(out_cb, max_chunk);
        const uint32_t rr = get_read_ptr(out_cb);
        for (uint32_t j = 0; j < own_tiles; ++j) {
            const uint32_t idx = own_off + j;
            const uint32_t m = idx / N_block;
            const uint32_t n = idx - m * N_block;
            if (m < valid_m && (n_base + n) < valid_n) {
#if defined(OUT_CHUNKS)
                write_out_tile(m_start + m, n_start + n_base + n, rr + j * tile_bytes);
#else
                noc_async_write_page((m_start + m) * Nt + (n_start + n_base + n), out, rr + j * tile_bytes);
#endif
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
    volatile tt_l1_ptr uint32_t* red_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_addr);
    const uint32_t redfree_addr = get_semaphore(redfree_sem_id);
    volatile tt_l1_ptr uint32_t* redfree_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(redfree_addr);
    const uint64_t prev_redfree = get_noc_addr(red_prev_x, red_prev_y, redfree_addr);
    const uint64_t next_recv = get_noc_addr(red_next_x, red_next_y, red_addr);

    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
        if (!is_bottom) {
            cb_reserve_back(cb_reduce, out_blk);      // wait our compute freed slot (nb-2)
            noc_semaphore_inc(prev_redfree, 1);       // tell prev: our slot (nb%2) is free for block nb
            noc_semaphore_wait_min(red_ptr, nb + 1);  // prev forwarded block nb into it (chain latency)
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
            noc_semaphore_wait_min(redfree_ptr, nb + 1);  // next signalled its slot is free
            const uint64_t dst = get_noc_addr(red_next_x, red_next_y, reduce_base + (nb % red_depth) * out_blk_bytes);
            noc_async_write(r, dst, out_blk_bytes);
            // Pipelined: payload THEN signal to the SAME peer on the SAME NoC (ordered, like the in0 ring) so
            // the receiver never observes readiness before its partial-sum has landed. Flush (not a full
            // barrier) so the out_cb source slot is reusable; completion is deferred to the final barrier.
            noc_semaphore_inc(next_recv, 1);  // block nb delivered (ordered after the payload write)
            noc_async_writes_flushed();       // payload departed L1 -> out_cb slot safe to reuse
        } else {
            // ROOT: issue output DRAM pages + flush (the reduction tail on the wall).
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
            noc_async_writes_flushed();  // output pages departed L1 -> out_cb slot safe to reuse
        }
        cb_pop_front(out_cb, out_blk);
    }
#endif  // RSCATTER vs chain
    // Pipelined: single deferred completion before return — drain this core's forwarded partial-sums / DRAM
    // output writes AND the non-posted reduction-readiness semaphore atomics (noc_semaphore_inc), so no
    // in-flight NoC transaction outlives the program (writes_flushed above only guarantees source-L1 reuse).
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
