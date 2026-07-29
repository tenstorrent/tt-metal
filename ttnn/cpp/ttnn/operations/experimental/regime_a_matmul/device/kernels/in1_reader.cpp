// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Regime-A in1 reader == consumer (runs on the core's in1 NoC/RISC).
//
// in1 is DRAM width-sharded across 8 banks; this core owns one bank's N-sub-band for its k-slice. It
// reads that sub-band's [kb, N_sub] blocks in ROTATED shard order so each in1[k] block pairs with the
// in0[k] block arriving via the in0 ring; the K-sum is commutative so any consistent pairing works.
//
// BALANCED TAILS: the core owns valid_k logical K tiles (of K_slice_capacity capacity, distributed over
// the ring) and valid_n logical N tiles (of N_bpc*N_sub capacity). Positions beyond valid_k / valid_n
// are LOCALLY ZERO-FILLED — never DRAM-read. Address strides come from the tensor layout
// (in1_shard_stride_n = physical per-bank shard width), NOT from schedule capacities. For divisible
// shapes valid == capacity, so the zero-fill paths never run and this is byte-identical to the fast path.
//
// M-split (m_slices > 1): the m==0 reader reads once + forwards each (fixed-size, possibly zero-padded)
// block to the Sm-1 slaves; slaves receive only.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t K_block = get_compile_time_arg_val(0);             // kb
    constexpr uint32_t N_block = get_compile_time_arg_val(1);             // N_sub
    constexpr uint32_t W = get_compile_time_arg_val(2);                   // in1 blocks per ring shard
    constexpr uint32_t G = get_compile_time_arg_val(3);                   // ring size (8 banks)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);          // bf16 tile bytes
    constexpr uint32_t N_bpc = get_compile_time_arg_val(5);               // N-subblocks per core
    constexpr uint32_t in1_shard_stride_n = get_compile_time_arg_val(6);  // physical per-bank shard width (tiles)
    constexpr uint32_t in1valid_sem = get_compile_time_arg_val(7);        // M-split: reader -> slaves "delivered"
    constexpr uint32_t in1ready_sem = get_compile_time_arg_val(8);        // M-split: slaves -> reader "slot free"
    constexpr uint32_t cb1_depth = get_compile_time_arg_val(9);           // cb1 depth in BLOCKS (TRID pipeline)

    const uint32_t in1_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t ring_pos = get_arg_val<uint32_t>(2);
    const uint32_t k_start = get_arg_val<uint32_t>(3);  // first logical K tile of this slice (balanced)
    const uint32_t n_local = get_arg_val<uint32_t>(4);  // column offset within this core's bank shard
    const uint32_t valid_k = get_arg_val<uint32_t>(5);  // valid K tiles (rest of capacity zero-filled)
    const uint32_t valid_n = get_arg_val<uint32_t>(6);  // valid N tiles this core owns
    const uint32_t mrole = get_arg_val<uint32_t>(7);    // 0 = slave, 1 = reader(read+fwd), 2 = solo
    const uint32_t mpeers = get_arg_val<uint32_t>(8);   // forward peer count
    // M-split peer coords (only present when Sm > 1) start at arg 9.

    constexpr uint32_t in1_cb = 1;
    constexpr uint32_t in1_blk = K_block * N_block;
    constexpr uint32_t in1_blk_bytes = in1_blk * tile_bytes;
    constexpr uint32_t seg_bytes = N_block * tile_bytes;  // one K-row of a block = N_sub tiles
    constexpr uint32_t words_per_tile = tile_bytes / 4u;

    // Zero `ntiles` bf16 tiles at L1 `addr`. Used ONLY for the small K-tail (l >= valid_k) within valid-N
    // subblocks: those tiles are summed into every valid output column, so they must be exactly 0.0 —
    // NOT left as (possibly NaN/Inf) uninitialized L1, since 0*NaN = NaN would poison the K reduction.
    auto zero_l1 = [](uint32_t addr, uint32_t ntiles) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        const uint32_t n = ntiles * words_per_tile;
        for (uint32_t i = 0; i < n; ++i) {
            p[i] = 0u;
        }
    };

    // ---- M-split SLAVE: receive in1 from the reader, do not touch DRAM. ----
    if (mrole == 0) {
        volatile tt_l1_ptr uint32_t* valid =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(in1valid_sem));
        uint64_t reader_ready =
            get_noc_addr(get_arg_val<uint32_t>(9), get_arg_val<uint32_t>(10), get_semaphore(in1ready_sem));
        const uint32_t nblk = N_bpc * G * W;
        for (uint32_t b = 0; b < nblk; ++b) {
            cb_reserve_back(in1_cb, in1_blk);
            noc_semaphore_inc(reader_ready, 1);
            noc_semaphore_wait_min(valid, b + 1);
            cb_push_back(in1_cb, in1_blk);
        }
        // Drain the non-posted `reader_ready` semaphore atomics before exit (the reader already observed them
        // via its own valid signalling, but the atomics must not outlive the program — the watcher flags this).
        noc_async_atomic_barrier();
        return;
    }

    // ---- M-split READER forward helper (no-op for solo). ----
    const uint32_t in1valid_addr = get_semaphore(in1valid_sem);
    const uint32_t in1ready_addr = get_semaphore(in1ready_sem);
    volatile tt_l1_ptr uint32_t* in1ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1ready_addr);
    uint32_t mbc = 0;
    auto mfwd = [&](uint32_t w1) {
        if (mrole != 1) {
            return;
        }
        noc_semaphore_wait_min(in1ready, (mbc + 1) * mpeers);
        for (uint32_t s = 0; s < mpeers; ++s) {
            uint32_t sx = get_arg_val<uint32_t>(9 + s * 2), sy = get_arg_val<uint32_t>(10 + s * 2);
#if !defined(SKIP_IN1_FORWARD)
            noc_async_write(w1, get_noc_addr(sx, sy, w1), in1_blk_bytes);
#else
            // TEST-ONLY: drop the M-split in1 forward PAYLOAD. The credit wait above and the validity
            // increment below are preserved, so the reader/slave handshake and block count are unchanged and
            // only the NoC copy disappears (slaves then compute on stale L1 - output intentionally invalid).
            (void)sx;
            (void)sy;
#endif
        }
        // Signal EARLY, then flush PER-BLOCK. The early valid-inc releases the slave without waiting on the
        // reader's flush (same-NoC write-before-inc keeps the destination from observing validity before the
        // payload lands); the per-block flush that follows is REQUIRED for SOURCE lifetime -- it guarantees
        // the async write has departed this CB slot before the slot is pushed, wrapped, and overwritten by a
        // later block (an exit-only barrier would be too late). The flush is merely off the slave-release
        // critical path, NOT removed.
        for (uint32_t s = 0; s < mpeers; ++s) {
            uint32_t sx = get_arg_val<uint32_t>(9 + s * 2), sy = get_arg_val<uint32_t>(10 + s * 2);
            noc_semaphore_inc(get_noc_addr(sx, sy, in1valid_addr), 1);
        }
        noc_async_writes_flushed();
        ++mbc;
    };

    // ---- Strided sub-band read in rotated shard order, with balanced tails. ----
    // NO local zero-fill here: invalid positions are simply NOT read (left as garbage). Correctness is
    // preserved by the writer, which zeros in0 for K/M tails -> 0*garbage == 0 kills the K-tail term; and
    // pad-N columns (>= valid_n) are never written to the output. This keeps the reader on its fast path
    // and confines the (cheap) tail zeroing to the small in0 buffer in the writer.
#if defined(IN1_ONE_PACKET)
    // ---- Stateful one-packet in1 row reads (diag bit25). Every read a core issues goes to the SAME DRAM bank
    // (bank_id is fixed per core) and full-width row reads all have the SAME size, so the NoC command buffer's
    // size/config registers only have to be written ONCE instead of per transaction; each read then supplies
    // only its offset. Requires size <= NOC_MAX_BURST_SIZE, checked at compile time.
    //
    // The low 32 bits of get_noc_addr_from_bank_id(bank, a) are (bank base + a), and the coordinate bits live
    // in the state, so `bank_lo + in1_addr + off` is the correct with_state argument.
    // CAUTION: a plain noc_async_read shares read_cmd_buf and CLOBBERS this state, so any block that cannot use
    // the uniform path must re-set it afterwards (see rearm below). Tail blocks are rare, so this costs nothing
    // on the steady-state path.
    constexpr bool one_packet_ok = (N_block * tile_bytes) <= NOC_MAX_BURST_SIZE;
    const uint32_t bank_lo = static_cast<uint32_t>(get_noc_addr_from_bank_id<true>(bank_id, 0));
    auto rearm = [&]() {
        if constexpr (one_packet_ok) {
            noc_async_read_one_packet_set_state(
                get_noc_addr_from_bank_id<true>(bank_id, in1_addr), N_block * tile_bytes);
        }
    };
    rearm();
#endif

    // ---- Issue one block's reads into L1 at `w1`. Does NOT barrier and does NOT touch the CB, so the caller
    // chooses the completion policy: the default path barriers immediately, the TRID pipeline defers. When
    // `trid` is non-zero every read is tagged with it so the caller can wait on that block alone. ----
    auto issue_block_reads = [&](uint32_t kblk, uint32_t ncol_base, uint32_t vcols, uint32_t w1, uint32_t trid) {
        if (vcols == 0u) {
            return;  // whole subblock is pad N: no reads; block stays garbage, output not written
        }
        // Coalesce the whole [K_block x vcols] block into ONE read when it is physically contiguous in the
        // bank shard: full owned width (vcols==N_block==shard stride), zero column offset, and NO K-tail in
        // this block. Consecutive K rows are then adjacent (gk*stride) with a contiguous L1 destination, so
        // one read replaces K_block per-row reads (-0.5..-3.1%, PCC-exact). Falls back to per-row otherwise.
        const bool contig = (vcols == N_block) && (N_block == in1_shard_stride_n) && ((n_local + ncol_base) == 0u) &&
                            ((kblk * K_block + K_block) <= valid_k);
        if (trid) {
            noc_async_read_set_trid(trid);
        }
        if (contig) {
            const uint32_t off = (k_start + kblk * K_block) * in1_shard_stride_n * tile_bytes;
#if !defined(SKIP_IN1_READ)
            noc_async_read(get_noc_addr_from_bank_id<true>(bank_id, in1_addr + off), w1, K_block * vcols * tile_bytes);
#else
            // TEST-ONLY diagnostic: drop ONLY the in1 DRAM read payload. The CB reserve/push, the rotated
            // shard order, the barrier, M-split forwarding, semaphores and all downstream compute/output work
            // are preserved; the block keeps its stale L1 contents, so the output is intentionally invalid.
            (void)off;
#endif
#if defined(IN1_ONE_PACKET)
            rearm();  // the coalesced read above is a plain noc_async_read and clobbers read_cmd_buf state
#endif
        } else {
            for (uint32_t kr = 0; kr < K_block; ++kr) {
                const uint32_t l = kblk * K_block + kr;  // capacity-local K index within the slice
                if (l < valid_k) {
                    const uint32_t gk = k_start + l;  // global logical K tile
                    const uint32_t off = (gk * in1_shard_stride_n + n_local + ncol_base) * tile_bytes;
#if !defined(SKIP_IN1_READ)
#if defined(IN1_ONE_PACKET)
                    // Uniform full-width row => reuse the preset size/config, supply only the offset.
                    if (one_packet_ok && vcols == N_block) {
                        noc_async_read_one_packet_with_state(bank_lo + in1_addr + off, w1);
                    } else {
                        noc_async_read(
                            get_noc_addr_from_bank_id<true>(bank_id, in1_addr + off), w1, vcols * tile_bytes);
                        rearm();  // that plain read clobbered read_cmd_buf state
                    }
#else
                    noc_async_read(get_noc_addr_from_bank_id<true>(bank_id, in1_addr + off), w1, vcols * tile_bytes);
#endif
#else
                    (void)off;  // diagnostic: drop the in1 DRAM read payload only (see above)
#endif
                    // cols [vcols, N_block) are pad-N (garbage): safe, those output cols aren't written.
                } else {
                    // K tail: summed into EVERY valid output col -> must be exactly 0.0 (both operands zeroed;
                    // writer also zeros in0's K/M tail), so the product is 0*0, never 0*NaN.
                    zero_l1(w1, N_block);
                }
                w1 += seg_bytes;
            }
        }
    };

#if defined(IN1_TRID_PIPELINE)
    // ---- TRID-PIPELINED read (solo/Sm==1 only; the M-split reader must have the data in hand before it can
    // forward, so it keeps the serial path below). The production path issues one block then takes a FULL
    // read barrier before pushing, so exactly ONE block is ever in flight and every block pays the whole DRAM
    // latency. Here each block's reads carry their own TRID, we run up to cb1_depth-1 blocks ahead, and wait
    // only on the OLDEST outstanding block before pushing it. Same reads, same order, same CB contents.
    //
    // Slot addressing: cb1 is exactly cb1_depth blocks, and we push exactly one block per retire, so block b
    // always lives at cb1_base + (b % cb1_depth) * in1_blk_bytes. cb1_base is captured before any push.
    // Before WRITING block b we must know the consumer has released its slot, which is what the
    // cb_reserve_back(ahead + 1) below guarantees (reserve counts free slots beyond the pushed pointer).
    // TRIDs are 1..cb1_depth (0 means "untagged"), so depth must be <= 15; the factory's cb1 depth is 4.
    if (mrole == 2u) {
        constexpr uint32_t D = (cb1_depth > 1u && cb1_depth <= 15u) ? cb1_depth : 1u;
        const uint32_t nblocks = N_bpc * G * W;
        cb_reserve_back(in1_cb, in1_blk);
        const uint32_t cb1_base = get_write_ptr(in1_cb);
        uint32_t issued = 0, pushed = 0;
        while (pushed < nblocks) {
            // issue as far ahead as the ring allows
            while (issued < nblocks && (issued - pushed) < D) {
                const uint32_t ahead = issued - pushed;
                cb_reserve_back(in1_cb, in1_blk * (ahead + 1u));  // consumer has released slot (issued % D)
                const uint32_t nb = issued / (G * W);
                const uint32_t rem = issued - nb * (G * W);
                const uint32_t step = rem / W;
                const uint32_t wb = rem - step * W;
                const uint32_t s = (ring_pos + G - step) % G;
                const uint32_t ncol_base = nb * N_block;
                const uint32_t vcols =
                    (ncol_base < valid_n) ? (((valid_n - ncol_base) < N_block) ? (valid_n - ncol_base) : N_block) : 0u;
                issue_block_reads(
                    s * W + wb, ncol_base, vcols, cb1_base + (issued % D) * in1_blk_bytes, (issued % D) + 1u);
                ++issued;
            }
            // retire the oldest outstanding block
            noc_async_read_barrier_with_trid((pushed % D) + 1u);
            cb_push_back(in1_cb, in1_blk);
            ++pushed;
        }
        noc_async_read_barrier();  // nothing should be outstanding, but leave the NoC quiescent
        return;
    }
#endif

    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
        const uint32_t ncol_base = nb * N_block;  // owned-column offset of this subblock
        // valid N columns within this subblock (0 => whole subblock is beyond the owned N range)
        [[maybe_unused]] const uint32_t vcols =
            (ncol_base < valid_n) ? (((valid_n - ncol_base) < N_block) ? (valid_n - ncol_base) : N_block) : 0u;
        for (uint32_t step = 0; step < G; ++step) {
            // Shard read order MUST match the in0 cb0 order. Ring: block `step` = shard (rp-step).
            const uint32_t s = (ring_pos + G - step) % G;
            for (uint32_t wb = 0; wb < W; ++wb) {
                const uint32_t kblk = s * W + wb;
                cb_reserve_back(in1_cb, in1_blk);
                const uint32_t w1 = get_write_ptr(in1_cb);
                // Serial policy: issue this block's reads (untagged), then wait for ALL of them before the
                // block is forwarded/pushed. Exactly one block is in flight.
                issue_block_reads(kblk, ncol_base, vcols, w1, 0u);
                if (vcols > 0u) {
                    noc_async_read_barrier();
                }
                mfwd(get_write_ptr(in1_cb));  // forward the fixed-size block
                cb_push_back(in1_cb, in1_blk);
            }
        }
    }
    // M-split READER exit drain: forwarded payloads are flushed per block, but the per-block `valid` semaphore
    // incs are non-posted atomics that were never drained -> the watcher flags pending NOC transactions at
    // kernel exit. Drain both writes and atomics once here. Guarded to the reader (mrole==1); the Sm==1 solo
    // path (mrole==2) has no forwarding and stays byte-identical.
    if (mrole == 1u) {
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
}
