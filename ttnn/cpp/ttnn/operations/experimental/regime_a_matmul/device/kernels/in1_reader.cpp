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

// Test-only causal timing zones (compile-gated; mask 0 => no-op => byte-identical). See DIAG_ZONES.
#ifdef DIAG_ZONES
#define RA_ZONE(n) DeviceZoneScopedN(n)
#else
#define RA_ZONE(n)
#endif

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
            noc_async_write(w1, get_noc_addr(sx, sy, w1), in1_blk_bytes);
        }
#ifdef DIAG_FWD_FLUSH_FIRST
        noc_async_writes_flushed();  // A/B baseline: OLD flush-before-signal
#endif
        for (uint32_t s = 0; s < mpeers; ++s) {
            uint32_t sx = get_arg_val<uint32_t>(9 + s * 2), sy = get_arg_val<uint32_t>(10 + s * 2);
            noc_semaphore_inc(get_noc_addr(sx, sy, in1valid_addr), 1);
        }
#ifndef DIAG_FWD_FLUSH_FIRST
        // DEFAULT: signal EARLY, then flush PER-BLOCK. The early valid-inc releases the slave without
        // waiting on the reader's flush (same-NoC write-before-inc keeps the destination from observing
        // validity before the payload lands); the per-block flush that follows is REQUIRED for SOURCE
        // lifetime -- it guarantees the async write has departed this CB slot before the slot is pushed,
        // wrapped, and overwritten by a later block (an exit-only barrier would be too late). The flush is
        // merely moved off the slave-release critical path, NOT removed. DIAG_FWD_FLUSH_FIRST = old order.
        noc_async_writes_flushed();
#endif
        ++mbc;
    };

    // ---- Strided sub-band read in rotated shard order, with balanced tails. ----
    // NO local zero-fill here: invalid positions are simply NOT read (left as garbage). Correctness is
    // preserved by the writer, which zeros in0 for K/M tails -> 0*garbage == 0 kills the K-tail term; and
    // pad-N columns (>= valid_n) are never written to the output. This keeps the reader on its fast path
    // and confines the (cheap) tail zeroing to the small in0 buffer in the writer.
    RA_ZONE("Z_IN1READ");  // function-scope: the whole strided in1 read+forward loop (BRISC critical path)
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
                uint32_t w1 = get_write_ptr(in1_cb);
                if (vcols > 0u) {
#ifndef DIAG_NO_COALESCE
                    // DEFAULT (adopted): coalesce the whole [K_block x vcols] block into ONE read when it is
                    // physically contiguous in the bank shard: full owned width (vcols==N_block==shard
                    // stride), zero column offset, and NO K-tail in this block (all K_block rows valid).
                    // Consecutive K rows are then adjacent (gk*stride) with a contiguous L1 destination
                    // (seg_bytes == vcols*tile_bytes), so one read replaces K_block per-row reads (-0.5..-3.1%,
                    // PCC-exact). Falls back to per-row otherwise. DIAG_NO_COALESCE forces per-row for A/B.
                    const bool contig = (vcols == N_block) && (N_block == in1_shard_stride_n) &&
                                        ((n_local + ncol_base) == 0u) && ((kblk * K_block + K_block) <= valid_k);
                    if (contig) {
                        const uint32_t off = (k_start + kblk * K_block) * in1_shard_stride_n * tile_bytes;
                        noc_async_read(
                            get_noc_addr_from_bank_id<true>(bank_id, in1_addr + off), w1, K_block * vcols * tile_bytes);
                        noc_async_read_barrier();
                    } else
#endif
                    {
                        for (uint32_t kr = 0; kr < K_block; ++kr) {
                            const uint32_t l = kblk * K_block + kr;  // capacity-local K index within the slice
                            if (l < valid_k) {
                                const uint32_t gk = k_start + l;  // global logical K tile
                                const uint32_t off = (gk * in1_shard_stride_n + n_local + ncol_base) * tile_bytes;
                                noc_async_read(
                                    get_noc_addr_from_bank_id<true>(bank_id, in1_addr + off), w1, vcols * tile_bytes);
                                // cols [vcols, N_block) are pad-N (garbage): safe, those output cols aren't written.
                            } else {
                                // K tail: summed into EVERY valid output col -> must be exactly 0.0 (both operands
                                // zeroed; writer also zeros in0's K/M tail), so the product is 0*0, never 0*NaN.
                                zero_l1(w1, N_block);
                            }
                            w1 += seg_bytes;
                        }
                        noc_async_read_barrier();
                    }
                }
                // vcols == 0 (whole subblock is pad N): no reads; block is garbage, output not written.
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
