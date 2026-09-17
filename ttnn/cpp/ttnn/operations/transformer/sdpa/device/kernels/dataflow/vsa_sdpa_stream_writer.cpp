// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming (v3) writer.
// LEADER: serves the leader reader's kreq by fetching each block's K tiles from DRAM into the
// shared stream slots (the second NoC of the fetch pair). Nothing else.
// WORKER: builds the persistent compute tiles, loads each pass's resident Q rows, serves the local
// reader's kreq by pulling K from the leader's L1 slot, and drains normalized row outputs to DRAM.
// K-serving and output-draining are polled in one loop (blocking either side alone would deadlock).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_msa_gather.hpp"
#include "dataflow_common.hpp"
#include "vsa_sum_service.hpp"
#include "vsa_decouple.hpp"
#include "api/debug/dprint.h"

constexpr uint32_t one_bf16_packed = 0x3F803F80u;

void kernel_main() {
    constexpr uint32_t n_q_tiles = get_compile_time_arg_val(0);  // S / 64 per head
    constexpr uint32_t R_MAX = get_compile_time_arg_val(1);
    constexpr uint32_t q_tiles_per_row = get_compile_time_arg_val(2);    // Sqt * DHt
    constexpr uint32_t out_tiles_per_row = get_compile_time_arg_val(3);  // Sqt * vDHt
    constexpr uint32_t k_tiles_per_block = get_compile_time_arg_val(4);
    constexpr uint32_t k_head_stride = get_compile_time_arg_val(5);
    constexpr uint32_t q_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(8);

    constexpr uint32_t cb_q_res = get_compile_time_arg_val(9);
    constexpr uint32_t cb_k_stream = get_compile_time_arg_val(10);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(11);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(12);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(13);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(14);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(15);
    constexpr uint32_t cb_qdone = get_compile_time_arg_val(16);
    constexpr uint32_t cb_out = get_compile_time_arg_val(17);
    constexpr uint32_t cb_shdr = get_compile_time_arg_val(18);  // row-sum service (vsa_sum_service.hpp)
    constexpr uint32_t cb_stiles = get_compile_time_arg_val(19);
    constexpr uint32_t cb_sumback = get_compile_time_arg_val(20);
    constexpr uint32_t cb_sacc = get_compile_time_arg_val(21);
    constexpr uint32_t Sqt = get_compile_time_arg_val(22);
    constexpr uint32_t stream_depth = get_compile_time_arg_val(23);  // decoupled K pulls (vsa_decouple.hpp)
    constexpr uint32_t cb_log = get_compile_time_arg_val(24);
    constexpr uint32_t log_depth = get_compile_time_arg_val(25);

    constexpr auto out_args = TensorAccessorArgs<26, 0>();
    constexpr auto k_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
    constexpr auto q_args =
        TensorAccessorArgs<k_args.next_compile_time_args_offset(), k_args.next_common_runtime_args_offset()>();

    uint32_t argi = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t q_addr = get_arg_val<uint32_t>(argi++);
    const uint32_t head = get_arg_val<uint32_t>(argi++);
    const uint32_t is_leader = get_arg_val<uint32_t>(argi++);
    const uint32_t n_passes = get_arg_val<uint32_t>(argi++);
    const uint32_t leader_x = get_arg_val<uint32_t>(argi++);
    const uint32_t leader_y = get_arg_val<uint32_t>(argi++);
    const uint32_t row_start = get_arg_val<uint32_t>(argi++);
    const uint32_t row_stride = get_arg_val<uint32_t>(argi++);
    const uint32_t row_count = get_arg_val<uint32_t>(argi++);
    constexpr uint32_t kPassArg = 12;         // pass_rows[n_passes] ...
    const uint32_t kRowsArg = 12 + n_passes;  // ... then the row list (q tiles, pass-major)
    (void)row_start;
    (void)row_stride;
    // after the rows: the leader stream ring depth and the CB regions its extra slots are carved from (v2)
    const uint32_t ldepth = get_arg_val<uint32_t>(kRowsArg + row_count);
    const uint32_t n_regions = get_arg_val<uint32_t>(kRowsArg + row_count + 1);
    const uint32_t region_argi = kRowsArg + row_count + 2;
    const uint32_t n_group_heads = get_arg_val<uint32_t>(region_argi + 2 * n_regions);
    const uint32_t head0 = get_arg_val<uint32_t>(region_argi + 2 * n_regions + 1);
    (void)n_group_heads;
    constexpr uint32_t kMaxLeaderSlots = 40;
    constexpr uint32_t k_block_bytes = k_tiles_per_block * k_tile_bytes;

    Noc noc;
    vsa_sum::Service sums{cb_shdr, cb_stiles, cb_sumback, get_write_ptr(cb_sacc), R_MAX, Sqt};
#define VSA_SERVE() sums.serve()
#if defined(VSA_NO_SUMS)
#undef VSA_SERVE
#define VSA_SERVE() ((void)0)
#endif
    experimental::CB q_cb(cb_q_res), k_cb(cb_k_stream), kreq_cb(cb_kreq), kack_cb(cb_kack);
    experimental::CB qdone_cb(cb_qdone), out_cb(cb_out);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto q = TensorAccessor(q_args, q_addr);

#ifdef VSA_IS_LEADER
    if (is_leader) {
        // Fetch K for every kreq {block_id, slot} from DRAM into the shared stream slots;
        // a sentinel kreq ends each pass. Fetches are pipelined kAckLag blocks deep with
        // per-block trids: block N's tile reads are tagged (N % 8) + 1, and its ack costs one
        // trid barrier (already landed with the pipeline full) instead of a DRAM round trip.
        // kAckLag stays BELOW the reader's fetch lag: the reader waits for block N's ack after
        // sending kreqs N..N+3 only, so an ack gated on kreq N+4 would deadlock.
        constexpr uint32_t kAckLag =
            vsa_dec::kFetchLag - 2;  // the reader publishes PAIRS: acks within lag 2 of its fetches
        static_assert(kAckLag + 1 <= 8, "K blocks in flight need one trid each");
#if defined(VSA_PROBE) && VSA_PROBE == 7
        return;  // probe 7: no K fetches, no acks (the reader skips its kack waits too)
#endif
        const uint32_t k_base = head * k_head_stride;
#ifdef VSA_RING
        // Ring mode: own-shard blocks come from the local K tensor, remote shards from the gathered K buffer (both
        // head-split [1, H, T, d]; see the reader's leader for the protocol). Ring constants are COMMON runtime args
        // after the accessor common args (vsa_sdpa_stream_descriptor.hpp kRingCommonArg*).
        constexpr auto gk_args =
            TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
        constexpr uint32_t ring_crt = gk_args.next_common_runtime_args_offset();
        const uint32_t gk_addr = get_common_arg_val<uint32_t>(ring_crt + 0);
        const uint32_t ring_index = get_common_arg_val<uint32_t>(ring_crt + 1);
        const uint32_t blocks_per_shard = get_common_arg_val<uint32_t>(ring_crt + 2);
        const uint32_t dht = get_common_arg_val<uint32_t>(ring_crt + 4);
        const uint32_t ht_local = get_common_arg_val<uint32_t>(ring_crt + 13);
        const uint32_t ht_total = get_common_arg_val<uint32_t>(ring_crt + 14);
        const auto gk = TensorAccessor(gk_args, gk_addr);
        const uint32_t k_local_base = head * ht_local * dht;  // K of this head: local tensor
        const uint32_t k_gath_base = head * ht_total * dht;   // and gathered buffer
        (void)k_base;
#endif
        // slot -> K address, the same table as the reader leader's: slots [0, stream_depth) in the stream CB, the
        // rest carved from the given regions as {K block, V block} pairs in order (the V block size equals K's here)
        uint32_t slot_k[kMaxLeaderSlots];
        {
            const uint32_t kb0 = k_cb.get_write_ptr();
            for (uint32_t s = 0; s < stream_depth; ++s) {
                slot_k[s] = kb0 + s * k_block_bytes;
            }
            uint32_t ns = stream_depth;
            constexpr uint32_t pair_bytes = 2 * k_block_bytes;
            for (uint32_t r = 0; r < n_regions; ++r) {
                const uint32_t cbid = get_arg_val<uint32_t>(region_argi + 2 * r);
                const uint32_t bytes = get_arg_val<uint32_t>(region_argi + 2 * r + 1);
                const uint32_t base = get_write_ptr(cbid);
                for (uint32_t off = 0; off + pair_bytes <= bytes && ns < ldepth; ++ns) {
                    slot_k[ns] = base + off;
                    off += pair_bytes;
                }
            }
            if (ns != ldepth || ldepth > kMaxLeaderSlots) {
                for (;;) {  // host/kernel carve mismatch
                    invalidate_l1_cache();
                }
            }
        }
        uint32_t nfetch = 0, nacked = 0;
        const auto ack_oldest = [&]() {
            experimental::async_read_barrier_with_trid(noc, (nacked % 8) + 1);
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
            ++nacked;
        };
        constexpr uint32_t kNoBlock = 0xFFFFFFFEu;
        const auto fetch_one = [&](uint32_t block_id, uint32_t slot) {
            experimental::set_read_trid(noc, (nfetch % 8) + 1);
            const uint32_t dst = slot_k[slot];
#ifdef VSA_RING
            // head-split layout: a block's K tiles are contiguous pages of this head; own shard from the local
            // tensor (local block id), every other shard from the gathered buffer (global block id)
            if (block_id / blocks_per_shard == ring_index) {
                const uint32_t k_tile0 = k_local_base + (block_id - ring_index * blocks_per_shard) * k_tiles_per_block;
                for (uint32_t i = 0; i < k_tiles_per_block; ++i) {
                    noc_async_read(k.get_noc_addr(k_tile0 + i), dst + i * k_tile_bytes, k_tile_bytes, noc.get_noc_id());
                }
            } else {
                const uint32_t k_tile0 = k_gath_base + block_id * k_tiles_per_block;
                for (uint32_t i = 0; i < k_tiles_per_block; ++i) {
                    noc_async_read(
                        gk.get_noc_addr(k_tile0 + i), dst + i * k_tile_bytes, k_tile_bytes, noc.get_noc_id());
                }
            }
#else
            // block_id: block | head_sel << 16 (the leader interleaves the group's heads)
            const uint32_t k_tile0 =
                (head0 + (block_id >> 16)) * k_head_stride + (block_id & 0xFFFFu) * k_tiles_per_block;
            (void)k_base;
#if defined(VSA_PROBE) && VSA_PROBE == 12
            // layout probe: one contiguous read per block (wrong bytes; see the reader's leader)
            noc_async_read(k.get_noc_addr(k_tile0), dst, k_block_bytes, noc.get_noc_id());
#else
            for (uint32_t i = 0; i < k_tiles_per_block; ++i) {
                noc_async_read(k.get_noc_addr(k_tile0 + i), dst + i * k_tile_bytes, k_tile_bytes, noc.get_noc_id());
            }
#endif
#endif
            experimental::set_read_trid(noc, 0);
            ++nfetch;
            if (nfetch - nacked > kAckLag) {
                ack_oldest();
            }
        };
        // Leader-as-worker: this core also holds resident rows, so the K service is one arm of a
        // polled loop that additionally loads each pass's Q rows and drains its own row outputs.
        if (row_count > 0) {
            dataflow_kernel_lib::
                calculate_and_prepare_reduce_scaler<cb_scale, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
            generate_bcast_col_scalar(experimental::CB(cb_col_identity), one_bf16_packed);
            constexpr uint32_t mask_tile_bytes = get_tile_size(cb_neginf);
            experimental::CB(cb_neginf).reserve_back(1);
            fill_neginf_tile<mask_tile_bytes>(cb_neginf, 0);
            experimental::CB(cb_neginf).push_back(1);
        }
        uint32_t drained = 0;
        uint32_t pass_base = 0;
        uint32_t pass_i = 0;
        uint32_t sentinels_seen = 0;
        while (sentinels_seen < n_passes || drained < row_count) {
            if (row_count > 0) {
                VSA_SERVE();  // one 16-row slice at most: bounded delay for the K service below
            }
            while (cb_pages_available_at_front(cb_kreq, 1)) {
                kreq_cb.wait_front(1);
                uint32_t b0, s0, b1, s1;
                {
                    volatile tt_l1_ptr uint32_t* rq =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                    invalidate_l1_cache();  // page written by NCRISC: bypass this RISC's stale L1 read cache
                    b0 = rq[0];
                    s0 = rq[1];
                    b1 = rq[2];
                    s1 = rq[3];
                }
                kreq_cb.pop_front(1);
                if (b0 == 0xFFFFFFFFu) {
                    while (nacked < nfetch) {
                        ack_oldest();
                    }
                    ++sentinels_seen;
                    continue;
                }
                fetch_one(b0, s0);
                if (b1 != kNoBlock) {
                    fetch_one(b1, s1);
                }
            }
            if (row_count > 0 && pass_base < row_count && drained >= pass_base) {
                const uint32_t pass_rows = get_arg_val<uint32_t>(kPassArg + pass_i++);
                for (uint32_t r = 0; r < pass_rows; ++r) {
                    const uint32_t q_tile = get_arg_val<uint32_t>(kRowsArg + pass_base + r);
                    const uint32_t page0 = (head * n_q_tiles + q_tile) * q_tiles_per_row;
                    for (uint32_t i = 0; i < q_tiles_per_row; ++i) {
                        // cb_q_res is RAM-mode: never reserved/pushed here, offsets from the base.
                        noc.async_read(
                            q,
                            q_cb,
                            q_tile_bytes,
                            {.page_id = page0 + i},
                            {.offset_bytes = (r * q_tiles_per_row + i) * q_tile_bytes});
                    }
                }
                noc.async_read_barrier();
                qdone_cb.reserve_back(1);
                qdone_cb.push_back(1);
                pass_base += pass_rows;
            }
            if (row_count > 0 && cb_pages_available_at_front(cb_out, out_tiles_per_row)) {
                out_cb.wait_front(out_tiles_per_row);
                const uint32_t q_tile = get_arg_val<uint32_t>(kRowsArg + drained);
                const uint32_t page0 = (head * n_q_tiles + q_tile) * out_tiles_per_row;
                for (uint32_t i = 0; i < out_tiles_per_row; ++i) {
                    noc.async_write(
                        out_cb, out, out_tile_bytes, {.offset_bytes = i * out_tile_bytes}, {.page_id = page0 + i});
                }
                noc.async_write_barrier();
                out_cb.pop_front(out_tiles_per_row);
                ++drained;
            }
        }
        return;
    }
#else

    // ---------------- WORKER ----------------
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scale, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(experimental::CB(cb_col_identity), one_bf16_packed);
    {
        constexpr uint32_t mask_tile_bytes = get_tile_size(cb_neginf);
        experimental::CB(cb_neginf).reserve_back(1);
        fill_neginf_tile<mask_tile_bytes>(cb_neginf, 0);
        experimental::CB(cb_neginf).push_back(1);
    }

    const uint32_t k_l1_base = k_cb.get_write_ptr();
    const uint32_t k_base_w = head * k_head_stride;  // this head's K pages (DRAM fallback of the decoupled protocol)

    uint32_t drained = 0;
    uint32_t pass_base = 0;
    uint32_t pass_i = 0;

    // K pulls are tagged with per-half trid groups (half h -> trids 4h+1..4h+4). A window-end
    // marker kreq {0xFFFFFFFF, half} queues a LAZY ack: it is pushed, in marker order, once a
    // non-blocking check says that half's pulls landed -- never a blocking drain (the reader's
    // symmetric V-side blocking drain measured 60% of its wall time).
    uint32_t pull_idx[2] = {0, 0};  // pulls issued in the open window of each half
    uint32_t ack_pending[4];        // FIFO of marker halves awaiting their lazy ack
    uint32_t ack_head = 0, ack_tail = 0;
    // Decoupled protocol (vsa_decouple.hpp): per-window record of the window's pulls {arrival, slot, from DRAM},
    // validated before the window's ack (a slot the leader may have refilled since is re-read from DRAM).
    constexpr uint32_t half_slots = stream_depth / 2;
#ifdef VSA_FREE_SLOTS
    constexpr uint32_t kRecs = 8;  // window records in flight (the reader's kMaxWin); K trid = 1 + rec
#else
    constexpr uint32_t kRecs = 2;  // the two halves
#endif
    struct KPull {
        uint32_t n, slot, dram, b;
    };
    KPull kp[kRecs][half_slots > 0 ? half_slots : 1];
    uint32_t kp_n[kRecs] = {};
    uint32_t pull_idx_rec[kRecs] = {};
    uint32_t ack_pending_rec[kRecs];
    (void)pull_idx_rec;
    (void)ack_pending_rec;
    vsa_dec::View dec{get_write_ptr(cb_log), log_depth, 1, 0, ldepth};
    const auto block_of_arrival = [&](uint32_t n) -> uint32_t {  // from the (published) log entry
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                   dec.log_l1 + (dec.log_of_arrival(n) % log_depth) * vsa_dec::kEntryWords * 4) &
               0xFFFFu;  // block | head_sel << 16
    };
    const auto read_k_from_dram = [&](uint32_t b, uint32_t slot) {
        for (uint32_t i = 0; i < k_tiles_per_block; ++i) {
            noc.async_read(
                k,
                k_cb,
                k_tile_bytes,
                {.page_id = k_base_w + b * k_tiles_per_block + i},
                {.offset_bytes = (slot * k_tiles_per_block + i) * k_tile_bytes});
        }
    };
    const auto khalf_landed = [&](uint32_t h) {
        for (uint32_t t = 0; t < 4; ++t) {
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc.get_noc_id(), h * 4 + 1 + t)) {
                return false;
            }
        }
        return true;
    };
#ifdef VSA_FREE_SLOTS
    // Free-slot pipeline: one K trid per window record (1 + rec), acks in marker (= record) order.
    auto serve_kreq_if_any = [&]() {
        while (ack_head != ack_tail &&
               ncrisc_noc_read_with_transaction_id_flushed(noc.get_noc_id(), 1 + ack_pending_rec[ack_head % kRecs])) {
            const uint32_t rec = ack_pending_rec[ack_head % kRecs];
#ifdef VSA_DECOUPLE
            bool fixed = false;
            for (uint32_t i = 0; i < kp_n[rec]; ++i) {
                if (kp[rec][i].dram == 0 && dec.slot_unsafe(kp[rec][i].n)) {
                    read_k_from_dram(kp[rec][i].b, kp[rec][i].slot);
                    fixed = true;
                }
            }
            if (fixed) {
                noc.async_read_barrier();
            }
#endif
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
            ++ack_head;
        }
        while (cb_pages_available_at_front(cb_kreq, 1)) {
            kreq_cb.wait_front(1);
            uint32_t leader_slot, w1, w2, w3;
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                invalidate_l1_cache();
                leader_slot = rq[0];
                w1 = rq[1];
                w2 = rq[2];
                w3 = rq[3];
            }
            kreq_cb.pop_front(1);
            if (leader_slot == 0xFFFFFFFFu) {  // window end {0xFFFFFFFF, 0, rec}
                const uint32_t rec = w2;
                ASSERT(rec < kRecs);
                ack_pending_rec[ack_tail % kRecs] = rec;
                ++ack_tail;
                kp_n[rec] = pull_idx_rec[rec];
                pull_idx_rec[rec] = 0;
                continue;
            }
            // pull {leader_slot, slot | rec << 8 | from_dram << 12, arrival, pass_len}
            const uint32_t slot = w1 & 0xffu;
            const uint32_t rec = (w1 >> 8) & 0xfu;
            const bool from_dram = ((w1 >> 12) & 1u) != 0;
            const uint32_t n = w2;
            dec.pass_len = w3 ? w3 : 1;
            dec.total = n_passes * dec.pass_len;
            const uint32_t b = block_of_arrival(n);  // the entry is still valid: the reader just read it
            if (pull_idx_rec[rec] < half_slots) {
                kp[rec][pull_idx_rec[rec]] = {n, slot, from_dram ? 1u : 0u, b};
            }
            ++pull_idx_rec[rec];
            experimental::set_read_trid(noc, 1 + rec);
            if (from_dram) {
                read_k_from_dram(b, slot);
            } else {
                noc_async_read(
                    get_noc_addr(leader_x, leader_y, leader_slot /* the leader's K address */, noc.get_noc_id()),
                    k_l1_base + slot * k_tiles_per_block * k_tile_bytes,
                    k_tiles_per_block * k_tile_bytes,
                    noc.get_noc_id());
            }
            experimental::set_read_trid(noc, 0);
        }
    };
#else
    auto serve_kreq_if_any = [&]() {
        while (ack_head != ack_tail && khalf_landed(ack_pending[ack_head & 3])) {
            const uint32_t h = ack_pending[ack_head & 3];
#ifdef VSA_DECOUPLE
            bool fixed = false;
            for (uint32_t i = 0; i < kp_n[h]; ++i) {
                if (kp[h][i].dram == 0 && dec.slot_unsafe(kp[h][i].n)) {
                    read_k_from_dram(kp[h][i].b, kp[h][i].slot);
                    fixed = true;
                }
            }
            if (fixed) {
                noc.async_read_barrier();
            }
#else
            (void)h;
#endif
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
            ++ack_head;
        }
        while (cb_pages_available_at_front(cb_kreq, 1)) {
            kreq_cb.wait_front(1);
            uint32_t leader_slot, w1, w2, w3;
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                invalidate_l1_cache();  // page written by NCRISC: bypass this RISC's stale L1 read cache
                leader_slot = rq[0];
                w1 = rq[1];
                w2 = rq[2];
                w3 = rq[3];
            }
            kreq_cb.pop_front(1);
            if (leader_slot == 0xFFFFFFFFu) {  // window end {0xFFFFFFFF, 0, half}: queue the lazy ack
                const uint32_t khalf = w2;
                ASSERT(khalf < 2);  // indexes pull_idx[] and the per-half trid groups
                ack_pending[ack_tail & 3] = khalf;
                ++ack_tail;
                kp_n[khalf] = pull_idx[khalf];
                pull_idx[khalf] = 0;
                continue;
            }
            // pull {leader_slot, slot | half << 8 | from_dram << 9, arrival, pass_len}
            const uint32_t slot = w1 & 0xffu;
            const uint32_t khalf = (w1 >> 8) & 1u;
            const bool from_dram = ((w1 >> 9) & 1u) != 0;
            const uint32_t n = w2;
            dec.pass_len = w3 ? w3 : 1;
            dec.total = n_passes * dec.pass_len;
            const uint32_t trid = khalf * 4 + 1 + (pull_idx[khalf] & 3);
            if (pull_idx[khalf] >= 4) {
                experimental::async_read_barrier_with_trid(noc, trid);  // reuse within this window
            }
            const uint32_t b = block_of_arrival(n);
            if (pull_idx[khalf] < half_slots) {
                kp[khalf][pull_idx[khalf]] = {n, slot, from_dram ? 1u : 0u, b};
            }
            ++pull_idx[khalf];
            experimental::set_read_trid(noc, trid);
            if (from_dram) {
                read_k_from_dram(b, slot);
            } else {
                noc_async_read(
                    get_noc_addr(leader_x, leader_y, leader_slot /* the leader's K address */, noc.get_noc_id()),
                    k_l1_base + slot * k_tiles_per_block * k_tile_bytes,
                    k_tiles_per_block * k_tile_bytes,
                    noc.get_noc_id());
            }
            experimental::set_read_trid(noc, 0);
        }
    };
#endif  // VSA_FREE_SLOTS

    while (pass_base < row_count || drained < row_count) {
        if (pass_base < row_count && drained >= pass_base) {
            const uint32_t pass_rows = get_arg_val<uint32_t>(kPassArg + pass_i++);
            for (uint32_t r = 0; r < pass_rows; ++r) {
                const uint32_t q_tile = get_arg_val<uint32_t>(kRowsArg + pass_base + r);
                const uint32_t page0 = (head * n_q_tiles + q_tile) * q_tiles_per_row;
                for (uint32_t i = 0; i < q_tiles_per_row; ++i) {
                    // cb_q_res is RAM-mode: never reserved/pushed here, offsets from the base.
                    noc.async_read(
                        q,
                        q_cb,
                        q_tile_bytes,
                        {.page_id = page0 + i},
                        {.offset_bytes = (r * q_tiles_per_row + i) * q_tile_bytes});
                }
                serve_kreq_if_any();
            }
            noc.async_read_barrier();
            qdone_cb.reserve_back(1);
            qdone_cb.push_back(1);
            pass_base += pass_rows;
        }

        serve_kreq_if_any();
        VSA_SERVE();  // one 16-row slice at most: bounded delay for the K service

        if (cb_pages_available_at_front(cb_out, out_tiles_per_row)) {
            out_cb.wait_front(out_tiles_per_row);
            const uint32_t q_tile = get_arg_val<uint32_t>(kRowsArg + drained);
            const uint32_t page0 = (head * n_q_tiles + q_tile) * out_tiles_per_row;
            for (uint32_t i = 0; i < out_tiles_per_row; ++i) {
                noc.async_write(
                    out_cb, out, out_tile_bytes, {.offset_bytes = i * out_tile_bytes}, {.page_id = page0 + i});
            }
            noc.async_write_barrier();
            out_cb.pop_front(out_tiles_per_row);
            ++drained;
        }
    }
#endif  // VSA_IS_LEADER
}
