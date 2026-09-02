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

    constexpr auto out_args = TensorAccessorArgs<18, 0>();
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

    Noc noc;
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
        constexpr uint32_t kAckLag = 2;  // the reader publishes PAIRS: it needs acks within lag 2 of its fetches
#if defined(VSA_PROBE) && VSA_PROBE == 7
        return;  // probe 7: no K fetches, no acks (the reader skips its kack waits too)
#endif
        const uint32_t k_base = head * k_head_stride;
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
            const uint32_t k_tile0 = k_base + block_id * k_tiles_per_block;
            for (uint32_t i = 0; i < k_tiles_per_block; ++i) {
                noc.async_read(
                    k, k_cb, k_tile_bytes, {.page_id = k_tile0 + i},
                    {.offset_bytes = (slot * k_tiles_per_block + i) * k_tile_bytes});
            }
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
        uint32_t sentinels_seen = 0;
        while (sentinels_seen < n_passes || drained < row_count) {
            while (cb_pages_available_at_front(cb_kreq, 1)) {
                kreq_cb.wait_front(1);
                uint32_t b0, s0, b1, s1;
                {
                    volatile tt_l1_ptr uint32_t* rq =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
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
                const uint32_t pass_rows = (row_count - pass_base < R_MAX) ? (row_count - pass_base) : R_MAX;
                for (uint32_t r = 0; r < pass_rows; ++r) {
                    const uint32_t ri = pass_base + r;  // chunk-cyclic (see reader)
                    const uint32_t q_tile = row_start + (ri >> VSA_ROW_CHUNK_LOG2) * row_stride +
                                            (ri & ((1u << VSA_ROW_CHUNK_LOG2) - 1));
                    const uint32_t page0 = (head * n_q_tiles + q_tile) * q_tiles_per_row;
                    for (uint32_t i = 0; i < q_tiles_per_row; ++i) {
                        // cb_q_res is RAM-mode: never reserved/pushed here, offsets from the base.
                        noc.async_read(
                            q, q_cb, q_tile_bytes, {.page_id = page0 + i},
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
                const uint32_t q_tile = row_start + (drained >> VSA_ROW_CHUNK_LOG2) * row_stride +
                                        (drained & ((1u << VSA_ROW_CHUNK_LOG2) - 1));
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

    uint32_t drained = 0;
    uint32_t pass_base = 0;

    // K pulls are tagged with per-half trid groups (half h -> trids 4h+1..4h+4). A window-end
    // marker kreq {0xFFFFFFFF, half} queues a LAZY ack: it is pushed, in marker order, once a
    // non-blocking check says that half's pulls landed -- never a blocking drain (the reader's
    // symmetric V-side blocking drain measured 60% of its wall time).
    uint32_t pull_idx[2] = {0, 0};       // pulls issued in the open window of each half
    uint32_t ack_pending[4];             // FIFO of marker halves awaiting their lazy ack
    uint32_t ack_head = 0, ack_tail = 0;
    const auto khalf_landed = [&](uint32_t h) {
        for (uint32_t t = 0; t < 4; ++t) {
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc.get_noc_id(), h * 4 + 1 + t)) {
                return false;
            }
        }
        return true;
    };
    auto serve_kreq_if_any = [&]() {
        while (ack_head != ack_tail && khalf_landed(ack_pending[ack_head & 3])) {
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
            ++ack_head;
        }
        while (cb_pages_available_at_front(cb_kreq, 1)) {
            kreq_cb.wait_front(1);
            uint32_t leader_slot, slot, khalf;
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                leader_slot = rq[0];
                slot = rq[1];
                khalf = rq[2];
            }
            kreq_cb.pop_front(1);
            if (leader_slot == 0xFFFFFFFFu) {  // window end: queue the lazy ack
                ack_pending[ack_tail & 3] = khalf;
                ++ack_tail;
                pull_idx[khalf] = 0;
                continue;
            }
            const uint32_t trid = khalf * 4 + 1 + (pull_idx[khalf] & 3);
            if (pull_idx[khalf] >= 4) {
                experimental::async_read_barrier_with_trid(noc, trid);  // reuse within this window
            }
            ++pull_idx[khalf];
            experimental::set_read_trid(noc, trid);
            noc_async_read(
                get_noc_addr(
                    leader_x, leader_y, k_l1_base + leader_slot * k_tiles_per_block * k_tile_bytes,
                    noc.get_noc_id()),
                k_l1_base + slot * k_tiles_per_block * k_tile_bytes, k_tiles_per_block * k_tile_bytes,
                noc.get_noc_id());
            experimental::set_read_trid(noc, 0);
        }
    };

    while (pass_base < row_count || drained < row_count) {
        if (pass_base < row_count && drained >= pass_base) {
            const uint32_t pass_rows = (row_count - pass_base < R_MAX) ? (row_count - pass_base) : R_MAX;
            for (uint32_t r = 0; r < pass_rows; ++r) {
                const uint32_t ri = pass_base + r;  // chunk-cyclic (see reader)
                const uint32_t q_tile = row_start + (ri >> VSA_ROW_CHUNK_LOG2) * row_stride + (ri & ((1u << VSA_ROW_CHUNK_LOG2) - 1));
                const uint32_t page0 = (head * n_q_tiles + q_tile) * q_tiles_per_row;
                for (uint32_t i = 0; i < q_tiles_per_row; ++i) {
                    // cb_q_res is RAM-mode: never reserved/pushed here, offsets from the base.
                    noc.async_read(
                        q, q_cb, q_tile_bytes, {.page_id = page0 + i},
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

        if (cb_pages_available_at_front(cb_out, out_tiles_per_row)) {
            out_cb.wait_front(out_tiles_per_row);
            const uint32_t q_tile = row_start + (drained >> 2) * row_stride + (drained & 3);
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
