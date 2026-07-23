// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash-attention reader.
//
// Per Q-block (batch, query-head, Q-chunk) this core owns:
//   * read the Q chunk once, natural (Sq_chunk_t, DHt) tile grid  -> cb_q_in
//     (held resident by compute across the whole KV loop).
//   * for each KV-block: read K TRANSPOSED at the tile-grid level so cb_k_in
//     is (DHt, Sk_chunk_t) — matmul_block<transpose=true> then flips each tile
//     internally, giving Q·Kᵀ. V read natural (Sk_chunk_t, DHt). Mask (custom)
//     read as a full (Sq_chunk_t, Sk_chunk_t) block.
// The reduce scalers (value 1.0, pool-type-aware fills for MAX and SUM
// REDUCE_ROW) are pushed once at startup; compute waits but never pops them.
//
// GQA/MQA is reader addressing only: kv_head = nq / (H_q / H_kv). No K/V
// duplication.
//
// Refinement 3 — K/V reuse multicast (compile-time USE_MCAST regime). When the
// (batch,head) groups map one-per-grid-row (B·H_q == grid rows) and there is no
// mask, K/V are reuse-shared across every core owning a Q-block of the same
// (batch,head): instead of each core re-pulling ~2.4 MB K+V from DRAM, ONE
// injector per row (col 0) reads each KV-block once and NoC-multicasts it across
// its row via the mcast_pipe SenderPipe/ReceiverPipe helper + ttnn.Mcast1D
// PerRow host wiring. All cores in a row process `rounds = ceil(q_num_chunks/GC)`
// Q-blocks in perfect lockstep on cb_k_in/cb_v_in (dummy slots re-run Q-chunk 0,
// a benign bit-identical redundant output), so the mcast landing address stays
// identical across the row. Q is per-core (varies along S_q → not shared).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t H_Q = get_compile_time_arg_val(1);
    constexpr uint32_t H_KV = get_compile_time_arg_val(2);
    constexpr uint32_t SQT = get_compile_time_arg_val(3);
    constexpr uint32_t SKVT = get_compile_time_arg_val(4);
    constexpr uint32_t DHT = get_compile_time_arg_val(5);
    constexpr uint32_t SQ_CHUNK_T = get_compile_time_arg_val(6);
    constexpr uint32_t SK_CHUNK_T = get_compile_time_arg_val(7);
    constexpr uint32_t Q_NUM_CHUNKS = get_compile_time_arg_val(8);
    constexpr uint32_t K_NUM_CHUNKS = get_compile_time_arg_val(9);
    constexpr uint32_t HAS_MASK = get_compile_time_arg_val(10);
    constexpr uint32_t MASK_BCAST = get_compile_time_arg_val(11);
    constexpr uint32_t USE_MCAST = get_compile_time_arg_val(12);
    constexpr uint32_t GC = get_compile_time_arg_val(13);

    // Mcast CT block occupies [14..18]; TensorAccessorArgs chain after it.
    constexpr auto q_args = TensorAccessorArgs<19>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto m_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t m_addr = get_arg_val<uint32_t>(3);
    const uint32_t q_start = get_arg_val<uint32_t>(4);
    const uint32_t q_count = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_q_in = 0;
    constexpr uint32_t cb_k_in = 1;
    constexpr uint32_t cb_v_in = 2;
    constexpr uint32_t cb_mask_in = 3;
    constexpr uint32_t cb_scaler_max = 4;
    constexpr uint32_t cb_scaler_sum = 5;

    // Scalers (value 1.0) — pool-type-aware overloads (MAX vs SUM REDUCE_ROW
    // use different fills). Pushed once; reduce<> never pops them.
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);

    const uint32_t tile_bytes = get_tile_size(cb_q_in);
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);
    [[maybe_unused]] const auto m_acc = TensorAccessor(m_args, m_addr, tile_bytes);

    constexpr uint32_t q_heads_per_kv = H_Q / H_KV;
    constexpr uint32_t MASK_H = MASK_BCAST ? 1u : H_Q;
    constexpr uint32_t q_block_tiles = SQ_CHUNK_T * DHT;
    constexpr uint32_t k_block_tiles = SK_CHUNK_T * DHT;
    constexpr uint32_t mask_block_tiles = SQ_CHUNK_T * SK_CHUNK_T;

    // ---- Refinement 3: K/V reuse-multicast regime (no mask; one head per row) ----
    if constexpr (USE_MCAST) {
        // Mcast CT block at 14; per-core mcast RT (rect|coords) at RT 10.
        constexpr auto mc = McastArgs</*CT=*/14, /*RT=*/10>();
        const uint32_t row_y = get_arg_val<uint32_t>(6);
        const uint32_t col_x = get_arg_val<uint32_t>(7);
        const uint32_t rounds = get_arg_val<uint32_t>(8);
        const uint32_t is_sender = get_arg_val<uint32_t>(9);

        const uint32_t nb = row_y / H_Q;
        const uint32_t nq = row_y % H_Q;
        const uint32_t kv_head = nq / q_heads_per_kv;
        const uint32_t q_base = (nb * H_Q + nq) * SQT;
        const uint32_t k_base = (nb * H_KV + kv_head) * SKVT;
        const uint32_t k_block_bytes = k_block_tiles * tile_bytes;

        Noc noc;

        // Read this core's Q chunk for round `rd` (Q is per-core, never mcast).
        auto read_q = [&](uint32_t rd) {
            const uint32_t qc_raw = rd * GC + col_x;
            const uint32_t q_chunk = (qc_raw < Q_NUM_CHUNKS) ? qc_raw : 0u;  // dummy slot -> Q-chunk 0
            cb_reserve_back(cb_q_in, q_block_tiles);
            uint32_t l1 = get_write_ptr(cb_q_in);
            for (uint32_t st = 0; st < SQ_CHUNK_T; ++st) {
                const uint32_t s_tile = q_chunk * SQ_CHUNK_T + st;
                for (uint32_t dt = 0; dt < DHT; ++dt) {
                    noc_async_read_tile((q_base + s_tile) * DHT + dt, q_acc, l1);
                    l1 += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_q_in, q_block_tiles);
        };

        if (is_sender) {
            auto pipe = mc.sender(noc);
            for (uint32_t rd = 0; rd < rounds; ++rd) {
                read_q(rd);
                for (uint32_t k_chunk = 0; k_chunk < K_NUM_CHUNKS; ++k_chunk) {
                    // K TRANSPOSED (DHT, SK_CHUNK_T) — read from DRAM, then broadcast.
                    cb_reserve_back(cb_k_in, k_block_tiles);
                    uint32_t l1k = get_write_ptr(cb_k_in);
                    for (uint32_t dt = 0; dt < DHT; ++dt) {
                        for (uint32_t kt = 0; kt < SK_CHUNK_T; ++kt) {
                            const uint32_t s_tile = k_chunk * SK_CHUNK_T + kt;
                            noc_async_read_tile((k_base + s_tile) * DHT + dt, k_acc, l1k);
                            l1k += tile_bytes;
                        }
                    }
                    noc_async_read_barrier();
                    if constexpr (mc.active) {
                        pipe.send(get_write_ptr(cb_k_in), get_write_ptr(cb_k_in), k_block_bytes);
                    }
                    cb_push_back(cb_k_in, k_block_tiles);

                    // V natural (SK_CHUNK_T, DHT) — read from DRAM, then broadcast.
                    cb_reserve_back(cb_v_in, k_block_tiles);
                    uint32_t l1v = get_write_ptr(cb_v_in);
                    for (uint32_t kt = 0; kt < SK_CHUNK_T; ++kt) {
                        const uint32_t s_tile = k_chunk * SK_CHUNK_T + kt;
                        for (uint32_t dt = 0; dt < DHT; ++dt) {
                            noc_async_read_tile((k_base + s_tile) * DHT + dt, v_acc, l1v);
                            l1v += tile_bytes;
                        }
                    }
                    noc_async_read_barrier();
                    if constexpr (mc.active) {
                        pipe.send(get_write_ptr(cb_v_in), get_write_ptr(cb_v_in), k_block_bytes);
                    }
                    cb_push_back(cb_v_in, k_block_tiles);
                }
            }
        } else {
            auto pipe = mc.receiver(noc);
            for (uint32_t rd = 0; rd < rounds; ++rd) {
                read_q(rd);
                for (uint32_t k_chunk = 0; k_chunk < K_NUM_CHUNKS; ++k_chunk) {
                    cb_reserve_back(cb_k_in, k_block_tiles);
                    pipe.receive();  // acks sender, waits VALID; K lands at cb_k_in write ptr
                    cb_push_back(cb_k_in, k_block_tiles);

                    cb_reserve_back(cb_v_in, k_block_tiles);
                    pipe.receive();  // V lands at cb_v_in write ptr
                    cb_push_back(cb_v_in, k_block_tiles);
                }
            }
        }
        return;
    }

    for (uint32_t idx = 0; idx < q_count; ++idx) {
        const uint32_t i = q_start + idx;
        const uint32_t q_chunk = i % Q_NUM_CHUNKS;
        const uint32_t t = i / Q_NUM_CHUNKS;
        const uint32_t nq = t % H_Q;
        const uint32_t nb = t / H_Q;
        const uint32_t kv_head = nq / q_heads_per_kv;
        const uint32_t mask_head = MASK_BCAST ? 0u : nq;

        // ---- Q chunk: natural (SQ_CHUNK_T, DHT) ----
        cb_reserve_back(cb_q_in, q_block_tiles);
        uint32_t l1 = get_write_ptr(cb_q_in);
        const uint32_t q_base = (nb * H_Q + nq) * SQT;
        for (uint32_t st = 0; st < SQ_CHUNK_T; ++st) {
            const uint32_t s_tile = q_chunk * SQ_CHUNK_T + st;
            for (uint32_t dt = 0; dt < DHT; ++dt) {
                noc_async_read_tile((q_base + s_tile) * DHT + dt, q_acc, l1);
                l1 += tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_q_in, q_block_tiles);

        const uint32_t k_base = (nb * H_KV + kv_head) * SKVT;
        const uint32_t m_base = (nb * MASK_H + mask_head) * SQT;

        for (uint32_t k_chunk = 0; k_chunk < K_NUM_CHUNKS; ++k_chunk) {
            // ---- K chunk: TRANSPOSED grid (DHT, SK_CHUNK_T) ----
            cb_reserve_back(cb_k_in, k_block_tiles);
            uint32_t l1k = get_write_ptr(cb_k_in);
            for (uint32_t dt = 0; dt < DHT; ++dt) {
                for (uint32_t kt = 0; kt < SK_CHUNK_T; ++kt) {
                    const uint32_t s_tile = k_chunk * SK_CHUNK_T + kt;
                    noc_async_read_tile((k_base + s_tile) * DHT + dt, k_acc, l1k);
                    l1k += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_k_in, k_block_tiles);

            // ---- V chunk: natural (SK_CHUNK_T, DHT) ----
            cb_reserve_back(cb_v_in, k_block_tiles);
            uint32_t l1v = get_write_ptr(cb_v_in);
            for (uint32_t kt = 0; kt < SK_CHUNK_T; ++kt) {
                const uint32_t s_tile = k_chunk * SK_CHUNK_T + kt;
                for (uint32_t dt = 0; dt < DHT; ++dt) {
                    noc_async_read_tile((k_base + s_tile) * DHT + dt, v_acc, l1v);
                    l1v += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_v_in, k_block_tiles);

            // ---- Mask chunk: full (SQ_CHUNK_T, SK_CHUNK_T) ----
            if constexpr (HAS_MASK) {
                cb_reserve_back(cb_mask_in, mask_block_tiles);
                uint32_t l1m = get_write_ptr(cb_mask_in);
                for (uint32_t st = 0; st < SQ_CHUNK_T; ++st) {
                    const uint32_t s_tile = q_chunk * SQ_CHUNK_T + st;
                    for (uint32_t kt = 0; kt < SK_CHUNK_T; ++kt) {
                        const uint32_t k_tile = k_chunk * SK_CHUNK_T + kt;
                        noc_async_read_tile((m_base + s_tile) * SKVT + k_tile, m_acc, l1m);
                        l1m += tile_bytes;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_mask_in, mask_block_tiles);
            }
        }
    }
}
