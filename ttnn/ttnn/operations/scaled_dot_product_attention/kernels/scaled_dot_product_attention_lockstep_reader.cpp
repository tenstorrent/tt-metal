// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for the Q-OUTER LOCKSTEP multicast SDPA variant.
//
// The whole point of this variant vs the KV-outer one: the loop structure is
// UNCHANGED from baseline (Q-outer — one q-chunk and one (m,l,O) state per core
// at a time), so the shipped optimized compute kernel is reused VERBATIM and the
// ~4.888 ms compute floor is untouched. The ONLY differences from the baseline
// reader are (1) head-grouped work (each head's q-chunks live on one row of the
// grid) and (2) the KV stream is delivered by ONE injector per row that reads it
// once from DRAM and Mcast1D-broadcasts it, while the row marches its KV index in
// LOCKSTEP (the mcast semaphore handshake is the barrier).
//
// WAVE model: a row of `cores_per_row` cores covers a head's n_q_chunks q-chunks
// in `num_waves = ceil(n_q_chunks / cores_per_row)` waves. In wave k, core column
// c processes q-chunk `qc = k*cores_per_row + c`, CLAMPED to n_q_chunks-1 when the
// last wave is short (the clamp recomputes the last q-chunk redundantly — same
// value, so writing it twice is harmless — which keeps every core active every
// wave, so there are no idle receivers to deadlock the broadcast, and the critical
// path stays num_waves q-chunks = the baseline max-per-core). Every wave the row
// re-reads the head's KV stream once via the injector (so ~cores_per_row-fold DRAM
// reduction, not 37x — fine, reads are hidden behind compute).
//
// Feeds the BASELINE compute CB layout: cb_q_in=0, cb_k_in=1, cb_v_in=2,
// cb_scaler=4, cb_scale=5. Per wave: 1 Q chunk to cb_q_in, then n_kv_chunks K/V
// chunks to cb_k_in/cb_v_in (K D-major, V row-major — the matmul contract).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

namespace {
constexpr uint32_t cb_q_in = 0;
constexpr uint32_t cb_k_in = 1;
constexpr uint32_t cb_v_in = 2;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_scale = 5;
}  // namespace

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Skv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t n_kv_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(4);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(5);
    constexpr uint32_t H = get_compile_time_arg_val(6);
    constexpr uint32_t H_kv = get_compile_time_arg_val(7);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(8);
    constexpr bool IS_SENDER = get_compile_time_arg_val(9) != 0;
    constexpr bool mcast_bcast = get_compile_time_arg_val(10) != 0;
    constexpr bool ablate_reader = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t num_waves = get_compile_time_arg_val(12);
    constexpr uint32_t cores_per_row = get_compile_time_arg_val(13);
    constexpr uint32_t n_q_chunks = get_compile_time_arg_val(14);

    constexpr auto mc = McastArgs</*CT=*/15, /*RT=*/6>();
    constexpr uint32_t TA = mc.next_compile_time_args_offset();  // == 20
    constexpr auto q_args = TensorAccessorArgs<TA>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t b = get_arg_val<uint32_t>(3);
    const uint32_t h = get_arg_val<uint32_t>(4);
    const uint32_t col = get_arg_val<uint32_t>(5);  // this core's column in the row

    const uint32_t tile_bytes = get_tile_size(cb_k_in);
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);

    // scaler (1.0) for MAX/SUM REDUCE_ROW; scale tile (RNE bf16) — once.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    {
        cb_reserve_back(cb_scale, 1);
        uint32_t wptr = get_write_ptr(cb_scale);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wptr);
        const uint32_t rne_bias = 0x7FFFu + ((scale_bits >> 16) & 1u);
        const uint16_t sb = static_cast<uint16_t>((scale_bits + rne_bias) >> 16);
        const uint32_t packed = (static_cast<uint32_t>(sb) << 16) | sb;
        const uint32_t words = get_tile_size(cb_scale) / 4;
        for (uint32_t i = 0; i < words; ++i) {
            p[i] = packed;
        }
        cb_push_back(cb_scale, 1);
    }

    const uint32_t kv_head = h / (H / H_kv);  // MHA => kv_head == h
    const uint32_t q_base = (b * H + h) * Sq_t;
    const uint32_t kv_base = (b * H_kv + kv_head) * Skv_t;

    constexpr uint32_t q_tiles = Sq_chunk_t * Dt;
    constexpr uint32_t kv_tiles = Skv_chunk_t * Dt;
    const uint32_t kv_bytes = kv_tiles * tile_bytes;

    auto read_q = [&](uint32_t qc) {
        const uint32_t sq_off = qc * Sq_chunk_t;
        cb_reserve_back(cb_q_in, q_tiles);
        uint32_t w = get_write_ptr(cb_q_in);
        for (uint32_t t = 0; t < q_tiles; ++t) {
            const uint32_t sq_g = sq_off + (t / Dt);
            const uint32_t d = t % Dt;
            noc_async_read_tile((q_base + sq_g) * Dt + d, q_acc, w);
            w += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q_in, q_tiles);
    };
    auto read_k = [&](uint32_t wptr, uint32_t skv_off) {
        uint32_t w = wptr;
        for (uint32_t t = 0; t < kv_tiles; ++t) {
            const uint32_t d = t / Skv_chunk_t;
            const uint32_t skv_g = skv_off + (t % Skv_chunk_t);
            noc_async_read_tile((kv_base + skv_g) * Dt + d, k_acc, w);
            w += tile_bytes;
        }
        noc_async_read_barrier();
    };
    auto read_v = [&](uint32_t wptr, uint32_t skv_off) {
        uint32_t w = wptr;
        for (uint32_t t = 0; t < kv_tiles; ++t) {
            const uint32_t skv_g = skv_off + (t / Dt);
            const uint32_t d = t % Dt;
            noc_async_read_tile((kv_base + skv_g) * Dt + d, v_acc, w);
            w += tile_bytes;
        }
        noc_async_read_barrier();
    };

    Noc noc;

    if constexpr (mcast_bcast && !ablate_reader && IS_SENDER) {
        auto pipe = mc.sender(noc);
        for (uint32_t k = 0; k < num_waves; ++k) {
            uint32_t qc = k * cores_per_row + col;
            if (qc >= n_q_chunks) {
                qc = n_q_chunks - 1;  // clamp-pad: recompute last q-chunk (redundant, same value)
            }
            read_q(qc);
            for (uint32_t j = 0; j < n_kv_chunks; ++j) {
                const uint32_t skv_off = j * Skv_chunk_t;
                cb_reserve_back(cb_k_in, kv_tiles);
                uint32_t kptr = get_write_ptr(cb_k_in);
                read_k(kptr, skv_off);
                if constexpr (mc.active) {
                    pipe.send(kptr, kptr, kv_bytes);
                }
                cb_push_back(cb_k_in, kv_tiles);
                cb_reserve_back(cb_v_in, kv_tiles);
                uint32_t vptr = get_write_ptr(cb_v_in);
                read_v(vptr, skv_off);
                if constexpr (mc.active) {
                    pipe.send(vptr, vptr, kv_bytes);
                }
                cb_push_back(cb_v_in, kv_tiles);
            }
        }
    } else if constexpr (mcast_bcast && !ablate_reader && !IS_SENDER) {
        auto pipe = mc.receiver(noc);
        for (uint32_t k = 0; k < num_waves; ++k) {
            uint32_t qc = k * cores_per_row + col;
            if (qc >= n_q_chunks) {
                qc = n_q_chunks - 1;
            }
            read_q(qc);
            for (uint32_t j = 0; j < n_kv_chunks; ++j) {
                cb_reserve_back(cb_k_in, kv_tiles);
                pipe.receive();
                cb_push_back(cb_k_in, kv_tiles);
                cb_reserve_back(cb_v_in, kv_tiles);
                pipe.receive();
                cb_push_back(cb_v_in, kv_tiles);
            }
        }
    } else {
        // Broadcast OFF: every core reads its own K/V from DRAM (KV-stream re-read
        // per wave, per core). Under ablate_reader reads are skipped (floor probe).
        for (uint32_t k = 0; k < num_waves; ++k) {
            uint32_t qc = k * cores_per_row + col;
            if (qc >= n_q_chunks) {
                qc = n_q_chunks - 1;
            }
            if constexpr (!ablate_reader) {
                read_q(qc);
            } else {
                cb_reserve_back(cb_q_in, q_tiles);
                cb_push_back(cb_q_in, q_tiles);
            }
            for (uint32_t j = 0; j < n_kv_chunks; ++j) {
                const uint32_t skv_off = j * Skv_chunk_t;
                cb_reserve_back(cb_k_in, kv_tiles);
                if constexpr (!ablate_reader) {
                    read_k(get_write_ptr(cb_k_in), skv_off);
                }
                cb_push_back(cb_k_in, kv_tiles);
                cb_reserve_back(cb_v_in, kv_tiles);
                if constexpr (!ablate_reader) {
                    read_v(get_write_ptr(cb_v_in), skv_off);
                }
                cb_push_back(cb_v_in, kv_tiles);
            }
        }
    }
}
