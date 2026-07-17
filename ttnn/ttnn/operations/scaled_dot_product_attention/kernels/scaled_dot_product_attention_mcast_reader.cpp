// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for the NoC-multicast (KV read-once + broadcast) SDPA variant.
//
// ONE source, two roles selected by the compile-time IS_SENDER flag (emitted on
// two core ranges by the descriptor): column-0 of each row is the SENDER (reads
// K/V from DRAM once per KV chunk and Mcast1D-broadcasts it across its row); the
// other columns are RECEIVERS (receive the broadcast). Every core also reads its
// OWN Q sub-chunks and fills the reduce scaler + attention scale once.
//
// mcast_bcast (compile-time): when TRUE (shipped mcast path) the sender broadcasts
// and receivers receive. When FALSE (ablation SDPA_MCAST_NO_BCAST) EVERY core
// reads its own K/V from DRAM (the KV-outer analogue of the baseline's redundant
// re-reads, no broadcast) — a clean A/B that isolates the broadcast's effect from
// the KV-outer restructure with the compute kept byte-identical.
//
// Layout of a KV chunk in cb_k_in / cb_v_in matches the compute matmul contract:
//   K is D-major (outer d, inner skv) for the transposed QKᵀ in1 read;
//   V is row-major (outer skv, inner d) for the PV in1 read.

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
constexpr uint32_t cb_q_stage = 0;
constexpr uint32_t cb_k_in = 1;
constexpr uint32_t cb_v_in = 2;
constexpr uint32_t cb_scaler = 3;
constexpr uint32_t cb_scale = 4;
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
    // MEASUREMENT-ONLY (env SDPA_MCAST_ABLATE_READER): skip the K/V DRAM reads AND
    // the broadcast, keeping cb_k_in/cb_v_in reserve/push intact so compute runs on
    // garbage — the KV-outer compute-floor probe. Forces the no-broadcast branch.
    constexpr bool ablate_reader = get_compile_time_arg_val(11) != 0;

    constexpr auto mc = McastArgs</*CT=*/12, /*RT=*/7>();
    constexpr uint32_t TA = mc.next_compile_time_args_offset();  // == 16
    constexpr auto q_args = TensorAccessorArgs<TA>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t b = get_arg_val<uint32_t>(3);
    const uint32_t h = get_arg_val<uint32_t>(4);
    const uint32_t q_start = get_arg_val<uint32_t>(5);
    const uint32_t q_cnt = get_arg_val<uint32_t>(6);

    const uint32_t tile_bytes = get_tile_size(cb_k_in);
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);

    // --- scaler (1.0) for both MAX and SUM REDUCE_ROW; one tile serves both ---
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    // --- scale tile: fill the whole tile with the resolved scale (RNE bf16) ---
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
    constexpr uint32_t kv_tiles = Skv_chunk_t * Dt;  // K (Dt*Skv) and V (Skv*Dt) both = kv_tiles
    const uint32_t kv_bytes = kv_tiles * tile_bytes;

    // --- Q sub-chunks: read this core's q_cnt owned Q chunks (in order) ---
    for (uint32_t s = 0; s < q_cnt; ++s) {
        const uint32_t sq_off = (q_start + s) * Sq_chunk_t;
        cb_reserve_back(cb_q_stage, q_tiles);
        uint32_t wptr = get_write_ptr(cb_q_stage);
        for (uint32_t t = 0; t < q_tiles; ++t) {
            const uint32_t sq_g = sq_off + (t / Dt);
            const uint32_t d = t % Dt;
            noc_async_read_tile((q_base + sq_g) * Dt + d, q_acc, wptr);
            wptr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q_stage, q_tiles);
    }

    // --- KV-outer stream ---
    // read_k / read_v fill a reserved cb slot at `wptr` from DRAM (matmul-contract
    // layouts). Used by the sender always, and by receivers only when !mcast_bcast.
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
        for (uint32_t j = 0; j < n_kv_chunks; ++j) {
            const uint32_t skv_off = j * Skv_chunk_t;
            // K
            cb_reserve_back(cb_k_in, kv_tiles);
            uint32_t kptr = get_write_ptr(cb_k_in);
            read_k(kptr, skv_off);
            if constexpr (mc.active) {
                pipe.send(kptr, kptr, kv_bytes);  // broadcast K to the row (self excluded)
            }
            cb_push_back(cb_k_in, kv_tiles);
            // V
            cb_reserve_back(cb_v_in, kv_tiles);
            uint32_t vptr = get_write_ptr(cb_v_in);
            read_v(vptr, skv_off);
            if constexpr (mc.active) {
                pipe.send(vptr, vptr, kv_bytes);  // broadcast V to the row
            }
            cb_push_back(cb_v_in, kv_tiles);
        }
    } else if constexpr (mcast_bcast && !ablate_reader && !IS_SENDER) {
        auto pipe = mc.receiver(noc);
        for (uint32_t j = 0; j < n_kv_chunks; ++j) {
            // K: reserve first so the write ptr is a safe landing address, then ack+wait.
            cb_reserve_back(cb_k_in, kv_tiles);
            pipe.receive();
            cb_push_back(cb_k_in, kv_tiles);
            // V
            cb_reserve_back(cb_v_in, kv_tiles);
            pipe.receive();
            cb_push_back(cb_v_in, kv_tiles);
        }
    } else {
        // Broadcast OFF: every core reads its own K/V from DRAM (KV-outer analogue of
        // the baseline's redundant re-reads). Under ablate_reader the reads are skipped
        // (CB reserve/push kept) for the compute-floor probe.
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
