// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Experiment height_weight_mcast: weight server, writer half (BRISC / NoC1). Per round: wait until every receiver of
// the group can take the round (monotonic ack counter), then multicast each chunk of the server reader's [t0, t1) to
// the same offsets of every rectangle of the group as soon as it landed, and free the slot once the multicast has read
// its source. Each chunk's linked multicast is closed by a 4 B write of zero to the receivers' flag of this slot (still
// INVALID there); the last chunk's closing write is the flag itself. With chained servers (several per group) server k
// multicasts round r only after server k - 1 finished its part of it (concurrent multicasts into one rectangle
// collapse the rate).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t sem_ack = get_compile_time_arg_val(2);
    constexpr uint32_t sem_flag0 = get_compile_time_arg_val(3);
    // v2: one flag per weight slot; receivers ack once at kernel start, then once per round that reuses a slot.
    constexpr uint32_t flag_slots = get_compile_time_arg_val(4);
    constexpr bool v2 = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t sem_chain = get_compile_time_arg_val(7);
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;
    // HWM_SRV_DUAL: this RISC reads the odd chunks of every round on NoC1 (the reader takes the even ones on NoC0).
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(10);
    constexpr auto weight_args = TensorAccessorArgs<11>();

    const uint32_t num_rounds = get_arg_val<uint32_t>(0);
    const uint32_t my_k = get_arg_val<uint32_t>(1);
    const uint32_t n_ack = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t t1 = get_arg_val<uint32_t>(4);
    const uint32_t chain_wait = get_arg_val<uint32_t>(5);  // 1: wait for the previous server of the group
    const uint32_t chain_next = get_arg_val<uint32_t>(6);  // 1: release the next server (x, y below)
    const uint32_t next_x = get_arg_val<uint32_t>(7);
    const uint32_t next_y = get_arg_val<uint32_t>(8);
    const uint32_t weight_addr = get_arg_val<uint32_t>(9);
    const uint32_t num_rects = get_arg_val<uint32_t>(10);
    constexpr uint32_t rects_arg = 11;
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    [[maybe_unused]] const auto weights = TensorAccessor(weight_args, weight_addr, tile_bytes);
    [[maybe_unused]] conv2d_3x3::TaggedReads<false> reads;
    volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ack));
    for (uint32_t slot = 0; slot < flag_slots; ++slot) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_flag0 + my_k * flag_slots + slot)) = VALID;
    }

    for (uint32_t r = 0; r < num_rounds; ++r) {
#ifdef HWM_SRV_DUAL
        {
            // This round's slot is free: this RISC popped its previous round in order.
            const uint32_t l1 = get_read_ptr(cb_weights);
            const uint32_t k_tile0 = (r % num_k_blocks) * block_k_tiles;
            for (uint32_t c = 1; c < num_chunks; c += 2) {
                reads.begin(1 + c);
                const uint32_t lo = t0 > c * chunk_tiles ? t0 : c * chunk_tiles;
                const uint32_t hi = t1 < (c + 1) * chunk_tiles ? t1 : (c + 1) * chunk_tiles;
                for (uint32_t t = lo; t < hi; ++t) {
                    reads.read(weights.get_noc_addr(k_tile0 * tensor_n_tiles + t), l1 + t * tile_bytes, tile_bytes);
                }
            }
            dataflow_kernel_lib::set_read_trid(reads.noc, 0);
        }
#endif
        {
            MaybeDeviceZoneScope("server_wait_ack");
            const uint32_t acks = v2 ? 1 + (r + 1 > flag_slots ? r + 1 - flag_slots : 0) : r + 1;
            noc_semaphore_wait_min(ack, acks * n_ack);
        }
        if (chain_wait) {
            MaybeDeviceZoneScope("server_wait_chain");
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_chain)), r + 1);
        }
        const uint32_t flag = get_semaphore(sem_flag0 + my_k * flag_slots + r % flag_slots);
        const uint32_t base = get_read_ptr(cb_weights);
        for (uint32_t c = 0; c < num_chunks; ++c) {
            const uint32_t lo = t0 > c * chunk_tiles ? t0 : c * chunk_tiles;
            const uint32_t hi = t1 < (c + 1) * chunk_tiles ? t1 : (c + 1) * chunk_tiles;
            {
                MaybeDeviceZoneScope("server_wait_read");
                cb_wait_front(cb_weights, (c + 1) * chunk_tiles);
#ifdef HWM_SRV_DUAL
                if (c & 1) {
                    reads.wait(1 + c);
                }
#endif
            }
            if (lo >= hi) {
                continue;
            }
            MaybeDeviceZoneScope("server_send");
            const uint32_t src = base + lo * tile_bytes;
            const uint32_t bytes = (hi - lo) * tile_bytes;
            const bool last = hi == t1;
            for (uint32_t i = 0; i < num_rects; ++i) {
                const uint32_t a = rects_arg + 5 * i;
                const uint32_t sx = get_arg_val<uint32_t>(a), sy = get_arg_val<uint32_t>(a + 1);
                const uint32_t ex = get_arg_val<uint32_t>(a + 2), ey = get_arg_val<uint32_t>(a + 3);
                const uint32_t nd = get_arg_val<uint32_t>(a + 4);
                noc_async_write_multicast(src, get_noc_multicast_addr(sx, sy, ex, ey, src), bytes, nd, true);
                noc_semaphore_set_multicast(
                    last ? flag : MEM_ZEROS_BASE, get_noc_multicast_addr(sx, sy, ex, ey, flag), nd, false);
            }
        }
        {
            // The reader refills this slot two rounds on: the multicast must have read it.
            MaybeDeviceZoneScope("server_flush");
            noc_async_writes_flushed();
        }
        if (chain_next) {
            noc_semaphore_inc(get_noc_addr(next_x, next_y, get_semaphore(sem_chain)), 1);
        }
        cb_pop_front(cb_weights, round_tiles);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
