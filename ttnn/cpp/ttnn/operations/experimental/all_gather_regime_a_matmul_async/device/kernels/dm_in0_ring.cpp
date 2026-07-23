// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused all-gather + regime_a_matmul — Phase A in0 STREAMING RING relay (fabric worker core, mux v2).
// (REGIME_A_AGMM_EXECUTION_PLAN.md Task 3; design: REGIME_A_AGMM_TASK3_STREAMING_RING.md.)
//
// DRAM-staged streaming ring, transport chunk = one kb K-block (Mt*kb bf16 tiles). General D (>=2) neighbor
// store-and-forward, bandwidth-optimal unidirectional ring all-gather over D-1 rounds interleaved into ONE
// per-chunk loop (send + drain per iteration — NOT sequential rounds, which would deadlock past num_slots).
//
// Per device d, iteration i (i = r*blocks_per_shard + b, round r = 0..D-2, block b = 0..blocks_per_shard-1):
//   send_shard = (d - r) mod D   (the shard we forward to the +1 neighbour this round)
//   recv_shard = (d - 1 - r) mod D (the shard we receive from the -1 neighbour this round)
// STEP 1 produce: r==0 -> read our OWN in0 shard block b from DRAM into the L1 send-slot, store it to local
//   gather DRAM at shard d's global offset, publish blk_ready[d] (local-first). r>=1 -> forward-from-DRAM: read
//   gather[send_shard] block b (stored in round r-1) into the send-slot (no L1 slot lifetime across rounds).
// STEP 2 send: wait a send-slot credit (only once >= num_slots chunks are outstanding), fabric-forward the
//   contiguous L1 block to the +1 neighbour's recv-slot (packet_payload_bytes packets, 1 hop), inc its recv_sem.
// STEP 3 drain: wait recv_sem (our recv-slot holds the neighbour's chunk), store it to local gather DRAM at
//   recv_shard's global offset, publish blk_ready[recv_shard], return a credit to the -1 neighbour (inc its
//   credit_sem via num_hops = D-1, ring wrap) once store+forward are source-safe.
// Send-before-drain is safe: to reach STEP 2 of iteration i the device has finished iteration i-1's drain, so
// the downstream device already drained iteration i-num_slots and its credit is present; a full wait-cycle
// round the ring is impossible (would need every device >= num_slots ahead of its +1 neighbour).
//
// Ordering/lifetime: payload stored (write barrier) before readiness; forward payload flushed before the send
// slot is reused; the -1 credit is returned only after the received block is stored (source-safe); payload
// writes + non-posted atomics are drained before exit. Bounded L1 slots (num_slots) force wraparound when
// (D-1)*blocks_per_shard > num_slots.
//
// Runtime args (must match all_gather_regime_a_matmul_async_program_factory.cpp create_at()):
//   a0 in0_shard_addr, a1 gather_local_addr, a2 Mt, a3 Kt_local, a4 Kt_global, a5 kb, a6 tile_bytes,
//   a7 blocks_per_shard (=Kt_local/kb), a8 D, a9 device_index, a10 inj_x, a11 inj_y, a12 num_slots,
//   a13 recv_sem_addr, a14 credit_sem_addr, a15 packet_payload_bytes, a16 num_compute_cores,
//   a17..a17+D-1 blk_ready[0..D-1] addresses, then num_compute_cores*(x,y), then FabricMuxV2Sender args.
// CT: TensorAccessorArgs(in0 shard) then TensorAccessorArgs(gather buffer). CBs: c_0 send slots, c_1 recv
// slots, c_2 packet header.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

void kernel_main() {
    size_t a = 0;
    const uint32_t in0_shard_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gather_local_addr = get_arg_val<uint32_t>(a++);
    const uint32_t Mt = get_arg_val<uint32_t>(a++);
    const uint32_t Kt_local = get_arg_val<uint32_t>(a++);
    const uint32_t Kt_global = get_arg_val<uint32_t>(a++);
    const uint32_t kb = get_arg_val<uint32_t>(a++);
    const uint32_t tile_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t blocks_per_shard = get_arg_val<uint32_t>(a++);
    const uint32_t D = get_arg_val<uint32_t>(a++);
    const uint32_t device_index = get_arg_val<uint32_t>(a++);
    const uint32_t inj_x = get_arg_val<uint32_t>(a++);
    const uint32_t inj_y = get_arg_val<uint32_t>(a++);
    const uint32_t num_slots = get_arg_val<uint32_t>(a++);
    const uint32_t recv_sem_addr = get_arg_val<uint32_t>(a++);
    const uint32_t credit_sem_addr = get_arg_val<uint32_t>(a++);
    const uint32_t packet_payload_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t num_compute_cores = get_arg_val<uint32_t>(a++);

    constexpr uint32_t kMaxD = 8u;
    uint32_t blk_ready_addr[kMaxD];
    for (uint32_t s = 0; s < D; ++s) {
        blk_ready_addr[s] = get_arg_val<uint32_t>(a++);
    }
    constexpr uint32_t kMaxComputeCores = 128u;
    uint32_t cc_x[kMaxComputeCores];
    uint32_t cc_y[kMaxComputeCores];
    for (uint32_t i = 0; i < num_compute_cores; ++i) {
        cc_x[i] = get_arg_val<uint32_t>(a++);
        cc_y[i] = get_arg_val<uint32_t>(a++);
    }

    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0_acc = TensorAccessor(in0_args, in0_shard_addr, tile_bytes);
    constexpr auto gather_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto gather_acc = TensorAccessor(gather_args, gather_local_addr, tile_bytes);

    constexpr uint32_t send_cb = 0, recv_cb = 1, hdr_cb = 2;
    const uint32_t send_base = get_write_ptr(send_cb);
    const uint32_t recv_base = get_write_ptr(recv_cb);
    auto* pkt_hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(hdr_cb));

    const uint32_t chunk_tiles = Mt * kb;
    const uint32_t chunk_bytes = chunk_tiles * tile_bytes;
    const uint32_t fwd_hops = 1u;         // data + recv_sem go one hop to the +1 neighbour
    const uint32_t credit_hops = D - 1u;  // credit wraps the ring to the -1 (backward) neighbour

    volatile tt_l1_ptr uint32_t* recv_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_addr);
    volatile tt_l1_ptr uint32_t* credit_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_sem_addr);

    auto sender = tt::tt_fabric::FabricMuxV2Sender<>::build_from_args(a);
    sender.open();

    auto fanout = [&](uint32_t s) {
        for (uint32_t i = 0; i < num_compute_cores; ++i) {
            noc_semaphore_inc(get_noc_addr(cc_x[i], cc_y[i], blk_ready_addr[s]), 1);
        }
        noc_async_atomic_barrier();
    };

    const uint32_t total_iters = (D - 1u) * blocks_per_shard;  // D-1 forward rounds, blocks_per_shard chunks each
    for (uint32_t i = 0; i < total_iters; ++i) {
        const uint32_t r = i / blocks_per_shard;      // round 0..D-2
        const uint32_t b = i - r * blocks_per_shard;  // block within the round's shard
        const uint32_t send_shard = (device_index + D - r) % D;
        const uint32_t recv_shard = (send_shard + D - 1u) % D;  // one shard "behind" send_shard on the ring
        const uint32_t slot = i % num_slots;
        const uint32_t send_slot = send_base + slot * chunk_bytes;
        const uint32_t recv_slot = recv_base + slot * chunk_bytes;

        // ---- STEP 1: produce the send payload into the L1 send-slot ----
        uint32_t p = send_slot;
        if (r == 0u) {
            // own shard: read block b from our in0 DRAM shard [M, K_local]
            for (uint32_t m = 0; m < Mt; ++m) {
                for (uint32_t kt = 0; kt < kb; ++kt) {
                    noc_async_read_page(m * Kt_local + b * kb + kt, in0_acc, p);
                    p += tile_bytes;
                }
            }
            noc_async_read_barrier();
            // store own block to local gather DRAM at shard d's global offset, then publish it (local-first)
            p = send_slot;
            for (uint32_t m = 0; m < Mt; ++m) {
                for (uint32_t kt = 0; kt < kb; ++kt) {
                    noc_async_write_page(m * Kt_global + send_shard * Kt_local + b * kb + kt, gather_acc, p);
                    p += tile_bytes;
                }
            }
            noc_async_write_barrier();
            fanout(send_shard);  // send_shard == device_index in round 0
        } else {
            // forward-from-DRAM: read shard send_shard block b out of local gather DRAM (stored in round r-1)
            for (uint32_t m = 0; m < Mt; ++m) {
                for (uint32_t kt = 0; kt < kb; ++kt) {
                    noc_async_read_page(m * Kt_global + send_shard * Kt_local + b * kb + kt, gather_acc, p);
                    p += tile_bytes;
                }
            }
            noc_async_read_barrier();
        }

        // ---- STEP 2: forward the contiguous L1 block to the +1 neighbour's recv-slot ----
        if (i >= num_slots) {  // reuse of the +1 neighbour's recv-slot: it must have freed chunk (i - num_slots)
            noc_semaphore_wait_min(credit_ptr, i - num_slots + 1u);
        }
        for (uint32_t off = 0; off < chunk_bytes; off += packet_payload_bytes) {
            const uint32_t sz = (chunk_bytes - off) < packet_payload_bytes ? (chunk_bytes - off) : packet_payload_bytes;
            const uint64_t dst = get_noc_addr(inj_x, inj_y, recv_slot + off);
            tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_write(
                &sender, pkt_hdr, send_slot + off, sz, tt::tt_fabric::NocUnicastCommandHeader{dst}, (uint8_t)fwd_hops);
            noc_async_writes_flushed();  // pkt_hdr + send_slot are the send source; drain before the loop reuses them
        }
        // recv_sem AFTER payload flushed (same channel => ordered): tells the +1 neighbour chunk i has landed
        tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
            &sender,
            pkt_hdr,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{get_noc_addr(inj_x, inj_y, recv_sem_addr), 1u, true},
            (uint8_t)fwd_hops);
        noc_async_writes_flushed();

        // ---- STEP 3: drain the neighbour's chunk from our recv-slot ----
        noc_semaphore_wait_min(recv_ptr, i + 1u);  // the -1 neighbour wrote chunk i into our recv-slot
        p = recv_slot;
        for (uint32_t m = 0; m < Mt; ++m) {
            for (uint32_t kt = 0; kt < kb; ++kt) {
                noc_async_write_page(m * Kt_global + recv_shard * Kt_local + b * kb + kt, gather_acc, p);
                p += tile_bytes;
            }
        }
        noc_async_write_barrier();  // recv_shard's block stored -> safe to publish + return the credit
        fanout(recv_shard);
        // return a credit to the -1 neighbour: our recv-slot for chunk i is now free (we forward from DRAM, not L1)
        tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
            &sender,
            pkt_hdr,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{get_noc_addr(inj_x, inj_y, credit_sem_addr), 1u, true},
            (uint8_t)credit_hops);
        noc_async_writes_flushed();
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    sender.close();
}
