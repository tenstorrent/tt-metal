// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused all-gather + regime_a_matmul — Phase A in0 INJECTOR (fabric worker core, fabric mux v2).
// (REGIME_A_AGMM_EXECUTION_PLAN.md Task 3; design: REGIME_A_AGMM_TASK3_BLUEPRINT.md.)
//
// Streaming producer. For every tile of this device's in0 K-shard: read it from DRAM, write it into the LOCAL
// slice of this device's DRAM gather buffer (at the shard's global-K offset), and fabric-unicast it into every
// other device's gather buffer at the SAME global offset (the gather buffer is a mesh tensor => identical NoC
// address on all devices, reached via forward hops 1..num_dests on the ring).
//
// Readiness is PER-SHARD and progressive. Each device e increments its own shard-landed GlobalSemaphore
// `shard_landed[e]` on every device (locally after its own write; over fabric on the others, ordered AFTER the
// payload on the same mux channel). This device's injector then walks s = 0..D-1 in GLOBAL-K order, waits for
// shard s to have landed locally, and fans out one increment of the monotonic `gather_ready` GlobalSemaphore to
// every regime_a compute core. gather_ready therefore reaches value k once shards 0..k-1 are all present, so a
// compute core can start reading in0 global K-tile kg as soon as gather_ready > kg/Kt_local — the matmul begins
// before the full gather completes. (The full-gather-barrier diagnostic is just the reader waiting for D.)
//
// Runtime-arg contract (must match all_gather_regime_a_matmul_async_program_factory.cpp create_at()):
//   a0  in0_shard_addr        DRAM base of this device's in0 shard [M, K_local]
//   a1  gather_local_addr     DRAM base of this device's gather buffer [M, K_global]
//   a2  Mt                    M tiles
//   a3  Kt_local              K tiles owned locally
//   a4  Kt_global             K tiles of the gather buffer = row stride
//   a5  k_tile_global_base    global K-tile offset of this device's shard (= device_index * Kt_local)
//   a6  tile_bytes            bf16 tile = 2048
//   a7  num_dests             # remote devices = D-1; reach each via forward hops 1..num_dests
//   a8  D                     device count along the gather axis
//   a9  device_index          this device's shard index (0..D-1)
//   a10 inj_x                 injector core x (symmetric across devices; where shard_landed/gather_ready live)
//   a11 inj_y                 injector core y
//   a12 gather_ready_addr     gather_ready GlobalSemaphore L1 address (monotonic prefix, fanned to compute)
//   a13 num_compute_cores
//   a14 .. a14+D-1            shard_landed[0..D-1] GlobalSemaphore L1 addresses
//   then num_compute_cores * (x, y)
//   then FabricMuxV2Sender::build_from_args() args (appended by the factory).
//
// CT args: TensorAccessorArgs(in0 shard) then TensorAccessorArgs(gather buffer). L1 scratch: CB c_0 = payload
// tile, CB c_1 = fabric packet header.

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
    const uint32_t k_tile_global_base = get_arg_val<uint32_t>(a++);
    const uint32_t tile_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t num_dests = get_arg_val<uint32_t>(a++);
    const uint32_t D = get_arg_val<uint32_t>(a++);
    const uint32_t device_index = get_arg_val<uint32_t>(a++);
    const uint32_t inj_x = get_arg_val<uint32_t>(a++);
    const uint32_t inj_y = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t num_compute_cores = get_arg_val<uint32_t>(a++);

    constexpr uint32_t kMaxD = 8u;
    uint32_t shard_landed_addr[kMaxD];
    for (uint32_t s = 0; s < D; ++s) {
        shard_landed_addr[s] = get_arg_val<uint32_t>(a++);
    }

    constexpr uint32_t kMaxComputeCores = 128u;
    uint32_t cc_x[kMaxComputeCores];
    uint32_t cc_y[kMaxComputeCores];
    for (uint32_t i = 0; i < num_compute_cores; ++i) {
        cc_x[i] = get_arg_val<uint32_t>(a++);
        cc_y[i] = get_arg_val<uint32_t>(a++);
    }

    // in0 shard + gather-buffer accessors from CT args.
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0_acc = TensorAccessor(in0_args, in0_shard_addr, tile_bytes);
    constexpr auto gather_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto gather_acc = TensorAccessor(gather_args, gather_local_addr, tile_bytes);

    // L1 scratch.
    constexpr uint32_t payload_cb = 0, hdr_cb = 1;
    const uint32_t payload_l1 = get_write_ptr(payload_cb);
    auto* pkt_hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(hdr_cb));

    auto sender = tt::tt_fabric::FabricMuxV2Sender<>::build_from_args(a);
    sender.open();

    // Stream each local shard tile: DRAM shard -> L1 -> local gather slice + fabric unicast to every remote.
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t kl = 0; kl < Kt_local; ++kl) {
            const uint32_t k_global = k_tile_global_base + kl;
            const uint32_t local_tile_id = m * Kt_local + kl;          // in shard [M, K_local]
            const uint32_t global_tile_id = m * Kt_global + k_global;  // in gather [M, K_global]

            noc_async_read_page(local_tile_id, in0_acc, payload_l1);
            noc_async_read_barrier();
            noc_async_write_page(global_tile_id, gather_acc, payload_l1);  // local gather write

            const uint64_t dst = get_noc_addr(global_tile_id, gather_acc);
            for (uint32_t h = 1; h <= num_dests; ++h) {
                tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_write(
                    &sender,
                    pkt_hdr,
                    payload_l1,
                    tile_bytes,
                    tt::tt_fabric::NocUnicastCommandHeader{dst},
                    (uint8_t)h);
                // pkt_hdr AND payload_l1 are the (shared) source of this non-blocking send; drain the copy to the
                // mux slot before the next iteration rewrites pkt_hdr's num_hops / the next tile's payload.
                noc_async_writes_flushed();
            }
        }
    }
    noc_async_write_barrier();  // this device's whole local shard has landed in the gather buffer

    // Mark THIS device's shard present everywhere: locally now, and on each remote AFTER its payload flushed
    // (same channel => ordered). Every device e increments shard_landed[e]; nobody else does.
    noc_semaphore_inc(get_noc_addr(inj_x, inj_y, shard_landed_addr[device_index]), 1);
    for (uint32_t h = 1; h <= num_dests; ++h) {
        const uint64_t remote_landed = get_noc_addr(inj_x, inj_y, shard_landed_addr[device_index]);
        tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
            &sender,
            pkt_hdr,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_landed, /*val=*/1u, /*flush=*/true},
            (uint8_t)h);
        noc_async_writes_flushed();  // pkt_hdr reused for the next inc
    }
    noc_async_write_barrier();
    sender.close();

    // Progressive fan-out: as shard s (in GLOBAL-K order) lands locally, bump gather_ready on every compute
    // core. gather_ready reaches k once shards 0..k-1 are all present, matching the reader's in-order K reads.
    for (uint32_t s = 0; s < D; ++s) {
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(shard_landed_addr[s]);
        noc_semaphore_wait_min(landed, 1);
        for (uint32_t i = 0; i < num_compute_cores; ++i) {
            noc_semaphore_inc(get_noc_addr(cc_x[i], cc_y[i], ready_addr), 1);
        }
        noc_async_atomic_barrier();
    }
}
