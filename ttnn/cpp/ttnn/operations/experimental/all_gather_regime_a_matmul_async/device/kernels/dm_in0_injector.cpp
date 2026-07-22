// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused all-gather + regime_a_matmul — Phase A in0 INJECTOR (fabric worker core).
// (REGIME_A_AGMM_EXECUTION_PLAN.md Task 3; design: REGIME_A_AGMM_TASK3_BLUEPRINT.md §B.2.)
//
// Reads this device's local in0 shard once, writes it into the LOCAL slice of the persistent DRAM gather
// buffer, then fabric-unicasts it (per tile) into every remote device's gather buffer at that device's
// global-K offset, and atomic-incs each remote device's readiness semaphore AFTER the payload is flushed
// (payload-before-readiness ordering on the same mux channel). Uses fabric mux v2.
//
// NOTE: this is the streaming/gather producer. The compile-gated AGMM_FULL_GATHER_BARRIER path increments a
// single per-device barrier semaphore once the whole shard has landed instead of per transport chunk.
//
// Runtime-arg contract (must match all_gather_regime_a_matmul_async_program_factory.cpp create()):
//   a0  in0_addr                 (DRAM base of this device's local in0 shard [M, K_local])
//   a1  gather_local_addr        (DRAM base of this device's own gather buffer [M, K_global])
//   a2  pkt_hdr_l1_addr          (L1 scratch for the fabric packet header)
//   a3  payload_l1_addr          (L1 scratch for one tile's payload)
//   a4  Mt                       (M tiles)
//   a5  Kt_local                 (K tiles owned locally by this device)
//   a6  Kt_global                (K tiles of the full gathered buffer = row stride)
//   a7  k_tile_global_base       (global K-tile offset of this device's shard)
//   a8  tile_bytes               (bf16 tile = 2048)
//   a9  n_remote                 (number of remote destinations to unicast to)
//   then, per remote destination r in [0, n_remote):
//     dst_gather_noc_base_lo/hi  (2 words: noc addr base of remote gather buffer)
//     ready_sem_noc_lo/hi        (2 words: noc addr of remote readiness semaphore)
//     num_hops                   (1 word: fabric distance to that destination)
//   ... followed by the FabricMuxV2Sender::build_from_args() args (appended last by the factory).
//
// TensorAccessor compile args for in0 (local shard) and the gather buffer are provided as CT args by the
// factory; addressing uses the shared linear addrgen (global tile id = m*Kt_global + k_global).

#include <cstdint>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

void kernel_main() {
    size_t a = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gather_local_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pkt_hdr_l1_addr = get_arg_val<uint32_t>(a++);
    const uint32_t payload_l1_addr = get_arg_val<uint32_t>(a++);
    const uint32_t Mt = get_arg_val<uint32_t>(a++);
    const uint32_t Kt_local = get_arg_val<uint32_t>(a++);
    const uint32_t Kt_global = get_arg_val<uint32_t>(a++);
    const uint32_t k_tile_global_base = get_arg_val<uint32_t>(a++);
    const uint32_t tile_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t n_remote = get_arg_val<uint32_t>(a++);

    struct RemoteDst {
        uint64_t gather_noc_base;
        uint64_t ready_sem_noc;
        uint32_t num_hops;
    };
    RemoteDst dsts[8];  // D<=8 supported (7 remote + self)
    for (uint32_t r = 0; r < n_remote; ++r) {
        uint32_t lo = get_arg_val<uint32_t>(a++);
        uint32_t hi = get_arg_val<uint32_t>(a++);
        dsts[r].gather_noc_base = (static_cast<uint64_t>(hi) << 32) | lo;
        lo = get_arg_val<uint32_t>(a++);
        hi = get_arg_val<uint32_t>(a++);
        dsts[r].ready_sem_noc = (static_cast<uint64_t>(hi) << 32) | lo;
        dsts[r].num_hops = get_arg_val<uint32_t>(a++);
    }

    // in0 (local shard) and gather-buffer accessors from CT args (factory-provided).
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0_acc = TensorAccessor(in0_args, in0_addr, tile_bytes);
    constexpr auto gather_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto gather_local = TensorAccessor(gather_args, gather_local_addr, tile_bytes);

    auto sender = tt::tt_fabric::FabricMuxV2Sender<>::build_from_args(a);
    auto* pkt_hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(pkt_hdr_l1_addr);
    sender.open();

    // Stream each local tile: read from DRAM shard -> local gather slice -> unicast to each remote gather.
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t kl = 0; kl < Kt_local; ++kl) {
            const uint32_t k_global = k_tile_global_base + kl;
            const uint32_t local_tile_id = m * Kt_local + kl;          // in shard [M, K_local]
            const uint32_t global_tile_id = m * Kt_global + k_global;  // in gather [M, K_global]

            // 1) read the shard tile into L1
            noc_async_read_tile(local_tile_id, in0_acc, payload_l1_addr);
            noc_async_read_barrier();
            // 2) write into this device's own gather buffer (local NoC)
            noc_async_write_tile(global_tile_id, gather_local, payload_l1_addr);
            // 3) fabric-unicast into every remote device's gather buffer at the same global offset
            for (uint32_t r = 0; r < n_remote; ++r) {
                const uint64_t dst = dsts[r].gather_noc_base + static_cast<uint64_t>(global_tile_id) * tile_bytes;
                tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_write(
                    &sender,
                    pkt_hdr,
                    payload_l1_addr,
                    tile_bytes,
                    tt::tt_fabric::NocUnicastCommandHeader{dst},
                    dsts[r].num_hops);
            }
        }
    }
    noc_async_writes_flushed();

    // 4) signal readiness AFTER payload is flushed (same channel => ordered).
    for (uint32_t r = 0; r < n_remote; ++r) {
        tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
            &sender,
            pkt_hdr,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dsts[r].ready_sem_noc, /*inc=*/1u, /*wrap=*/32u},
            dsts[r].num_hops);
    }

    noc_async_write_barrier();
    sender.close();
}
