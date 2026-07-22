// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused all-gather + regime_a_matmul — Phase A in0 INJECTOR (fabric worker core, fabric mux v2).
// (REGIME_A_AGMM_EXECUTION_PLAN.md Task 3, milestone #17; design: REGIME_A_AGMM_TASK3_BLUEPRINT.md §B.2.)
//
// D=2 full-gather-barrier producer. For every tile of this device's in0 K-shard:
//   1. read it from the shard DRAM buffer into L1,
//   2. write it into the LOCAL slice of this device's DRAM gather buffer (at the shard's global-K offset),
//   3. fabric-unicast it into the forward neighbour's gather buffer at the SAME global offset (the gather
//      buffer is a mesh tensor => identical address on the neighbour, so this device's TensorAccessor noc
//      address is valid there).
// Then, after the payload has flushed on the fabric channel, atomic-inc the neighbour's gather_progress
// GlobalSemaphore (payload-before-readiness on the same channel). Finally wait until this device's own
// gather_progress reaches D-1 (all remote shards landed here) and fan out the local gather_ready semaphore to
// every regime_a compute core, releasing their in0 reads.
//
// Runtime-arg contract (must match all_gather_regime_a_matmul_async_program_factory.cpp create_at()):
//   a0  in0_shard_addr            (DRAM base of this device's in0 shard [M, K_local])
//   a1  gather_local_addr         (DRAM base of this device's gather buffer [M, K_global])
//   a2  Mt                        (M tiles)
//   a3  Kt_local                  (K tiles owned locally)
//   a4  Kt_global                 (K tiles of the gather buffer = row stride)
//   a5  k_tile_global_base        (global K-tile offset of this device's shard)
//   a6  tile_bytes                (bf16 tile = 2048)
//   a7  num_dests                 (# remote devices = D-1; reach each via forward hops 1..num_dests)
//   a8  progress_sem_addr         (gather_progress GlobalSemaphore L1 address; local read + remote target)
//   a9  nbr_inj_x                 (neighbour injector core x — where its gather_progress lives)
//   a10 nbr_inj_y                 (neighbour injector core y)
//   a11 ready_sem_id              (local gather_ready semaphore id, fanned out to compute cores)
//   a12 expected_remote           (remote shards to await before fan-out = D-1)
//   a13 num_compute_cores
//   a14.. per compute core: (x, y)  [num_compute_cores pairs]
//   ...   then FabricMuxV2Sender::build_from_args() args (appended by the factory).
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
    const uint32_t num_dests = get_arg_val<uint32_t>(a++);  // reach devices 1..num_dests hops forward (ring)
    const uint32_t progress_sem_addr = get_arg_val<uint32_t>(a++);
    const uint32_t nbr_inj_x = get_arg_val<uint32_t>(a++);
    const uint32_t nbr_inj_y = get_arg_val<uint32_t>(a++);
    const uint32_t ready_sem_id = get_arg_val<uint32_t>(a++);
    const uint32_t expected_remote = get_arg_val<uint32_t>(a++);
    const uint32_t num_compute_cores = get_arg_val<uint32_t>(a++);

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

    // Stream each local shard tile: DRAM shard -> L1 -> local gather slice + fabric unicast to neighbour gather.
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t kl = 0; kl < Kt_local; ++kl) {
            const uint32_t k_global = k_tile_global_base + kl;
            const uint32_t local_tile_id = m * Kt_local + kl;          // in shard [M, K_local]
            const uint32_t global_tile_id = m * Kt_global + k_global;  // in gather [M, K_global]

            noc_async_read_page(local_tile_id, in0_acc, payload_l1);
            noc_async_read_barrier();
            noc_async_write_page(global_tile_id, gather_acc, payload_l1);  // local gather write

            // Same global-K offset on every device's gather buffer (mesh tensor => identical NoC address).
            const uint64_t dst = get_noc_addr(global_tile_id, gather_acc);
            for (uint32_t h = 1; h <= num_dests; ++h) {
                tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_write(
                    &sender,
                    pkt_hdr,
                    payload_l1,
                    tile_bytes,
                    tt::tt_fabric::NocUnicastCommandHeader{dst},
                    (uint8_t)h);
                // pkt_hdr AND payload_l1 are the (shared) source of this non-blocking send; drain the copy to
                // the mux slot before the next iteration rewrites pkt_hdr's num_hops / the next tile's payload.
                noc_async_writes_flushed();
            }
        }
    }
    noc_async_writes_flushed();  // local gather writes departed L1

    // Signal readiness to each remote AFTER the payload flushed (same channel => ordered delivery).
    const uint64_t nbr_progress = get_noc_addr(nbr_inj_x, nbr_inj_y, progress_sem_addr);
    for (uint32_t h = 1; h <= num_dests; ++h) {
        tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
            &sender,
            pkt_hdr,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{nbr_progress, /*val=*/1u, /*flush=*/true},
            (uint8_t)h);
        noc_async_writes_flushed();  // pkt_hdr reused for the next inc
    }

    noc_async_write_barrier();
    sender.close();

    // Wait until every remote shard has landed in THIS device's gather buffer, then release the compute readers.
    volatile tt_l1_ptr uint32_t* progress_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(progress_sem_addr);
    noc_semaphore_wait_min(progress_ptr, expected_remote);

    const uint32_t ready_addr = get_semaphore(ready_sem_id);
    for (uint32_t i = 0; i < num_compute_cores; ++i) {
        noc_semaphore_inc(get_noc_addr(cc_x[i], cc_y[i], ready_addr), 1);
    }
    noc_async_atomic_barrier();
}
