// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert_slot_copy writer (RISCV_0, NoC1, this port core's fabric connection):
//  1. receiver: a ready (atomic increment) to each sender core of the source chip - this chip's program has started,
//     so every program before it on this chip (the MoE that reads the slot) is done;
//  2. per job: a local copy (NoC writes into this chip's DRAM), or a send: wait for the destination's ready, then this
//     part's chunks as fabric unicast writes into the destination's DRAM, then a flushed atomic increment of its done
//     word (it vouches for the writes before it on the link);
//  3. receiver: wait for the done of every sender core, so the programs after this one see the slot filled.
// Compile-time args: 0 CB_DATA, 1 CB_TAB.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "slot_copy_common.hpp"

void kernel_main() {
    using namespace tt::tt_fabric;
    constexpr uint32_t cb_data = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tab = get_compile_time_arg_val(1);

    const uint32_t dst[2] = {get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3)};
    const uint32_t sems = get_arg_val<uint32_t>(5);
    const uint32_t me = get_arg_val<uint32_t>(6);
    const uint32_t n_jobs = get_arg_val<uint32_t>(7);
    const uint32_t n_ready = get_arg_val<uint32_t>(esc::RCV0);
    const uint32_t src_node = get_arg_val<uint32_t>(esc::RCV0 + 1);
    bool fabric = n_ready != 0;  // (the connection only when this core sends: a local-only call opens none)
    for (uint32_t j = 0; j < n_jobs; ++j) {
        fabric |= esc::job(j).dst != esc::LOCAL;
    }
    size_t a = esc::NARGS;
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    if (fabric) {
        conn.open();
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* shdr = PacketHeaderPool::allocate_header();

    // 1. readies to the source's sender cores (word 1 + me of each)
    if (n_ready) {
        (void)fabric_set_unicast_route(shdr, src_node & 0xFFFF, src_node >> 16);
        for (uint32_t l = 0; l < n_ready; ++l) {
            const uint32_t xy = get_arg_val<uint32_t>(esc::RCV0 + 2 + l);
            shdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                get_noc_addr(xy & 0xFFFF, xy >> 16, sems + (1 + me) * esc::SEM_STRIDE), 1, false});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(shdr), sizeof(PACKET_HEADER_TYPE));
        }
    }

    // 2. jobs
    for (uint32_t j = 0; j < n_jobs; ++j) {
        const esc::Job jb = esc::job(j);
        cb_wait_front(cb_tab, 1);
        const volatile tt_l1_ptr uint32_t* tab =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_tab));
        if (jb.dst == esc::LOCAL) {
            esc::for_my_chunks(
                tab, jb.part, jb.parts, [&](uint32_t w, uint32_t, uint32_t, uint32_t db, uint32_t dof, uint32_t len) {
                    cb_wait_front(cb_data, 1);
                    noc_async_write(get_read_ptr(cb_data), get_noc_addr_from_bank_id<true>(db, dst[w] + dof), len);
                    noc_async_writes_flushed();
                    cb_pop_front(cb_data, 1);
                });
            noc_async_write_barrier();
        } else {
            // the destination's ready: its program started (its MoE no longer reads the slot)
            volatile tt_l1_ptr uint32_t* ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sems + (1 + jb.dst) * esc::SEM_STRIDE);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(sems + (1 + jb.dst) * esc::SEM_STRIDE), 0u - 1u);
            noc_async_atomic_barrier();
            (void)fabric_set_unicast_route(hdr, jb.fabric_chip, jb.fabric_mesh);
            esc::for_my_chunks(
                tab, jb.part, jb.parts, [&](uint32_t w, uint32_t, uint32_t, uint32_t db, uint32_t dof, uint32_t len) {
                    cb_wait_front(cb_data, 1);
                    hdr->to_noc_unicast_write(
                        NocUnicastCommandHeader{get_noc_addr_from_bank_id<true>(db, dst[w] + dof)}, len);
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(get_read_ptr(cb_data), len, reinterpret_cast<uint32_t>(hdr));
                    noc_async_writes_flushed();  // (the payload and the header left L1: both may be reused)
                    cb_pop_front(cb_data, 1);
                });
            hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(jb.rx, jb.ry, sems), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
        }
        cb_pop_front(cb_tab, 1);
    }

    // 3. the senders' done (word 0)
    if (n_ready) {
        volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sems);
        noc_semaphore_wait_min(done, n_ready);
        noc_semaphore_inc(get_noc_addr(sems), 0u - n_ready);
        noc_async_atomic_barrier();
    }
    if (fabric) {
        conn.close();
    }
}
