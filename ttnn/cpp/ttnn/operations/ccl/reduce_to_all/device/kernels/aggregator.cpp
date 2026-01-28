// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Aggregator kernel for reduce_to_all operation
// Replaces heavyweight mux with lightweight packet forwarding.
//
// Design:
// - Single direction-agnostic kernel, same code for BRISC (FWD) and NCRISC (BWD)
// - Direction implicit in fabric connection RT args (src→dst determines direction)
// - BRISC and NCRISC run on same core but access different L1 regions (via buffer_offset)
// - Each handles 2 rounds (R1 and R2) with packets_per_round packets each
//
// Memory layout per aggregator core (68KB total):
// - BRISC region (offset 0): [FWD R1 slots][FWD R2 slots] (4 slots)
// - NCRISC region (offset 4*slot_size): [BWD R1 slots][BWD R2 slots] (4 slots)
// - 2 slots per round (from 2 workers with same worker_type per link)
//
// Workflow per round:
// 1. Wait for all workers to write their packets (sem reaches packets_per_round)
// 2. Forward all packets via fabric
// 3. Reset semaphore for next round

#include "fabric/fabric_edm_packet_header.hpp"
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "api/debug/dprint.h"

#include <cstdint>

// Compile-time args
static constexpr uint32_t ct_packets_per_round = get_compile_time_arg_val(0);  // Workers per link per type (e.g., 2)
static constexpr uint32_t ct_slot_size = get_compile_time_arg_val(1);          // Total packet size (header + L + S + M)
static constexpr uint32_t ct_slots_per_round = ct_packets_per_round;           // 1 slot per worker

void kernel_main() {
    // =========================================================================
    // Parse runtime args
    // =========================================================================
    size_t arg_idx = 0;

    // Buffer addresses
    const uint32_t buffer_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t buffer_offset = get_arg_val<uint32_t>(arg_idx++);

    // Semaphore addresses (R1 and R2 for this direction)
    const uint32_t r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    DPRINT << "Aggregator: buffer_base=" << buffer_base << " offset=" << buffer_offset << ENDL();
    DPRINT << "Aggregator: r1_sem=" << r1_sem_addr << " r2_sem=" << r2_sem_addr << ENDL();

    // Build fabric connection from remaining args
    // Direction is implicit in src→dst encoding from append_fabric_connection_rt_args
    DPRINT << "Aggregator: Building fabric connection from args..." << ENDL();
    auto fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

    // Complete connection opening
    DPRINT << "Aggregator: Opening fabric connection..." << ENDL();
    fabric_connection.open();
    DPRINT << "Aggregator: Fabric connection opened!" << ENDL();

    // =========================================================================
    // Compute slot addresses
    // =========================================================================
    // Base address for this RISC's region (BRISC: offset=0, NCRISC: offset=4*slot_size)
    const uint32_t my_buffer_base = buffer_base + buffer_offset;

    // R1 slots: first `slots_per_round` slots
    // R2 slots: next `slots_per_round` slots
    const uint32_t r1_base = my_buffer_base;
    const uint32_t r2_base = my_buffer_base + (ct_slots_per_round * ct_slot_size);

    // =========================================================================
    // Round 1: Wait for workers, forward packets
    // =========================================================================
    volatile tt_l1_ptr uint32_t* r1_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r1_sem_addr);

    // Wait for all workers to deposit their R1 packets
    DPRINT << "Aggregator: Waiting for R1 semaphore (need " << ct_packets_per_round << ")..." << ENDL();
    noc_semaphore_wait_min(r1_sem_ptr, ct_packets_per_round);
    DPRINT << "Aggregator: R1 semaphore acquired, forwarding packets..." << ENDL();

    // Forward all R1 packets
    for (uint32_t slot = 0; slot < ct_slots_per_round; slot++) {
        const uint32_t slot_addr = r1_base + (slot * ct_slot_size);

        // Wait for an empty write slot in fabric
        DPRINT << "Aggregator: R1 slot " << slot << " waiting for fabric slot..." << ENDL();
        fabric_connection.wait_for_empty_write_slot();

        // Send the complete packet (header + payload already prepared by worker)
        DPRINT << "Aggregator: R1 slot " << slot << " sending..." << ENDL();
        fabric_connection.send_payload_flush_non_blocking_from_address(slot_addr, ct_slot_size);
    }
    DPRINT << "Aggregator: R1 forwarding complete!" << ENDL();

    // Reset R1 semaphore for potential future use (if operation is traced/repeated)
    noc_semaphore_set(r1_sem_ptr, 0);

    // =========================================================================
    // Round 2: Wait for workers, forward packets
    // =========================================================================
    volatile tt_l1_ptr uint32_t* r2_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r2_sem_addr);

    // Wait for all workers to deposit their R2 packets
    DPRINT << "Aggregator: Waiting for R2 semaphore (need " << ct_packets_per_round << ")..." << ENDL();
    noc_semaphore_wait_min(r2_sem_ptr, ct_packets_per_round);
    DPRINT << "Aggregator: R2 semaphore acquired, forwarding packets..." << ENDL();

    // Forward all R2 packets
    for (uint32_t slot = 0; slot < ct_slots_per_round; slot++) {
        const uint32_t slot_addr = r2_base + (slot * ct_slot_size);

        // Wait for an empty write slot in fabric
        DPRINT << "Aggregator: R2 slot " << slot << " waiting for fabric slot..." << ENDL();
        fabric_connection.wait_for_empty_write_slot();

        // Send the complete packet (header + payload already prepared by worker)
        DPRINT << "Aggregator: R2 slot " << slot << " sending..." << ENDL();
        fabric_connection.send_payload_flush_non_blocking_from_address(slot_addr, ct_slot_size);
    }
    DPRINT << "Aggregator: R2 forwarding complete!" << ENDL();

    // Reset R2 semaphore
    noc_semaphore_set(r2_sem_ptr, 0);

    // =========================================================================
    // Cleanup: Close fabric connection
    // =========================================================================
    DPRINT << "Aggregator: Closing fabric connection..." << ENDL();
    fabric_connection.close();
    DPRINT << "Aggregator: Done!" << ENDL();
}
