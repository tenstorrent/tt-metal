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
// - Non-blocking: forwards packets as soon as they arrive (no batching)
//
// Memory layout per aggregator core:
// - BRISC region (offset 0): [slot0][slot1][slot2][slot3] (R1 + R2 slots interleaved)
// - NCRISC region (offset N*slot_size): same layout for opposite direction
//
// Semaphore protocol (bit-packed):
// - Single semaphore per direction, each client signals with (1 << slot_idx)
// - Aggregator polls semaphore and forwards any ready slots immediately
// - Maximum 32 clients per direction (limited by 32-bit semaphore width)
//
// Workflow:
// 1. Poll semaphore (with L1 invalidate for coherency)
// 2. For each ready bit not yet sent, forward packet via fabric
// 3. Repeat until all slots sent

#include "fabric/fabric_edm_packet_header.hpp"
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"

#include <cstdint>

// Compile-time args
static constexpr uint32_t ct_total_slots = get_compile_time_arg_val(0);  // Total slots (R1 + R2 combined)
static constexpr uint32_t ct_slot_size = get_compile_time_arg_val(1);    // Bytes per slot (header + L + S + M)

// Derived constants
static constexpr uint32_t ct_all_sent_mask = (1u << ct_total_slots) - 1;

// Maximum clients is 32 (one bit per client in 32-bit semaphore)
static_assert(ct_total_slots <= 32, "Aggregator supports at most 32 slots (limited by 32-bit semaphore)");

void kernel_main() {
    // =========================================================================
    // Parse runtime args
    // =========================================================================
    size_t arg_idx = 0;

    // Buffer base address for this direction's slots
    const uint32_t buffer_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t buffer_offset = get_arg_val<uint32_t>(arg_idx++);

    // Single semaphore for all slots (bit-packed: bit i set when slot i ready)
    const uint32_t sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    // Build fabric connection from remaining args
    auto fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

    fabric_connection.open();

    // =========================================================================
    // Compute base address for this RISC's region
    // =========================================================================
    const uint32_t my_buffer_base = buffer_base + buffer_offset;

    // =========================================================================
    // Non-blocking forwarding loop
    // =========================================================================
    // Poll semaphore and forward packets as soon as they're ready.
    // Each worker signals by doing: noc_semaphore_inc(sem, 1 << slot_idx)
    // We track which slots have been sent via sent_mask.

    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    uint32_t sent_mask = 0;

    DPRINT << "Aggregator started: buffer_base=" << buffer_base << ", offset=" << buffer_offset << ", sem=" << sem_addr
           << ", total_slots=" << ct_total_slots << ", all_sent_mask=" << ct_all_sent_mask << ENDL();

    {
        DeviceZoneScopedN("AGG-FORWARD-LOOP");

        do {
            invalidate_l1_cache();

            uint32_t sem_value = *sem_ptr;
            uint32_t pending = sem_value & ~sent_mask;

            // Drain all ready slots before polling again
            while (pending != 0) {
                uint32_t slot = __builtin_ctz(pending);
                uint32_t slot_addr = my_buffer_base + (slot * ct_slot_size);

                DPRINT << "Aggregator forwarding slot " << slot << " at addr " << slot_addr << ENDL();

                fabric_connection.wait_for_empty_write_slot();
                fabric_connection.send_payload_flush_non_blocking_from_address(slot_addr, ct_slot_size);

                sent_mask |= (1u << slot);
                pending &= ~(1u << slot);
            }
        } while (sent_mask != ct_all_sent_mask);
    }

    DPRINT << "Aggregator complete: sent_mask=" << sent_mask << ENDL();

    // =========================================================================
    // Cleanup
    // =========================================================================
    fabric_connection.close();
    noc_async_full_barrier();
}
