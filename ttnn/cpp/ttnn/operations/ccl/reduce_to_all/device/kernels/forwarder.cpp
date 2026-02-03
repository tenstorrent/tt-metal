// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Forwarder kernel replaces mux with lightweight packet forwarding.
//
// Design:
// - Single direction-agnostic kernel, same code for BRISC (FWD) and NCRISC (BWD)
// - Direction implicit in fabric connection RT args (src→dst determines direction)
// - BRISC and NCRISC run on same core but access different L1 regions (via buffer_offset)
// - Non-blocking: forwards packets as soon as they arrive (no batching)
//
// Memory layout per forwarder core:
// - BRISC region (offset 0): [slot0][slot1][slot2][slot3] (R1 + R2 slots interleaved)
// - NCRISC region (offset N*slot_size): same layout for opposite direction
//
// Semaphore protocol (bit-packed):
// - Single semaphore per direction, each client signals with (1 << slot_idx)
// - forwarder polls semaphore and forwards any ready slots immediately
// - Maximum 32 clients per direction (limited by 32-bit semaphore width)
//
// Workflow:
// 1. Poll semaphore for ready slots
// 2. For each ready bit not yet sent, forward packet via fabric
// 3. Repeat until all slots sent

#include "fabric/fabric_edm_packet_header.hpp"
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"

#include <cstdint>

static constexpr uint32_t total_slots = get_compile_time_arg_val(0);  // Total slots (R1 + R2 combined)
static constexpr uint32_t slot_size = get_compile_time_arg_val(1);    // Bytes per slot (header + L + S + M)

static constexpr uint32_t all_sent_mask = (1u << total_slots) - 1;

// Maximum clients is 32 (one bit per client in 32-bit semaphore)
static_assert(total_slots <= 32, "forwarder supports at most 32 slots (limited by 32-bit semaphore)");

void kernel_main() {
    size_t arg_idx = 0;

    const uint32_t buffer_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t buffer_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    const uint32_t my_buffer_base = buffer_base + buffer_offset;

    auto fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    fabric_connection.open();

    // =========================================================================
    // Non-blocking forwarding loop
    // =========================================================================
    // Poll semaphore and forward packets as soon as they're ready.
    // Each worker signals by doing: noc_semaphore_inc(sem, 1 << slot_idx)
    // We track which slots have been sent via sent_mask.

    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    uint32_t sent_mask = 0;

    do {
        invalidate_l1_cache();

        uint32_t sem_value = *sem_ptr;
        uint32_t pending = sem_value & ~sent_mask;

        // Drain all ready slots before polling again
        while (pending != 0) {
            uint32_t slot = __builtin_ctz(pending);
            uint32_t slot_addr = my_buffer_base + (slot * slot_size);

            fabric_connection.wait_for_empty_write_slot();
            fabric_connection.send_payload_flush_non_blocking_from_address(slot_addr, slot_size);

            sent_mask |= (1u << slot);
            pending &= ~(1u << slot);
        }
    } while (sent_mask != all_sent_mask);

    fabric_connection.close();
    noc_async_full_barrier();
}
