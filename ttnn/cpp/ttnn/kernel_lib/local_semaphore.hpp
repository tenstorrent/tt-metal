// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Program semaphores used as plain L1 words between the RISC-Vs of one core, so the TRISCs can take part:
// look a semaphore up by id, post a value to it, and wait until it reaches a value. No NoC traffic.

#include <cstdint>

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif
#include "api/debug/waypoint.h"
#include "hostdev/dev_msgs.h"

namespace local_semaphore {

// Base of this launch's program semaphores. Dataflow firmware exposes it as sem_l1_base; TRISC firmware does not,
// so a TRISC reads it from the launch message. BRISC advances launch_msg_rd_ptr only after every RISC of the core
// finishes, so the pointer names this launch throughout.
inline uint32_t base() {
#if defined(COMPILE_FOR_TRISC)
    const uint32_t rd = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    const volatile tt_l1_ptr kernel_config_msg_t* cfg = &GET_MAILBOX_ADDRESS_DEV(launch[rd])->kernel_config;
    return cfg->kernel_config_base[static_cast<int>(ProgrammableCoreType::TENSIX)] +
           cfg->sem_offset[static_cast<int>(ProgrammableCoreType::TENSIX)];
#else
    return reinterpret_cast<uint32_t>(sem_l1_base[static_cast<int>(ProgrammableCoreType::TENSIX)]);
#endif
}

// L1 byte address of semaphore `id`: the same word on every RISC of the core (the runtime lays them out
// L1_ALIGNMENT apart).
inline uint32_t addr(uint32_t id) { return base() + id * L1_ALIGNMENT; }

inline void fence() { asm volatile("fence" ::: "memory"); }

// Store `value`; the fences keep this RISC's earlier L1 writes ahead of it and later accesses behind it.
inline void post(uint32_t sem_addr, uint32_t value) {
    fence();
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr) = value;
    fence();
}

// Spin until the semaphore is at least `value`. Invalidates before each poll in case the data cache is enabled.
// With the watcher enabled, a RISC stuck here shows waypoint LSMW; LSMD means it got past.
inline void wait_at_least(uint32_t sem_addr, uint32_t value) {
    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    WAYPOINT("LSMW");
    do {
        invalidate_l1_cache();
    } while (*sem < value);
    fence();
    WAYPOINT("LSMD");
}

}  // namespace local_semaphore
