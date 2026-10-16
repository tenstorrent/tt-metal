// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

// Data-movement helpers of the fused chunk_gdn hand-off for the two operations the Device 2.0 API has no wrapper
// for: an atomic increment and a polled wait on a plain L1 word. The producers' credit words credit[h][slot] are
// BH x NBUF words inside a CB tile, not host-allocated semaphores, so Semaphore<> (constructed from a semaphore id)
// cannot name them and Noc has no atomic-increment method. Both helpers are the primitives the 2.0 Semaphore<>
// methods compile to (up(noc, x, y, v) and wait(v)), applied to an arbitrary address.
namespace gdn_handoff {

// Atomic +1 on the word at l1_addr of core (noc_x, noc_y): one non-posted atomic on NOC_UNICAST_WRITE_VC.
FORCE_INLINE void credit_inc(const Noc& noc, uint32_t noc_x, uint32_t noc_y, uint32_t l1_addr) {
    const uint64_t dst = get_noc_addr(noc_x, noc_y, l1_addr, noc.get_noc_id());
    noc_semaphore_inc(dst, 1, noc.get_noc_id());
}

// Spin until the local word equals value, invalidating the L1 cache before every read.
FORCE_INLINE void wait_word(const CoreLocalMem<volatile uint32_t>& word, uint32_t value) {
    do {
        invalidate_l1_cache();
    } while (*word != value);
}

}  // namespace gdn_handoff
