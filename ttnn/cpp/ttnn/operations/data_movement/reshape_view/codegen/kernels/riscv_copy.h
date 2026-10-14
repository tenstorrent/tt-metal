// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#if !defined(ARCH_QUASAR)
#include "ckernel.h"
#endif

namespace codegen_kernels {

// Copies with RISC stores of T, which must divide `bytes` and both addresses. Returns the last store's address.
template <typename T>
FORCE_INLINE uint32_t riscv_copy(uint32_t dst, uint32_t src, uint32_t bytes) {
    volatile tt_l1_ptr T* to = reinterpret_cast<volatile tt_l1_ptr T*>(dst);
    const volatile tt_l1_ptr T* from = reinterpret_cast<const volatile tt_l1_ptr T*>(src);
    const uint32_t count = bytes / sizeof(T);
    for (uint32_t i = 0; i < count; ++i) {
        to[i] = from[i];
    }
    return dst + bytes - sizeof(T);
}

// Word stores where all three are 4 B multiples, else halfword stores: every gated dtype is 2 or 4 B. A source the
// NOC wrote needs invalidate_l1_cache() first, once per NOC wait rather than per copy.
FORCE_INLINE uint32_t riscv_copy_elements(uint32_t dst, uint32_t src, uint32_t bytes) {
    if (((dst | src | bytes) & 3) == 0) {
        return riscv_copy<uint32_t>(dst, src, bytes);
    }
    return riscv_copy<uint16_t>(dst, src, bytes);
}

// Wormhole orders a load after a store only when both start at the same byte address, so the read-back that makes
// every earlier RISC store land in L1 before a NOC read or a CB push must hit the last store exactly.
FORCE_INLINE void drain_stores(uint32_t last_store) {
    volatile tt_l1_ptr uint16_t* last = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(last_store);
#if defined(ARCH_QUASAR)
    (void)*last;
#else
    (void)ckernel::load_blocking(last);
#endif
}

}  // namespace codegen_kernels
