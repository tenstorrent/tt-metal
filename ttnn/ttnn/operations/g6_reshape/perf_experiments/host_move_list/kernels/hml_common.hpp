// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// host_move_list shared pieces. Raw NoC calls on purpose: the bench measures the floor of a mover whose every
// transfer was decided on the host, so no helper / accessor walk sits between the arg and the NoC command.

#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"

#if HML_ZONES
#define HML_ZONE(name) DeviceZoneScopedN(name)
#else
#define HML_ZONE(name)
#endif

namespace hml {

template <uint32_t kElem>
FORCE_INLINE void copy_run(uint32_t src, uint32_t dst, uint32_t bytes) {
    if (kElem == 4 || ((src | dst | bytes) & 3) == 0) {
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        for (uint32_t w = 0; w < bytes / 4; ++w) {
            d[w] = s[w];
        }
    } else {
        volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
        volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst);
        for (uint32_t w = 0; w < bytes / 2; ++w) {
            d[w] = s[w];
        }
    }
}

}  // namespace hml
