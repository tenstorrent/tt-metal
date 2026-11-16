// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "hostdevcommon/fabric_common.h"

namespace deepseek_prefill {

// This chip's index in the op's chip table (chip_position_table.hpp): its fabric chip id for a whole-mesh table,
// else 0 (a per-chip program's one-entry table).
template <bool Lookup>
inline uint32_t chip_table_index() {
    if constexpr (Lookup) {
        return reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE)->my_device_id;
    } else {
        return 0;
    }
}

}  // namespace deepseek_prefill
