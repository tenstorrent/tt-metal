// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace sdpa::streaming {

struct StateTransfer {
    enum Action : uint32_t { Save, Restore };
    enum Word : uint32_t { Operation, Slot, Numerator, Maximum, Denominator, Local, Chunks, ValidGroups, Words };
    static constexpr uint32_t page_bytes = 4096;

    template <bool fp32>
    static constexpr uint32_t plane_bytes(uint32_t plane) {
        return plane == 0 ? 131072 : plane == 1 ? 16384 : plane == 2 ? 32768 : fp32 ? 0 : 131072;
    }

    template <bool fp32>
    static constexpr uint32_t pages = (131072 + 16384 + 32768 + (fp32 ? 0 : 131072)) / page_bytes;
};

}  // namespace sdpa::streaming
