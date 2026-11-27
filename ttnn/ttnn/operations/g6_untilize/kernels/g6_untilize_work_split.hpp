// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_untilize {

struct CoreRows {
    uint32_t core_idx;
    uint32_t start;
    uint32_t count;
};

// Contiguous balanced row range of this core in the rect_w-wide rectangle at logical (0,0).
// Must match _core_rows() in g6_untilize_program_descriptor.py (describe_plan reports it).
template <uint32_t tensor_rows, uint32_t num_cores, uint32_t rect_w>
FORCE_INLINE CoreRows core_rows() {
    constexpr uint32_t base = tensor_rows / num_cores;
    constexpr uint32_t extra = tensor_rows % num_cores;
    const uint32_t core_idx = get_relative_logical_y() * rect_w + get_relative_logical_x();
    const uint32_t start = core_idx * base + (core_idx < extra ? core_idx : extra);
    return {core_idx, start, base + (core_idx < extra ? 1u : 0u)};
}

}  // namespace g6_untilize
