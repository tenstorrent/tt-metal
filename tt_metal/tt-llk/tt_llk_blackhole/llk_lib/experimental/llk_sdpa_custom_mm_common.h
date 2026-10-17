// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Apart from the unpack LLK so the compute API can pick the drain at compile time on every thread.
// Posts of 8 or more tiles, or per tile at 4 or more columns and 8 or more K tiles, reach the consumer in bursts that the
// next call's early start slows, so those calls drain first
constexpr bool sdpa_custom_mm_unpack_waits_at_end(const std::uint32_t signal_granularity, const std::uint32_t ct_dim, const std::uint32_t kt_dim)
{
    return signal_granularity >= 8 || (signal_granularity == 1 && ct_dim >= 4 && kt_dim >= 8);
}
