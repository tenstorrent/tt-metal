// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Schedule derived by the reader from the trace-safe gathered-extent tensor and
// handed to the colocated writer through a circular buffer. Keep this layout
// shared: disagreement between reader and writer can deadlock the collective.
struct HighBwAllGatherMetadataSchedule {
    uint32_t slice_start;
    uint32_t slice_count;
    uint32_t final_start;
    uint32_t final_count;
    uint32_t data_valid_granularity;
};

static_assert(sizeof(HighBwAllGatherMetadataSchedule) == 5 * sizeof(uint32_t));
