// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
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

inline constexpr uint32_t high_bw_all_gather_metadata_slice_start_word =
    offsetof(HighBwAllGatherMetadataSchedule, slice_start) / sizeof(uint32_t);
inline constexpr uint32_t high_bw_all_gather_metadata_slice_count_word =
    offsetof(HighBwAllGatherMetadataSchedule, slice_count) / sizeof(uint32_t);
inline constexpr uint32_t high_bw_all_gather_metadata_final_start_word =
    offsetof(HighBwAllGatherMetadataSchedule, final_start) / sizeof(uint32_t);
inline constexpr uint32_t high_bw_all_gather_metadata_final_count_word =
    offsetof(HighBwAllGatherMetadataSchedule, final_count) / sizeof(uint32_t);
inline constexpr uint32_t high_bw_all_gather_metadata_data_valid_granularity_word =
    offsetof(HighBwAllGatherMetadataSchedule, data_valid_granularity) / sizeof(uint32_t);

static_assert(sizeof(HighBwAllGatherMetadataSchedule) == 5 * sizeof(uint32_t));
