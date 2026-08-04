// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

constexpr uint32_t MAX_OUTPUT_TENSORS = 3;
// write_index/read_index are padded so each counter owns its own 16-byte region, keeping the
// fabric atomic increments that target them clear of the neighbouring fields.
constexpr uint32_t UINT32_ALIGNED_COUNT = 4;

// Ring state shared between buffered_send and buffered_recv. The receiver fills it in and pushes a
// copy into the sender's landing zone; both sides then drive the counters. A non-zero num_tensors
// means the handshake has already happened.
struct alignas(32) OutputTensorInfo {
    uint32_t num_tensors;
    uint32_t page_size;
    uint32_t num_pages;
    uint32_t sender_config_l1_addr;
    uint32_t receiver_config_l1_addr;
    uint32_t base_addr[MAX_OUTPUT_TENSORS];

    // base_addr is already aligned to 8 * 4 = 32 bytes.
    uint32_t write_index[UINT32_ALIGNED_COUNT];
    uint32_t read_index[UINT32_ALIGNED_COUNT];
};
