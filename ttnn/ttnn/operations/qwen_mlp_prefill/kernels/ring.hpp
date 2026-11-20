// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "named_args.hpp"

constexpr uint32_t RING_CHIPS = 4;

enum class MessageKind : uint32_t { End, Data, Increment };

struct Message {
    MessageKind kind;
    uint32_t hops;
    uint32_t bytes;
    uint32_t dst_x;
    uint32_t dst_y;
    uint32_t dst_address;
    uint32_t semaphore_address;
    uint32_t sequence;
};

constexpr uint32_t MESSAGE_BYTES = sizeof(Message);

// Index of chip `peer` among the three other chips, as seen from chip `self`.
inline uint32_t peer_index(uint32_t peer, uint32_t self) { return ((peer - self) & (RING_CHIPS - 1)) - 1; }
