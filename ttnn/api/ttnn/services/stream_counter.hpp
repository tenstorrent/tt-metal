// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

// A 32-bit L1 word a stream service's kernels or workers bump, and how far it moves per transfer.
struct StreamCounter {
    CoreCoord core;  // logical
    DeviceAddr addr = 0;
    uint32_t step = 0;
};

// How a transfer moves into a device: it lands in the service's FIFO, then the consuming op copies it out.
struct StreamInputCounters {
    StreamCounter arrived;
    StreamCounter consumed;
};

// How a transfer moves out of a device: the producing op writes it, the service hands it to fabric, and the
// next stage's receiver takes it.
struct StreamOutputCounters {
    StreamCounter produced;
    StreamCounter pushed;
    StreamCounter forwarded;
};

}  // namespace tt::tt_metal
