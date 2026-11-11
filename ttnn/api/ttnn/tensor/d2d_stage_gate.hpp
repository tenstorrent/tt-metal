// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2D stage-gate values, shared by host and kernels.

#pragma once

#include <cstdint>

namespace ttnn::d2d_stream {

// Value of a gate word.
enum class GateState : uint32_t {
    Closed = 0,
    Open = 1,
};

// What a transfer does at its gate. Carried in the transfer's metadata.
enum class GateMode : uint32_t {
    KeepOpen = 0,        // wait for the gate, leave it open
    CloseOnTransit = 1,  // wait for the gate, close it once this transfer passes
};

// Slot id for a transfer that skips the gate. Any slot id >= num_gates does the same.
inline constexpr uint32_t kUngatedSlot = 0xFFFFFFFF;

}  // namespace ttnn::d2d_stream
