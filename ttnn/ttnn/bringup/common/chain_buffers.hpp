// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The kernel's DataflowBuffers for eltwise_chain elements: chain elements take the DataflowBuffer of each input(...) /
// output(...) spec as leading constructor arguments and bind it by reference (main's #60306), so a buffer must be one
// object a CB for the kernel's whole run, not a per-call temporary. bringup_chain::dfb(spec) makes the CB's object on
// its first use (from the spec that names it) and returns that object every time after:
//
//   ckl::BinaryFpu<ckl::BinaryFpuOp::Add, X_IN, R_IN>{bringup_chain::dfb(X_IN), bringup_chain::dfb(R_IN)}

#include <cstdint>
#include <new>

#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace bringup_chain {

constexpr uint32_t MAX_CBS = 64;

alignas(DataflowBuffer) inline uint8_t storage[MAX_CBS][sizeof(DataflowBuffer)];
inline bool made[MAX_CBS];

ALWI DataflowBuffer& buffer(uint32_t cb) {
    if (!made[cb]) {
        new (storage[cb]) DataflowBuffer(static_cast<uint16_t>(cb));
        made[cb] = true;
    }
    return *std::launder(reinterpret_cast<DataflowBuffer*>(storage[cb]));
}

// The DataflowBuffer of an input / output spec, or of a broadcast input's spec.
ALWI DataflowBuffer& dfb(const compute_kernel_lib::InputSpec& spec) { return buffer(spec.cb_id); }
ALWI DataflowBuffer& dfb(const compute_kernel_lib::OutputSpec& spec) { return buffer(spec.cb_id); }
ALWI DataflowBuffer& dfb(const compute_kernel_lib::BroadcastInputSpec& spec) { return buffer(spec.input_spec.cb_id); }

}  // namespace bringup_chain
