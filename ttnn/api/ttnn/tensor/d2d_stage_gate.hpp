// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2D stage-gate word values and gate_flags bits, shared by the host
// (d2d_stream_service.hpp), the receiver kernel, and any device-side gate writer.
// Kept free of host-only includes so kernels can include it.

#pragma once

#include <cstdint>

namespace ttnn {

inline constexpr uint32_t kStageGateClosed = 0;
inline constexpr uint32_t kStageGateOpen = 1;
// gate_flags word in the per-transfer metadata.
inline constexpr uint32_t kStageGateFlagCloseOnTransit = 1u << 0;  // close the gate behind this transfer
inline constexpr uint32_t kStageGateFlagBypass = 1u << 31;         // do not gate this transfer

}  // namespace ttnn
