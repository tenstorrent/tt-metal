// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2D stage-gate word values and flag bits, shared by host and kernels.

#pragma once

#include <cstdint>

namespace ttnn {

inline constexpr uint32_t kStageGateClosed = 0;
inline constexpr uint32_t kStageGateOpen = 1;
// Flags in the transfer's metadata.
inline constexpr uint32_t kStageGateFlagCloseOnTransit = 1u << 0;  // close the gate after this transfer passes
inline constexpr uint32_t kStageGateFlagBypass = 1u << 31;         // do not gate this transfer

}  // namespace ttnn
